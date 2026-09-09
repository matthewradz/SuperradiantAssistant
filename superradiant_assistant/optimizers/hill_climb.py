"""Tiny hill-climb / random-restart optimizer for Phase 0."""
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional

from superradiant_assistant.params import ParamSpec
from superradiant_assistant.signals import ShotSignal


def _shot_globals(shot) -> Dict[str, Any]:
    """Where a shot's parameter values actually are.

    `atom_loading_globals` is only filled when the shot file's globals group is
    literally named "Atom Loading", which is a ybclock convention -- on every
    other apparatus it is `{}`. Reading it alone meant `base` fell back to
    `p.init` on every iteration, so the climb never left its starting point:
    fifteen shots at `180 + Random(0).uniform(-36, 36)`, the same fifteen angles
    every run, independent of anything measured. `all_globals` wins on a clash
    because it is what the shot file recorded.
    """
    return {**(getattr(shot, "atom_loading_globals", None) or {}),
            **(getattr(shot, "all_globals", None) or {})}


#: 1/phi. An additive recurrence on this constant is the standard way to get a
#: sequence whose *every prefix* is spread out, which is what a shot budget that
#: may stop early needs -- a grid only covers the range once it is finished.
_INV_PHI = 0.6180339887498949


class HillClimbOptimizer:
    """Cover the range, then close on the best part of it by interpolation.

    Two phases, because the failure mode of a pure local search here was not
    subtle. Every proposal used to be `best + uniform(-10%, +10%) * span`, and
    with one knob on [0, 360] deg that confines a whole run to a 72 deg window
    around wherever it started. Three live runs of fifteen shots each stayed
    inside 150-240 deg and never saw the 10 deg peak, which the same apparatus
    had measured at 0.8133 V the same afternoon; the best any of them reported
    was 0.64 V against a 0.8 V target.

    Phase 1 spends half the budget on a phase-randomised low-discrepancy sweep
    of the *whole* range. Phase 2 is Brent's step -- parabolic interpolation
    through the best point and its neighbours, golden-section when the parabola
    is degenerate -- which converges instead of wandering.

    Simulated against this apparatus's measured curve (four peaks 90.6 deg
    apart, a 25% height envelope from beam walk, 5.5% multiplicative noise),
    fifteen shots land within 8 deg of one of the two tallest peaks in ~60% of
    runs and report above 0.80 V in ~48%. The old code did neither, ever.

    Still deliberately simple: no surrogate model, no noise model, and the point
    it calls best is a single noisy shot rather than a fitted optimum -- so on a
    curve whose form is known, sweep-and-fit still beats it.
    """

    def __init__(self, params: List[ParamSpec], target_metric: str,
                 step_frac: float = 0.1, seed: Optional[int] = None,
                 budget: int = 15, explore_frac: float = 0.5):
        self.params = params
        self.target_metric = target_metric
        self.step_frac = step_frac
        self.rng = random.Random(seed)
        self.budget = max(2, int(budget))
        # At least three, or a two-knob run spends its whole exploration on one
        # corner and calls that the global picture.
        self.explore_n = max(3, int(round(explore_frac * self.budget)))
        # The spacing of the exploration sequence is fixed, its phase is not.
        # Without this the first N angles are the same in every run, which is the
        # bug that has now bitten twice: the old code repeated fifteen angles
        # exactly, and a fixed-phase golden sequence repeated its six. On a curve
        # with several peaks that decides which peak the run converges to before
        # a single shot has been fired.
        self._phase = [self.rng.random() for _ in params]

    def suggest(self, history: List[ShotSignal]) -> Dict[str, float]:
        scored = [(v, s) for v, s in
                  ((getattr(s, self.target_metric, None), s) for s in history)
                  if v is not None]
        if len(scored) < self.explore_n:
            return self._explore(len(scored))
        return self._contract(scored)

    # ---------------------------------------------------------------- phase 1

    def _explore(self, i: int) -> Dict[str, float]:
        """The i-th point of a sequence that spreads over the full bounds.

        Not random: two random draws can land next to each other and waste a
        shot that costs minutes. Not a fixed grid either, because the budget can
        end early and a half-finished grid has a systematically unvisited half.
        """
        out: Dict[str, float] = {}
        for k, p in enumerate(self.params):
            # A different phase per parameter, so two knobs are not swept in
            # lockstep along the diagonal.
            frac = (self._phase[k] + (i + 1) * _INV_PHI) % 1.0
            out[p.name] = p.clip(p.lo + frac * (p.hi - p.lo))
        return out

    # ---------------------------------------------------------------- phase 2

    def _contract(self, scored) -> Dict[str, float]:
        """Brent's step, driven by the loop rather than driving it.

        `scipy.optimize.minimize_scalar` implements this, but it owns the
        iteration: it calls the objective itself. Here the objective is "queue a
        shot in BLACS, wait for lyse", and `run_loop` owns that -- along with the
        hooks, the operator's confirmation, the abort check and the cost cap. So
        the policy is used, not the driver: parabolic interpolation through the
        best point and its two neighbours, golden-section on the wider half of
        the bracket when the parabola is useless.
        """
        scored.sort(key=lambda t: t[0], reverse=True)
        got = _shot_globals(scored[0][1])
        base = {p.name: _coerce_scalar(got.get(p.name), fallback=p.init)
                for p in self.params}

        # Parabolic interpolation is one-dimensional. With several knobs, do it
        # along one axis at a time, holding the rest at the best point.
        j = len(scored) - self.explore_n
        axis = self.params[j % len(self.params)]

        samples = []
        for y, s in scored:
            x = _shot_globals(s).get(axis.name)
            x = None if x is None else _coerce_scalar(x, fallback=float("nan"))
            if x is not None:
                samples.append((x, float(y)))
        if len(samples) < 2:
            return {**base, axis.name: axis.clip(base[axis.name]
                                                 + (axis.hi - axis.lo) * 0.1)}

        proposal = dict(base)
        proposal[axis.name] = self._brent_step(samples, axis, j)
        return proposal

    def _brent_step(self, samples, p: ParamSpec, j: int) -> float:
        span = p.hi - p.lo
        tol = span * 1e-3                       # two shots this close measure once
        pts = sorted(samples)
        xs = [x for x, _ in pts]
        ib = max(range(len(pts)), key=lambda i: pts[i][1])
        xb = pts[ib][0]

        def fresh(x):
            x = p.clip(x)
            return x if all(abs(x - q) > tol for q in xs) else None

        # 1. Parabola through the best point and its neighbours in x. This is the
        #    step that actually finds a smooth peak in a couple of shots; the old
        #    optimizer had no notion of it and could only wander.
        if 0 < ib < len(pts) - 1:
            (x0, y0), (x1, y1), (x2, y2) = pts[ib - 1], pts[ib], pts[ib + 1]
            num = ((x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0))
            den = ((x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0))
            if abs(den) > 1e-15:
                xv = x1 - 0.5 * num / den
                # Only inside the bracket: a vertex outside it means the three
                # points are not describing a peak, and extrapolating on that is
                # how a short sweep invented a dark level once already.
                if x0 < xv < x2:
                    got = fresh(xv)
                    if got is not None:
                        return got

        # 2. Golden section on the wider side of the bracket. Also the escape
        #    when the best point sits at the edge of what has been sampled --
        #    then there is no bracket yet and the step goes outward.
        lo_side = xb - (xs[ib - 1] if ib > 0 else p.lo)
        hi_side = (xs[ib + 1] if ib < len(pts) - 1 else p.hi) - xb
        for cand in ((xb + hi_side * (1 - _INV_PHI),
                      xb - lo_side * (1 - _INV_PHI))
                     if hi_side >= lo_side else
                     (xb - lo_side * (1 - _INV_PHI),
                      xb + hi_side * (1 - _INV_PHI))):
            got = fresh(cand)
            if got is not None:
                return got

        # 3. Everything is within tol of something already measured: the search
        #    has converged. Re-measure the best point -- a repeat at the optimum
        #    is the one thing that tells noise from signal, and it is what the
        #    reported best most needs.
        return p.clip(xb)


def _coerce_scalar(v, fallback: float) -> float:
    """Turn lists / arrays / weird globals into a single float, else fallback."""
    if v is None:
        return fallback
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        try:
            return float(v[0])
        except Exception:
            return fallback
    try:
        return float(v)
    except (TypeError, ValueError):
        return fallback
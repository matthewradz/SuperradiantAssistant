"""The optimizer has to cover the range and then converge on it.

Two live failures are encoded here as regression tests:

  * fifteen shots at `180 + Random(0).uniform(-36, 36)` -- the same fifteen
    angles every run, because `atom_loading_globals` is `{}` on this apparatus
    and the base fell back to `p.init`;
  * a fixed-phase exploration sequence, which repeated its own six angles every
    run and so decided which peak the search converged to before firing a shot.

The last section simulates the whole loop against this apparatus's measured
curve, because "does it actually find the peak" is not something the unit checks
above can answer.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superradiant_assistant.optimizers.hill_climb import (
    HillClimbOptimizer, _shot_globals,
)
from superradiant_assistant.params import ParamSpec

bad = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


class Shot:
    """A shot the way this apparatus writes them: globals in `all_globals`."""

    def __init__(self, angle, mean, legacy=False):
        self.all_globals = {} if legacy else {"waveplate_angle": angle}
        self.atom_loading_globals = {"waveplate_angle": angle} if legacy else {}
        self.CHAN1_mean = mean

    def metric_value(self, name):
        return getattr(self, name, None)


PARAM = ParamSpec(name="waveplate_angle", lo=0.0, hi=360.0, init=180.0)
FROZEN = [180.0, 204.8, 198.57, 174.28, 162.64, 180.81, 173.16, 200.43,
          165.84, 178.31, 186.0, 209.38, 180.34, 164.29, 198.42]


def angles(opt, hist, n):
    out = []
    for _ in range(n):
        a = opt.suggest(hist)["waveplate_angle"]
        out.append(a)
        hist = hist + [Shot(a, 0.5)]          # flat metric: no gradient to follow
    return out


print("\n=== 1. the frozen trajectory, reproduced from first principles ===")
rng = random.Random(0)
old = [180.0] + [round(min(max(180.0 + rng.uniform(-0.1, 0.1) * 360.0, 0.0),
                          360.0), 2) for _ in range(14)]
print(f"    {old}")
check(old == FROZEN,
      "180 + Random(0).uniform(-36,36) reproduces the angles seen on hardware")
got = [round(a, 2) for a in angles(HillClimbOptimizer([PARAM], "CHAN1_mean",
                                                      budget=15, seed=0), [], 15)]
check(got != FROZEN, "the optimizer no longer walks that path")
check(max(got) - min(got) > 180.0,
      f"and it spans the range ({min(got):.0f}-{max(got):.0f} deg, was 162-209)")

print("\n=== 2. globals are read from wherever the shot put them ===")
check(_shot_globals(Shot(198.42, 0.7675))["waveplate_angle"] == 198.42,
      "all_globals is read (this apparatus)")
check(_shot_globals(Shot(198.42, 0.7675, legacy=True))["waveplate_angle"] == 198.42,
      "atom_loading_globals still works (ybclock)")
check(_shot_globals(object()) == {}, "a shot with neither yields nothing, not a crash")

print("\n=== 3. exploration covers the whole range, not a window ===")
ex = angles(HillClimbOptimizer([PARAM], "CHAN1_mean", budget=20, seed=4), [], 10)
buckets = {int(a // 90) for a in ex}
print(f"    first 10: {[round(a) for a in ex]}")
check(len(buckets) == 4, f"all four quadrants of [0,360] are visited ({buckets})")
gaps = sorted(a for a in ex)
worst = max(b - a for a, b in zip(gaps, gaps[1:]))
check(worst < 90.0, f"no gap wider than a period ({worst:.0f} deg)")

print("\n=== 4. two runs no longer explore the same angles ===")
a1 = angles(HillClimbOptimizer([PARAM], "CHAN1_mean", budget=15), [], 6)
a2 = angles(HillClimbOptimizer([PARAM], "CHAN1_mean", budget=15), [], 6)
check(a1 != a2, "the exploration phase is phase-randomised")
s1 = angles(HillClimbOptimizer([PARAM], "CHAN1_mean", budget=15, seed=11), [], 6)
s2 = angles(HillClimbOptimizer([PARAM], "CHAN1_mean", budget=15, seed=11), [], 6)
check(s1 == s2, "an explicit seed still reproduces a run, for tests")

print("\n=== 5. parabolic interpolation lands on the vertex ===")
# Three points on a clean parabola peaking at 100 deg; the next proposal should
# be the vertex, not a fixed step.
def para(a):
    return 1.0 - ((a - 100.0) / 100.0) ** 2


opt = HillClimbOptimizer([PARAM], "CHAN1_mean", budget=6, seed=2, explore_frac=0.5)
hist = [Shot(a, para(a)) for a in (60.0, 90.0, 130.0)]
nxt = opt.suggest(hist)["waveplate_angle"]
print(f"    from 60/90/130 -> proposes {nxt:.3f} deg (true vertex 100.000)")
check(abs(nxt - 100.0) < 1e-6, "the vertex is found exactly on a parabola")

print("\n=== 6. it converges instead of wandering ===")
opt = HillClimbOptimizer([PARAM], "CHAN1_mean", budget=15, seed=5)
hist = []
for _ in range(15):
    a = opt.suggest(hist)["waveplate_angle"]
    hist.append(Shot(a, para(a)))
tail = [s.all_globals["waveplate_angle"] for s in hist[-4:]]
spread = max(tail) - min(tail)
print(f"    last four: {[round(a, 2) for a in tail]}  spread {spread:.3f} deg")
check(spread < 5.0, "the last four shots sit within a few degrees of each other")
best = max(hist, key=lambda s: s.CHAN1_mean)
check(abs(best.all_globals["waveplate_angle"] - 100.0) < 2.0,
      f"and on the peak ({best.all_globals['waveplate_angle']:.2f} deg)")

print("\n=== 7. against this apparatus's measured curve ===")
# Ground truth from today: peak 0.8133 V at 10 deg (the 0-45 sweep), peak
# 0.610 V at 195.6 deg (the optimize runs), period 90.6 deg and fast axis
# 9.5 deg (the 2026-08-12 360 deg fit). Peak height varies with a
# 360-deg-period envelope -- the beam walks as the plate turns.
PERIOD, THETA0 = 90.60, 9.50
MID, DEPTH = (0.8133 + 0.610) / 2, (0.8133 - 0.610) / 2
NOISE, FLOOR = 0.055, 0.010


def truth(t):
    height = MID + DEPTH * math.cos(math.radians(t - THETA0))
    return height * math.cos(math.pi * (t - THETA0) / PERIOD) ** 2


def simulate(seed, budget=15):
    r = random.Random(seed)
    o = HillClimbOptimizer([PARAM], "CHAN1_mean", budget=budget, seed=seed)
    h = []
    for _ in range(budget):
        a = o.suggest(h)["waveplate_angle"]
        h.append(Shot(a, truth(a) * (1 + r.gauss(0, NOISE)) + r.gauss(0, FLOOR)))
    b = max(h, key=lambda s: s.CHAN1_mean)
    return b.all_globals["waveplate_angle"], b.CHAN1_mean


def to_tall_peak(a):
    """Distance to the nearest of the two tallest peaks, 9.5 and 100.1 deg."""
    return min(abs((a - 9.5 + 180) % 360 - 180), abs((a - 100.1 + 180) % 360 - 180))


runs = [simulate(s) for s in range(300)]
near = sum(1 for a, _ in runs if to_tall_peak(a) < 8.0) / len(runs)
over = sum(1 for _, v in runs if v > 0.80) / len(runs)
mean = sum(v for _, v in runs) / len(runs)
print(f"    300 simulated 15-shot runs: near a tall peak {100*near:.0f}%, "
      f"reported >0.80 V {100*over:.0f}%, mean best {mean:.4f} V")
# The three live runs scored 0% and 0% on these two. Thresholds are set well
# below the measured simulation so ordinary variation does not fail the suite.
check(near > 0.40, "it reaches one of the two tallest peaks in most runs")
check(over > 0.25, "and clears the 0.80 V target in a good fraction of them")
check(mean > 0.70, f"mean reported best is well above the 0.64 V it used to get")

print("\n=== 8. degenerate inputs still return something legal ===")
o = HillClimbOptimizer([PARAM], "CHAN1_mean", budget=15, seed=1)
first = o.suggest([])["waveplate_angle"]
check(PARAM.lo <= first <= PARAM.hi, f"an empty history proposes in range ({first:.1f})")
nom = o.suggest([Shot(200.0, None)])["waveplate_angle"]
check(PARAM.lo <= nom <= PARAM.hi, "a history with no readable metric is survivable")
flat = HillClimbOptimizer([PARAM], "CHAN1_mean", budget=8, seed=1)
h = [Shot(a, 0.5) for a in (10.0, 20.0, 30.0, 40.0)]
for _ in range(6):
    v = flat.suggest(h)["waveplate_angle"]
    check_ok = PARAM.lo <= v <= PARAM.hi
    h.append(Shot(v, 0.5))
check(check_ok, "a perfectly flat metric never proposes out of bounds")

print("\n" + "=" * 60)
print(f"  {len(bad)} failure(s)" if bad else "  ALL CHECKS PASSED")
for b_ in bad:
    print("    - " + b_)
sys.exit(1 if bad else 0)

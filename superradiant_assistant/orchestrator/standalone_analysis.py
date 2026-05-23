"""Standalone analysis — reads HDF5 directly, fits data, returns numbers for the LLM.

No plotting here. The user only ever sees plots from lyse's multishot analysis.
This runs silently behind the scenes to generate the report.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import List, Optional

import numpy as np


def _read_shots(data_root: Path):
    from superradiant_assistant.interfaces.hdf5_reader import list_shots, read_shot
    signals = []
    for p in list_shots(data_root):
        try:
            signals.append(read_shot(p))
        except Exception:
            pass
    return signals


def analyze_resonance_sweep(
    data_root: Path,
    sweep_param: str,
    plot_ratio: str,
) -> dict:
    """Read HDF5 shots, fit Lorentzian dip, return findings for LLM report.
    No display — lyse handles the plot."""
    signals = _read_shots(data_root)
    if not signals:
        return {"found": False, "error": "no shots found"}

    xs, ys = [], []
    for sig in signals:
        x = sig.all_globals.get(sweep_param)
        if x is None:
            continue
        try:
            x = float(x)
        except (TypeError, ValueError):
            continue

        if "/" in plot_ratio:
            num_name, den_name = plot_ratio.split("/", 1)
            num = getattr(sig, num_name.strip(), None)
            den = getattr(sig, den_name.strip(), None)
            if num is None or den is None or den == 0:
                continue
            if math.isnan(num) or math.isnan(den):
                continue
            y = num / den
        else:
            y = getattr(sig, plot_ratio.strip(), None)
            if y is None or math.isnan(y):
                continue

        xs.append(x)
        ys.append(y)

    if not xs:
        return {"found": False, "error": f"no valid data for {sweep_param}"}

    pairs = sorted(zip(xs, ys))
    xs_arr = np.array([p[0] for p in pairs])
    ys_arr = np.array([p[1] for p in pairs])

    min_ratio  = float(np.min(ys_arr))
    min_idx    = int(np.argmin(ys_arr))
    resonance_freq     = xs_arr[min_idx]
    fit_resonance_freq = None
    fit_linewidth      = None

    # Lorentzian dip fit (silent — for the report only)
    if min_ratio < 0.7:
        try:
            from scipy.optimize import curve_fit

            def _dip(x, x0, w, A, offset):
                return offset - A / (1 + ((x - x0) / w) ** 2)

            span = xs_arr[-1] - xs_arr[0]
            p0 = [resonance_freq, span / 6, 1 - min_ratio, 1.0]
            bounds = ([xs_arr[0], 1e-9, 0, 0], [xs_arr[-1], span, 1, 2])
            popt, _ = curve_fit(_dip, xs_arr, ys_arr, p0=p0, bounds=bounds, maxfev=3000)
            fit_resonance_freq = float(popt[0])
            fit_linewidth      = abs(float(popt[1]))
            resonance_freq     = fit_resonance_freq
        except Exception:
            pass  # fall back to minimum point

    return {
        "found":                  min_ratio < 0.5,
        "resonance_freq_mhz":     float(resonance_freq),
        "fit_resonance_freq_mhz": fit_resonance_freq,
        "fit_linewidth_mhz":      fit_linewidth,
        "min_ratio":              min_ratio,
        "n_points":               len(xs),
    }


def analyze_larmor_calibration(data_root: Path) -> dict:
    """Read HDF5 shots, fit Ramsey fringes, return correction for LLM report.
    No display — lyse handles the plot."""
    from scipy.optimize import curve_fit

    signals = _read_shots(data_root)
    if not signals:
        return {"correction_hz": None, "error": "no shots found"}

    times, sz_vals, rf_detunings = [], [], []
    for sig in signals:
        T = sig.all_globals.get("calibration_precession_time")
        if T is None:
            continue
        try:
            T = float(T)
        except (TypeError, ValueError):
            continue
        n3, n4 = sig.Neta_3, sig.Neta_4
        if n3 is None or n4 is None or (n3 + n4) == 0:
            continue
        sz = (n3 - n4) / (n3 + n4)
        if math.isnan(sz):
            continue
        times.append(T)
        sz_vals.append(sz)
        try:
            rf_detunings.append(float(sig.all_globals.get("rf_larmor_detuning", 100)))
        except (TypeError, ValueError):
            rf_detunings.append(100.0)

    if len(times) < 5:
        return {"correction_hz": None, "error": "not enough data points"}

    pairs = sorted(zip(times, sz_vals))
    t_arr  = np.array([p[0] for p in pairs])
    sz_arr = np.array([p[1] for p in pairs])
    larmor_detuning_khz = np.mean(rf_detunings) / 1e3

    def _ramsey(t, contrast, frequency, offset, phi):
        return contrast * np.cos(2 * np.pi * frequency * t + phi) + offset

    correction_hz = None
    fit_freq_khz  = None
    try:
        p0 = [0.4, larmor_detuning_khz, 0.0, 0.0]
        popt, _ = curve_fit(_ramsey, t_arr, sz_arr, p0=p0, maxfev=5000)
        fit_freq_khz  = abs(popt[1])
        correction_hz = -fit_freq_khz * 1000 + larmor_detuning_khz * 1e3
    except Exception:
        pass

    return {
        "correction_hz":      correction_hz,
        "fit_freq_khz":       fit_freq_khz,
        "larmor_detuning_hz": float(np.mean(rf_detunings)),
        "n_points":           len(times),
    }

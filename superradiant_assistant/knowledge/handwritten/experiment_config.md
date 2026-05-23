# Experiment Configuration — Sequences, Analysis Scripts, and Globals

This document describes the exact sequences, analysis scripts, and globals
the assistant is allowed to use. Do not use anything outside this list.

---

## Sequences

### calibrate_larmor_frequency_clean.py
- **Path**: `sequences/assistant_sequences/calibrate_larmor_frequency_clean.py`
- **Use case**: Larmor frequency calibration
- **What it does**: Ramsey sequence that applies RF pulses and measures spin precession to calibrate the Larmor frequency.
- **Key globals to set**: `rf_larmor_frequency`, `calibration_precession_time`
- **Pair with**: `calibrate_larmor_frequency_clean.py` analysis script

### clock_transition_RABI.py
- **Path**: `sequences/assistant_sequences/clock_transition_RABI.py`
- **Use case**: Clock resonance finding
- **What it does**: Rabi spectroscopy sequence. Shines clock light at `clock_pi_resonance_frequency_list` and measures population transfer via Neta_5/Neta_4.
- **Key globals to set**: `clock_pi_resonance_frequency_list` (sweep around `clock_pi_resonance_frequency` ± 1 MHz)
- **Pair with**: `improved_cost_clean.py` with `parameter_str = "clock_pi_resonance_frequency_list"` and `y_op_string = "Neta_5/Neta_4"`

---

## Analysis Scripts

### improved_cost_clean.py
- **Path**: `analysis/scripts/meta/improved_cost_clean.py`
- **Use case**: General sweep plotting (resonance finding, parameter scans)
- **How it works**:
  1. Loads dataframe: `df = data()`
  2. Filters shots by `delta_duration` (groups repeated scans)
  3. Collects Neta_1..Neta_9 from `cavity_scan_analysis` results
  4. Evaluates `y_op_string` against the Neta dict (e.g. `"Neta_5/Neta_4"`)
  5. Collects `parameter_str` column as x-axis
  6. Averages y values over repeated x values, computes error bars
  7. Plots with `plt.errorbar`, optionally fits (linear or Lorentzian)
- **Fields to edit**:
  - `parameter_str`: name of the swept global, e.g. `"clock_pi_resonance_frequency_list"`
  - `y_op_string`: y-axis expression, e.g. `"Neta_5/Neta_4"` or `"Neta_3"`
- **Supported y expressions**: any arithmetic combination of Neta_1..Neta_9, e.g. `"Neta_5/Neta_4"`, `"Neta_3"`, `"(Neta_3+Neta_4)/2"`

### calibrate_larmor_frequency_clean.py
- **Path**: `analysis/scripts/meta/calibrate_larmor_frequency_clean.py`
- **Use case**: Larmor frequency calibration
- **How it works**: Fits Ramsey fringe data from Neta values, extracts the Larmor frequency offset, and prints the correction needed for `rf_larmor_frequency`.
- **Output**: Prints "Larmor Correction Needed: X Hz" — add X to current `rf_larmor_frequency`

---

## Globals the Assistant Can Edit

| Global | Type | Range | Description |
|---|---|---|---|
| `z_bias_field_loading` | float | -0.110 to -0.090 | Z-axis bias field during loading |
| `y_bias_field_loading` | float | 0.0 to 0.5 | Y-axis bias field during loading |
| `green_mot_frequency` | float | 47.5 to 49.5 MHz | Green MOT frequency on handoff to cavity |
| `lattice_loading_frequency` | float | 48.0 to 50.0 MHz | Green lattice frequency on handoff |
| `TD_loading` | bool | True/False | Enable 2D transverse loading |
| `calibration_precession_time` | float | 0.1 to 20 ms | Ramsey wait time for Larmor calibration |
| `rf_larmor_frequency` | float | 10000 to 11000 Hz | Larmor frequency; add correction after calibration |
| `clock_pi_resonance_frequency_list` | float | clock_pi_resonance_frequency ± 1 MHz | Clock light frequency per shot in RABI sequence |

---

## Use Case Recipes

### Find clock resonance
1. Sequence: `clock_transition_RABI.py`
2. Sweep: `clock_pi_resonance_frequency_list` across ~3 kHz in ~250 Hz steps
3. Analysis: `improved_cost_clean.py` with `parameter_str="clock_pi_resonance_frequency_list"`, `y_op_string="Neta_5/Neta_4"`
4. On resonance: Neta_5/Neta_4 dips from ~1 to ~0.2–0.3 (Lorentzian)

### Calibrate Larmor frequency
1. Sequence: `calibrate_larmor_frequency_clean.py`
2. Set `calibration_precession_time` (start with 2 ms)
3. Analysis: `calibrate_larmor_frequency_clean.py`
4. Read correction from output, add to `rf_larmor_frequency`

### Optimize atom loading
1. Sequence: any sequence with atom loading (e.g. `clock_transition_RABI.py` or `calibrate_larmor_frequency_clean.py`)
2. Sweep/optimize: `z_bias_field_loading`, `y_bias_field_loading`, `green_mot_frequency`, `lattice_loading_frequency`
3. Analysis: `improved_cost_clean.py` with `y_op_string="Neta_2"` (atom number after loading)
4. Goal: maximize Neta_2 above threshold

### Answer code questions
- The assistant can trace through sequence files and subsequences in the knowledge base
- For questions like "where is X global used?", search the subsequence files
- For "what does this loading step do?", read the function body from the subsequence

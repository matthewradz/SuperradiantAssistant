# MIT Vuletic Lab Overview (Phase 1)

This lab runs neutral 171-Yb atoms inside a high-finesse optical cavity for spin-squeezing and clock-transition experiments.

## Control stack
- **labscript suite**: experiment definition (labscriptlib/ybclock).
- **runmanager**: parameter management; "globals" are organized into groups such as `Atom Loading`, `Squeezing`, `Cavity Scan Parameters`, `OpticalClock`.
- **BLACS**: shot queuing and execution.
- **lyse**: deterministic analysis after each shot.

## Key analyses (Phase 1)
- `cavity_scan_analysis.py`: produces `Neta_1..Neta_N` atom-number estimates based on how many exp_cavity.scan() there are per sequence, fit qualities `chi_square_*`, `r_sq_*`, fit parameters `kappa`, `gamma`, etc. Outputs are stored as attrs on `results/cavity_scan_analysis`.
- `extract_photon_arrival_times.py`: produces processed_arrivals_ch_0..N for each scan which are mapped relative to cavity resonance frequency giving Neta_N.
- `improved_cost_vs_single_parameter_Qi.py`: multi-shot analysis for plotting sweeps. This is typically used to plot some function of Neta, say Neta_4/Neta_3, versus a swept parameter, say clock frequency.

## Phase-1 wrapper conventions
- The wrapper does NOT modify hardware-control code (sequences, devices).
- It only chooses values for runmanager globals and decides which existing analysis scripts to apply.
- All results read by the orchestrator come from HDF5 `results/` and `globals/`.

## Cavity scan metrics per shot (Neta_1..Neta_5)
In `recycling_on_clock_transition_release_recap_FPGA_yellow_FNC.py` there are 5 exp_cavity.scan() calls:
- **Neta_1, Neta_2**: from `measure_and_prepare_atoms` (loading check, before cooling)
- **Neta_3**: after isCool / release_recapture (post-cooling check)
- **Neta_4**: in recycling loop — BEFORE the clock pulse (yellow_doublepass_switch.go_high/go_low)
- **Neta_5**: in recycling loop — AFTER the clock pulse
- **Neta_5/Neta_4**: clock transfer ratio. Off resonance ≈ 1. On resonance ≈ 0.2–0.3 (Lorentzian dip).
- **delta_duration**: lyse multi-shot grouping variable — selects which shots belong to the same sequence for analysis.

## Clock frequency globals
- `clock_pi_resonance_frequency`: best estimate of clock resonance (MHz), e.g. 82.46521
- `clock_pi_resonance_frequency_list`: per-shot swept value (set to a single float when sweeping across shots)
- Unit conversions: 1 kHz = 0.001 MHz; 250 Hz = 0.00025 MHz; 3 kHz = 0.003 MHz
- Typical sweep: center ≈ 82.465 MHz, range ±1.5 kHz, step 250 Hz → 13 points

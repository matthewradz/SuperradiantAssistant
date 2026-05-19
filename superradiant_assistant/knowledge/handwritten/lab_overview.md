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
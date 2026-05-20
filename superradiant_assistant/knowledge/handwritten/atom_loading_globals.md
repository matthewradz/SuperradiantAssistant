# Atom Loading globals (selected)

These live under `globals/Atom Loading` in each shot HDF5 and are set in runmanager.
They are consumed by `loading_subsequences_8_5_24.py` (the primary loading subsequence),
called via `load_from_oven_to_optical_lattice_with_blue_abs_image()` in the cooling sequences.

## Blue MOT globals
Used inside `blue_mot()` in `loading_subsequences_8_5_24.py`:

- `LS_blue_mot_duration`: duration of the blue MOT loading stage (seconds).
- `LS_blue_mot_intensity`: intensity of the 399 nm blue MOT light (arb. units, 0–1).
- `LS_green_mot_intensity`: intensity of the green light during blue MOT (two-color MOT for Doppler pre-cooling).
- `x_bias_blue_mot`: x-axis bias magnetic field during blue MOT loading; sets trapping position.
- `y_bias_blue_mot`: y-axis bias magnetic field during blue MOT loading.
- `z_bias_blue_mot`: z-axis bias magnetic field during blue MOT loading.
- `yellow_doublepass_freq`: frequency (MHz) of the yellow 578 nm doublepass AOM, set via FPGA_DDS9 channel 1 at the start of the blue MOT stage.

Note: `mot_coil_current` (gradient coil) is hardcoded to 9.1 A in the subsequence and is NOT a runmanager global.

## Lattice / handoff globals
- `green_mot_frequency`: frequency of the green MOT on handoff to the cavity (relative MHz); typical 48.9, safe range 48–50.
- `z_bias_field_loading`: z-axis bias field during lattice loading; typical −0.101, safe range −0.110 to −0.090.
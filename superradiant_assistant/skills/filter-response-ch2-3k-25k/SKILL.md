---
name: filter-response-ch2-3k-25k
description: Characterize CH2 low-pass filter frequency response from 3 kHz to 25 kHz at 1.0 V amplitude.
apparatus: testbench
tags: filter,frequency-response,awg,scope,multishot
---
# Filter Response Measurement (3 kHz to 25 kHz)

Measure the frequency response of a filter placed on AWG CH2 relative to AWG CH1 reference up to 25 kHz.

## Hardware Configuration & Wiring
- **AWG CH1** -> Scope **CHAN1** (unfiltered reference drive).
- **AWG CH2** -> **Filter** -> Scope **CHAN2** (filtered signal).

## Procedure

1. **Verify Globals**:
   - `amplitude` = 1.0 V
   - `phase` = 0.0 deg

2. **Configure Lyse Analysis Routines**:
   - **Singleshot**: `filter_ch1_ch2_analysis.py` (calculates FFT line amplitudes and ratio `CHAN2_amp / CHAN1_amp`).
   - **Multishot**: `filter_ch2_multishot_20260811.py` (plots scatter plot of ratio vs frequency up to 25 kHz and fits low-pass curve).

3. **Execute Sweep**:
   - Sequence: `filter_scan_ch2.py`
   - Swept parameter: `sine_frequency`
   - Sweep mode: `explicit`
   - Start frequency: `3000` Hz
   - End frequency: `25000` Hz
   - Number of points: `23`

## Expected Result
- **Passband (3 kHz - 13 kHz)**: Transfer ratio ~0.80–1.00 (-2 dB to 0 dB).
- **-3 dB Cutoff**: $f_c \approx 14.6\text{ kHz}$ (transition between 14 kHz and 15 kHz).
- **Stopband (15 kHz - 25 kHz)**: Attenuation falls rapidly to -22 dB at 20 kHz and -40 dB at 25 kHz.

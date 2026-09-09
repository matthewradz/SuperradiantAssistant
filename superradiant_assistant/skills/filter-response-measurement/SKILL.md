---
name: filter-response-measurement
description: Measure filter frequency response up to 25 kHz by sweeping AWG sine frequency and plotting CH2/CH1 ratio in lyse.
apparatus: testbench
tags: filter, frequency response, awg, scope, lyse
---
# Filter Response Measurement Procedure

This procedure measures the frequency response of an electronic filter connected to AWG CH2 by sweeping the output sine frequency up to 25 kHz and comparing the amplitude on CH2 against CH1 (direct connection).

## Experimental Setup & Wiring
- AWG CH1 connected directly to Scope CH1 (reference channel).
- AWG CH2 connected through the filter to Scope CH2.
- Scope channels CHAN1 and CHAN2 enabled.

## Code & Analysis Routines
1. **Sequence Script:** `Cesium/Sequences/filter_scan_ch2.py`
   - Drives RigolDG1022 CH1 and CH2 at `sine_frequency` (Hz) with `amplitude` (V) and `phase`.
2. **Singleshot Analysis:** `Cesium/singleshot_routines/filter_ch2_singleshot.py`
   - Calculates dominant FFT spectral line amplitude for CHAN1 and CHAN2.
   - Saves results `CHAN1_amp`, `CHAN2_amp`, `transfer_ratio` (CH2/CH1), and `transfer_db`.
3. **Multishot Analysis:** `Cesium/multishot_routines/filter_ch2_multishot.py`
   - Reads `sine_frequency` and `transfer_ratio` across all shots in the sequence.
   - Fits a low-pass / bandpass filter model to extract cutoff frequency $f_c$ and plots amplitude ratio vs frequency up to 25 kHz.

## Execution Steps
1. Load lyse routines:
   - Singleshot: `filter_ch2_singleshot.py`
   - Multishot: `filter_ch2_multishot.py`
2. Load sequence `filter_scan_ch2.py` in runmanager.
3. Run explicit sweep on global `sine_frequency` from 100 Hz to 25000 Hz with 26 points.
4. Check lyse figure 'Filter Response CH2/CH1' for passband range and -3 dB cutoff frequency.


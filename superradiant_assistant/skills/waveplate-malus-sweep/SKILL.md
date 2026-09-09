---
name: waveplate-malus-sweep
description: Sweep the PRM1/MZ8 waveplate angle and measure the transmission through the PBS against angle. Fits Malus with a free phase and period, which is how the plate order is confirmed.
apparatus: cesium
tags: waveplate,polarisation,malus,prm1z8,tdc001,scope,rotation,sweep
---
# Waveplate angle sweep (transmission through a PBS)

Rotate the half-wave plate and record how much light the PBS passes. The curve is
Malus, `cos^2(2*theta)`, so its period in mount angle is 90 deg.

## Hardware

    fibre -> collimator -> BS -> waveplate (PRM1/MZ8 on TDC001) -> PBS -> detector -> scope CHAN1

- `waveplate` is a `PRM1Z8` under `waveplate_controller` (`TDC001`) in the Cesium
  connection table. Units are **degrees**; the worker converts to encoder counts.
- The mount turns at **10 deg/s** and each shot blocks until it arrives, so a
  point costs about `angle_step / 10` seconds on top of the shot itself. A
  72-point 360 deg sweep took 155 s driving the hardware directly, longer through
  BLACS.

## Procedure

1. **Confirm the pieces exist** — `get_runmanager_globals`, `list_scripts`,
   `get_lyse_routines`. Needed:
   - global `waveplate_angle`, limits 0 to 360 deg
   - sequence `waveplate_rotate.py`
   - **singleshot** `waveplate_transmission.py`
   - **multishot** `waveplate_malus_multishot.py`

2. **Load both lyse routines** with `set_lyse_routines`. Without the singleshot
   the shots run, the waveplate turns, and every sweep reports
   `shots_ran_but_no_results` — there is no number to plot.

3. **Collapse every other global to a scalar.** `run_sweep` refuses to engage when
   another global is still multi-valued: 2 requested points against a 3-value
   `phase` is 6 shots, and it says so. Note `get_runmanager_globals` only reports
   globals declared in config.json, so the offending one may be invisible; read
   the refusal message, it names the count.

4. **Sweep** with `run_sweep`, `sweep_param='waveplate_angle'`, sequence
   `waveplate_rotate.py`.
   - **Sweep at least 120 deg**, and 360 deg if the question is about the optic
     rather than about one setting. See the trap below.
   - 5 deg steps resolve the curve well; 10 deg is enough for a cutoff-style
     answer.

5. **Read back** with `read_shot_results`, and report `CHAN1_mean` against
   `waveplate_angle`. The multishot routine prints the fit and saves the figure.

## Expected result (measured 2026-08-12, 72 points over 360 deg)

    period   90.60 deg     <- 90 deg confirms a HALF-wave plate
    theta0    9.50 deg     <- fast axis, on the mount scale
    A       681.10 mV
    dark    -15.74 mV      <- the detector's own offset; optical extinction is good
    rms      45.82 mV      = 6.7% of A

Four minima, all at -20 mV. Contrast about 42:1.

## Traps, each of which cost real time

- **The maximum is not at 0 deg.** The mount's zero is a mechanical mark with no
  relation to the input polarisation; here the fast axis is at 9.5 deg. Fit
  `cos^2(2*(theta - theta0))` with `theta0` free. Fitting without it left a 10%
  residual whose sign pattern (-,+,+,+,+,-) was the missing phase.
- **A short sweep invents a dark level.** With `theta0 = 9.5 deg` the true minimum
  is at 55 deg, so a 0-45 deg sweep never reaches it. Fitting that range reported
  `dark = 44 mV` and `contrast 14.8:1`; the full 360 deg sweep gave `-20 mV` and
  `42:1`. The short fit was extrapolating its own slope.
- **Do not use the scope's own measurements.** On the MSO2202A `:MEAS:ITEM?` does
  not exist and `:MEAS:VAVG?` returns 9.9e37 or a stale value often enough to be
  useless. Compute from the samples in `/data/traces`, which is what the
  singleshot routine does.
- **A flat trace on one ADC code is normal**, not a fault: a steady beam has less
  noise than one code once the vertical range is right. What is a fault is
  `clipped=1`, where the mean is the screen edge. The multishot routine drops
  those points.
- **The peaks are not all the same height** — 823, 789, 767, 688 mV over the four
  periods, while the four minima agree to a millivolt. That asymmetry is a
  360-deg-period envelope about 12% deep: the plate is not quite normal to the
  beam, so rotating it walks the beam. It explains a third of the residual.
- **The rest of the residual is the laser, not the apparatus.** Scatter is
  38.6 mV near the peaks and 7.1 mV at extinction, i.e. proportional to the
  signal: multiplicative intensity noise of 5-6%, plus a drift of +46 mV over
  155 s. Averaging longer does not remove it. **A second detector on the
  beamsplitter's other port into CHAN2 does** — the singleshot routine already
  saves `transmission = CHAN1_mean / CHAN2_mean` when CHAN2 is captured.

## If the scope misbehaves

The RigolScope worker now refuses to hand over a trace it cannot vouch for. If a
shot fails with one of these, the message is the diagnosis:

- *"the scope never triggered"* — `arm_mode='SINGLE'` on a DC level. `:SING`
  waits for a real trigger event and ignores the sweep mode even when the panel
  says AUTO, and a steady level never crosses the threshold. Cesium's connection
  table uses `arm_mode='RUN'`.
- *"identical to the previous shot's"* — the scope handed back a buffer it had
  already given us; nothing was measured.
- *"STILL CLIPPED"* — the signal is off screen at the coarsest scale.

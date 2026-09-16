---
name: hardware-safety
description: Rules for changing experiment parameters and queueing shots on real hardware.
tags: safety, hardware, limits, confirmation, audit
---

# Working with real hardware

Three tools change the state of the apparatus: `set_runmanager_global`,
`run_sweep`, and `run_optimization`. Everything else is read-only.

## What is enforced in code

Range limits come from `config.json` and are checked inside the tool, not by the
schema. A parameter write outside its declared interval is refused, and so is a
sweep whose endpoints fall outside the range of the parameter it sweeps. These
checks cannot be argued around — rephrasing the request does not change them.

Parameters with no declared min/max cannot be range-checked, so scalar writes to
them are refused rather than passed through. The exception is `TD_loading`, a
boolean with no meaningful interval.

`run_sweep` and `set_runmanager_global` require operator confirmation, and every
call is written to the audit log with its reason. This is why `reason` is a
required argument: it is the record of why the apparatus was changed.

## What is not enforced

Some sweeps cannot be bounded because neither the parameter nor a base scalar
declares a range in `config.json` — `clock_pi_resonance_frequency_list` is the
current example. The confirmation prompt says so explicitly when this happens.
For those, the numbers you propose are the only check that exists. Derive them
from a value you actually read, not from memory.

## Habits that matter

**Read before you write.** `get_runmanager_globals` first, so a correction is
applied to the real current value rather than an assumed one.

**Sweep width is a cost.** Every point is a shot. A 101-point sweep is 101 real
experiments. Prefer coarse-then-fine over one enormous scan.

**Report numbers, not impressions.** "Loading improved" is not a result;
"Neta_2 716.7, was 690.2" is.

**A refusal is information.** If a write is refused as out of range, the value is
wrong or the limit in `config.json` is wrong. Say which you think it is and let
the operator decide — do not look for another route to the same write.

## When the apparatus is not running

The read and write tools need the labscript GUIs (runmanager, BLACS, lyse) open.
When they aren't, those tools return an error rather than pretending to succeed;
a fake success would suggest the machine had been reconfigured when it hadn't.
Analysis of already-recorded data works offline.

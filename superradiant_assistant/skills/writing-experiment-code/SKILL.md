---
name: writing-experiment-code
description: How to write a shot sequence or a lyse analysis routine for this apparatus, and how its data is laid out. Load before writing either.
apparatus: cesium
tags: shot, sequence, labscript, lyse, analysis, traces, scope, template
---

# Writing code for the Cesium apparatus

Load this before calling `write_shot` or `write_analysis`. It holds the shapes
that actually work here, which general labscript knowledge does not give you.

## Imports — the code guard is a whitelist, not a blacklist

Anything not listed below is refused, and the refusal costs a full round trip:
the whole file has to be regenerated from scratch, because `write_shot` and
`write_analysis` take the complete source, not a diff.

Shot scripts may import: `numpy`, `math`, `labscript`,
`labscriptlib.Cesium.connection_table`.

Analysis routines may import: `numpy`, `math`, `h5py`, `matplotlib` /
`matplotlib.pyplot`, `scipy` / `scipy.signal` / `scipy.optimize`, `lyse`,
`analysislib.Cesium.analysis_utils.data_classes`.

**Always forbidden, in either kind of file** — these are the actual escape
hatches, not a style preference: `os`, `sys`, `pathlib`, `subprocess`, `shutil`,
`socket`, `requests`, `urllib`, `importlib`, `pickle`, `ctypes`, `tempfile`,
`glob`, `multiprocessing`, `threading`, `pty`, `signal`, `builtins`, plus the
names `eval`, `exec`, `compile`, `__import__`, `open`, `input`, `globals`,
`locals`, `vars`, `getattr`, `setattr`, `delattr`, `breakpoint`, `memoryview`.

This means: no `pathlib.Path` and no `os.path` for building a sibling filename
next to a shot — do it with plain string ops on `path` instead:

```python
norm = path.replace('\\', '/')
shot_dir, shot_file = norm.rsplit('/', 1)
shot_id = shot_file.rsplit('.', 1)[0]
fig.savefig(shot_dir + '/' + shot_id + '__something.png', dpi=130)
```

## Hardware

- **RigolDG1022** — two-channel function generator. CH1 and CH2 are independent.
- **scope1** — Rigol DS1104Z. It captures **whichever channels are displayed on
  its front panel**, nothing else. A channel that is switched off produces no
  data no matter what the shot asks for.
- Wiring in use: AWG CH1 → scope CHAN1, AWG CH2 → scope CHAN2.

## Where the data is

`/data/traces` is a **group inside the shot's HDF5 file**. It is not a folder on
disk, and there are no .npy or .csv files anywhere. The group holds one
structured dataset per capture, with columns `('t', 'CHAN1', 'CHAN2')` — `t` in
seconds, channels in volts, typically 1200 points.

Read it like this:

```python
with h5py.File(path, 'r') as f:
    traces = f['/data/traces']
    data = traces[list(traces.keys())[0]][()]
t   = data['t'].astype(float)
ch1 = data['CHAN1'].astype(float)
```

A shot that was compiled but never executed by BLACS has **no** `/data/traces`.
That is not an analysis fault — raise a clear error and let lyse show it.

## Shot sequence skeleton

```python
import numpy as np
from labscript import start, stop, add_time_marker
from labscriptlib.Cesium.connection_table import ConnectionTable

if __name__ == '__main__':
    ConnectionTable()
    start()
    add_time_marker(t=0, label='start')
    RigolDG1022.program_sine(frequency=[sine_frequency],
                             amplitude=[amplitude], phase=[phase], channel=1)
    RigolDG1022.program_sine(frequency=[sine_frequency],
                             amplitude=[amplitude], phase=[phase], channel=2)
    stop(5.0)
```

Rules that are not optional:

- `ConnectionTable()` first — device names like `RigolDG1022` do not exist until
  it has run.
- Every argument is a **list**, even for a single value.
- Keep the file **pure ASCII**. labscript copies each imported script into the
  shot file with a bare `open().read()`, which decodes with the system codepage
  (GBK here) and aborts compilation on any UTF-8 byte.
- Sweeps come from runmanager globals; the shot itself never loops.

## lyse analysis skeleton

lyse `exec`s the routine top to bottom. There is no `__main__` block and no
wrapper function — code inside `if __name__ == '__main__':` never runs.

```python
import numpy as np
import h5py
from lyse import Run, path

run = Run(path)
with h5py.File(path, 'r') as f:
    traces = f.get('/data/traces')
    if traces is None or len(traces) == 0:
        raise ValueError('shot has no traces: it never ran')
    data = traces[list(traces.keys())[0]][()]

ch1 = data['CHAN1'].astype(float)
run.save_result('CHAN1_vpp', float(np.ptp(ch1)))
```

- Results **must** go through `run.save_result(name, value)` on a `Run` object.
  `lyse.routine_storage` is scratch space shared between shots and has no
  `save_result`; using it saves nothing and reports no error.
- Do not use `raise SystemExit` to skip a shot. lyse catches `Exception`, and
  `SystemExit` escapes the handler and trips an unrelated crash.
- Whatever you save becomes a metric the agent can read back and threshold on.
  A plot alone is not a result.

## Measuring amplitude properly

For frequency-response work use the **amplitude of the dominant spectral line**,
not peak-to-peak or std. Once a signal is attenuated, noise and DC offset
dominate those two and the response looks flat when it is not.

```python
v = ch1 - ch1.mean()
spec = np.abs(np.fft.rfft(v * np.hanning(len(v))))
k = int(np.argmax(spec[1:]) + 1)          # skip DC
amp = spec[k] / len(v) * 4.0              # Hanning halves coherent gain
```

Frequency resolution is the record length: 1200 points at 5 µs is 6 ms, so bins
are ~167 Hz apart. A measured peak snapping to a multiple of that is resolution,
not error.

## Plotting across a sweep: use a multishot routine

A single-shot routine sees one shot. A curve against the swept parameter needs
every shot, so it belongs in a **multishot** routine — `write_analysis` with
`kind='multishot'`.

This distinction is not cosmetic. Sweep-wide plotting inside a single-shot
routine runs once per shot: 26 shots means 26 full dataframe fetches and 26
figure redraws, which locks up the lyse GUI and leaves no usable plot.

```python
import numpy as np
import matplotlib.pyplot as plt
from lyse import data

df = data()                      # one row per shot, all results
ratio_col = ('filter_analysis', 'transfer_ratio')   # (routine, result)

# data() returns EVERY shot in the lyse box, not just this sweep. Keep the
# newest sequence that actually has results -- see the rule below.
usable = df[df[ratio_col].notna()]
if len(usable):
    latest = usable['sequence'].max()
    df = df[df['sequence'] == latest]

freqs  = df['sine_frequency'].values
ratios = df[ratio_col].values

ok = np.isfinite(freqs) & np.isfinite(ratios)
order = np.argsort(freqs[ok])
freqs, ratios = freqs[ok][order], ratios[ok][order]

plt.figure('Filter response')
plt.clf()
plt.plot(freqs / 1e3, ratios, 'o-')
plt.axhline(1 / np.sqrt(2), color='r', ls='--', label='-3 dB')
plt.xlabel('Frequency (kHz)')
plt.ylabel('CH2 / CH1')
plt.title(f'Filter response\n{latest} ({len(freqs)} shots)')
plt.grid(True)
plt.legend()
```

Rules for a multishot routine:

- `data()` gives the dataframe. There is no `path` and no `Run` — those are
  single-shot concepts and there is no one shot to open.
- Result columns are `(routine_name, result_name)` tuples, because two routines
  may save the same name.
- **Filter to one sequence before plotting.** lyse's dataframe is cumulative:
  every shot ever added to the box stays there, so `data()` hands back previous
  sweeps too. Plotting all of it put five or six points at each frequency, from
  different days, and shifted the fitted cutoff from 14.91 kHz to 15.10 kHz.
  `plt.clf()` does not help — the stale points are in the data, not on the
  canvas. Filter on the `sequence` column (a timestamp, one per `engage`);
  `sequence_index` is the same thing as an integer.
  Pick the newest sequence **that has results**, not simply the newest: rows
  arrive one at a time while a sweep is being analysed, and a sweep whose
  single-shot routine produced nothing would otherwise blank the window.
- **Put the sequence in the title.** A window left open from an earlier run is
  indistinguishable from a fresh one otherwise.
- **Do not let a failed fit masquerade as a measurement.** Falling back to the
  `p0` guess and labelling it `fc=14.00 kHz` reads exactly like a result. Check
  you have more points than parameters, and say so on the figure when you do
  not fit.
- **Never call `plt.show()`** in any lyse routine. lyse displays the figure
  itself; `show()` blocks the analysis thread.
- Naming a figure (`plt.figure('Filter response')`) is fine and keeps it in one
  window across reruns.

## Existing files worth reading

- `Sequences/filter_scan.py` — drives the same sine on both channels
- `Sequences/Test_Speed.py` — single channel, the oldest working sequence
- `singleshot_routines/scope_test.py` — saves `CHANn_amp`, `CHANn_vpp`,
  `CHANn_peak_freq`, `transfer_ratio`, `transfer_db`


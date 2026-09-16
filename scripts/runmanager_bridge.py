"""Subprocess bridge: called by runmanager_iface.py (superradiant env) to talk to
runmanager.remote (ybclock_3_11_24 env). Reads a JSON command from argv[1], executes
it against the running runmanager GUI, and prints a JSON result to stdout."""
import os
import sys
import json
import numpy as _np

class _SafeEncoder(json.JSONEncoder):
    """Convert numpy types to native Python for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        return super().default(obj)

sys.path.insert(0, os.environ.get('LABSCRIPT_SUITE_USERLIB')
                or os.path.expanduser('~/labscript-suite/userlib'))

cmd = json.loads(sys.stdin.read())
action = cmd['action']

# Importing runmanager.remote costs ~4.5 s and connecting a Client another
# ~3.9 s, and this bridge is a fresh interpreter for every single call. Paying
# that on a request that only talks to lyse made a one-line query take eight
# seconds and the agent look hung. Connect lazily, so each action pays only for
# what it actually uses.
_client = None


def get_client():
    global _client
    if _client is None:
        import runmanager.remote as rm
        _client = rm.Client()
    return _client


class _LazyClient:
    """Stand-in so existing `client.<method>(...)` calls connect on first use."""

    def __getattr__(self, name):
        return getattr(get_client(), name)


client = _LazyClient()


def _lyse_request(payload, timeout=20):
    """Send one request to the running lyse and return its reply.

    The default zmq timeout is a few seconds, which is not enough: changing the
    routine list stops the subprocess of every routine being removed and starts
    one for each being added, so the reply can easily arrive after the default
    deadline and look like lyse being unreachable.
    """
    from labscript_utils.ls_zprocess import zmq_get
    from labscript_utils.labconfig import LabConfig
    port = int(LabConfig().get('ports', 'lyse'))
    return zmq_get(port, 'localhost', payload, timeout=timeout)

if action == 'set_globals':
    globals_dict = cmd['globals']
    # Values that look like numpy/Python expressions should be passed raw
    # so runmanager evaluates them (e.g. np.linspace(...) becomes an array of shots)
    raw_dict = {}
    expr_dict = {}
    for k, v in globals_dict.items():
        if isinstance(v, str) and any(tok in v for tok in ('np.', 'linspace', 'arange', 'array')):
            expr_dict[k] = v
        else:
            raw_dict[k] = v
    if raw_dict:
        client.set_globals(raw_dict, raw=False)
    if expr_dict:
        client.set_globals(expr_dict, raw=True)
    print(json.dumps({'ok': True}))

elif action == 'engage':
    client.engage()
    print(json.dumps({'ok': True}))

elif action == 'set_globals_and_engage':
    globals_dict = cmd['globals']
    raw_dict = {}
    expr_dict = {}
    for k, v in globals_dict.items():
        if isinstance(v, str) and any(tok in v for tok in ('np.', 'linspace', 'arange', 'array')):
            expr_dict[k] = v
        else:
            raw_dict[k] = v
    if raw_dict:
        client.set_globals(raw_dict, raw=False)
    if expr_dict:
        client.set_globals(expr_dict, raw=True)
    if client.error_in_globals():
        print(json.dumps({'ok': False, 'error': 'runmanager reports errors in globals'}))
        sys.exit(1)
    client.engage()
    print(json.dumps({'ok': True}))

elif action == 'set_labscript_file':
    client.set_labscript_file(cmd['path'])
    print(json.dumps({'ok': True}))

elif action == 'get_labscript_file':
    print(json.dumps({'ok': True, 'result': client.get_labscript_file()}))

elif action == 'set_shot_output_folder':
    client.set_shot_output_folder(cmd['path'])
    print(json.dumps({'ok': True}))

elif action == 'error_in_globals':
    print(json.dumps({'ok': True, 'result': client.error_in_globals()}))

elif action == 'get_globals':
    print(json.dumps({'ok': True, 'result': client.get_globals()}, cls=_SafeEncoder))

elif action == 'n_shots':
    print(json.dumps({'ok': True, 'result': client.n_shots()}))

elif action == 'list_globals_files':
    # Which globals files the running GUI has open, and the groups in each.
    import runmanager
    files = cmd.get('files') or []
    out = {}
    for fn in files:
        try:
            out[fn] = {g: runmanager.get_globalslist(fn, g)
                       for g in runmanager.get_grouplist(fn)}
        except Exception as e:
            out[fn] = {'error': str(e)}
    print(json.dumps({'ok': True, 'result': out}, cls=_SafeEncoder))

elif action == 'lyse_get_routines':
    kind = cmd.get('kind', 'singleshot')
    print(json.dumps({'ok': True,
                      'result': _lyse_request('get %s routines' % kind)},
                     cls=_SafeEncoder))

elif action == 'lyse_set_routines':
    # Ask the running lyse to use these routines. Requires the request that
    # stock lyse does not implement; the server's own refusal text is passed
    # straight back rather than being smoothed into something like success.
    kind = cmd.get('kind', 'singleshot')
    print(json.dumps({'ok': True, 'result': _lyse_request({
        '%s_routines' % kind: cmd['routines'],
        'replace': cmd.get('replace', True),
    })}, cls=_SafeEncoder))

elif action == 'compile_check':
    # Compile a sequence in this throwaway process and return the traceback if
    # it fails. runmanager pipes compile errors to its GUI output pane and never
    # writes them to a log, so this is the only way the agent can see the actual
    # reason a shot failed to build rather than guessing from "no files appeared".
    import os, shutil, tempfile, traceback, runpy, io as _io, contextlib
    import runmanager, labscript

    labscript_file = cmd['labscript_file']
    globals_file = cmd['globals_file']

    tmpdir = tempfile.mkdtemp(prefix='compilecheck_')
    try:
        gcopy = os.path.join(tmpdir, 'globals.h5')
        shutil.copy(globals_file, gcopy)
        # Collapse any swept global to a single value: this checks whether the
        # sequence compiles, not how many shots it expands to.
        for group in runmanager.get_grouplist(gcopy):
            for name in runmanager.get_globalslist(gcopy, group):
                try:
                    runmanager.set_expansion(gcopy, group, name, '')
                except Exception:
                    pass
        out = os.path.join(tmpdir, 'check.h5')
        # A shot's own print() statements would otherwise land on stdout and
        # corrupt the JSON this bridge returns, so a sequence that compiled
        # cleanly came back unparseable while a broken one parsed fine.
        captured = _io.StringIO()
        with contextlib.redirect_stdout(captured):
            runmanager.make_run_file_from_globals_files(labscript_file, [gcopy], out)
            labscript.labscript_init(out, labscript_file=labscript_file)
            runpy.run_path(labscript_file, run_name='__main__')
        print(json.dumps({'ok': True, 'result': {
            'compiles': True, 'error': None,
            'output': captured.getvalue()[-500:]}}))
    except Exception:
        print(json.dumps({'ok': True, 'result': {
            'compiles': False, 'error': traceback.format_exc(limit=12)}}))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

elif action == 'new_global':
    # Create a global in the globals HDF5 file. runmanager.remote has no
    # command for this — only set_globals on names that already exist — so it
    # is done through runmanager's own file API instead.
    import runmanager
    fn, group, name = cmd['file'], cmd['group'], cmd['name']
    existing = runmanager.get_globalslist(fn, group)
    if name in existing:
        print(json.dumps({'ok': False,
                          'error': f"global '{name}' already exists in group '{group}'"}))
        sys.exit(1)
    runmanager.new_global(fn, group, name)
    runmanager.set_value(fn, group, name, str(cmd['value']))
    if cmd.get('units'):
        runmanager.set_units(fn, group, name, str(cmd['units']))
    runmanager.set_expansion(fn, group, name, '')
    print(json.dumps({'ok': True, 'result': {
        'name': name, 'group': group, 'file': fn,
        'value': runmanager.get_value(fn, group, name),
    }}, cls=_SafeEncoder))

else:
    print(json.dumps({'ok': False, 'error': f'unknown action: {action}'}))
    sys.exit(1)

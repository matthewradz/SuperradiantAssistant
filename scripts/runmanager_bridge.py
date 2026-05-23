"""Subprocess bridge: called by runmanager_iface.py (superradiant env) to talk to
runmanager.remote (ybclock_3_11_24 env). Reads a JSON command from argv[1], executes
it against the running runmanager GUI, and prints a JSON result to stdout."""
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

sys.path.insert(0, r'C:\Users\radzi\Documents\labscript-suite\labscript-suite\userlib')
import runmanager.remote as rm

cmd = json.loads(sys.stdin.read())
client = rm.Client()
action = cmd['action']

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

elif action == 'set_shot_output_folder':
    client.set_shot_output_folder(cmd['path'])
    print(json.dumps({'ok': True}))

elif action == 'error_in_globals':
    print(json.dumps({'ok': True, 'result': client.error_in_globals()}))

elif action == 'get_globals':
    print(json.dumps({'ok': True, 'result': client.get_globals()}, cls=_SafeEncoder))

elif action == 'n_shots':
    print(json.dumps({'ok': True, 'result': client.n_shots()}))

else:
    print(json.dumps({'ok': False, 'error': f'unknown action: {action}'}))
    sys.exit(1)

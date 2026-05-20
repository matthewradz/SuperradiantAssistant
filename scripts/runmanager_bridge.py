"""Subprocess bridge: called by runmanager_iface.py (superradiant env) to talk to
runmanager.remote (ybclock_3_11_24 env). Reads a JSON command from argv[1], executes
it against the running runmanager GUI, and prints a JSON result to stdout."""
import sys
import json

sys.path.insert(0, r'C:\Users\radzi\Documents\labscript-suite\labscript-suite\userlib')
import runmanager.remote as rm

cmd = json.loads(sys.stdin.read())
client = rm.Client()
action = cmd['action']

if action == 'set_globals':
    client.set_globals(cmd['globals'])
    print(json.dumps({'ok': True}))

elif action == 'engage':
    client.engage()
    print(json.dumps({'ok': True}))

elif action == 'set_globals_and_engage':
    client.set_globals(cmd['globals'])
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
    print(json.dumps({'ok': True, 'result': client.get_globals()}))

elif action == 'n_shots':
    print(json.dumps({'ok': True, 'result': client.n_shots()}))

else:
    print(json.dumps({'ok': False, 'error': f'unknown action: {action}'}))
    sys.exit(1)

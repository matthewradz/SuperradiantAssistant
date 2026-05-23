"""Subprocess bridge: talks to the running lyse GUI via its ZMQ WebServer."""
import sys
import json

sys.path.insert(0, r'C:\Users\radzi\Documents\labscript-suite\labscript-suite\userlib')

from labscript_utils.ls_zprocess import ZMQClient
from labscript_utils.labconfig import LabConfig

cmd = json.loads(sys.stdin.read())
action = cmd['action']

lc = LabConfig()
port    = lc.getint('ports', 'lyse', fallback=42519)
host    = 'localhost'
timeout = lc.getfloat('timeouts', 'communication_timeout', fallback=60)

client = ZMQClient()

if action == 'hello':
    resp = client.get(port, host, 'hello', timeout=timeout)
    print(json.dumps({'ok': True, 'result': str(resp)}))

elif action == 'add_shot':
    filepath = cmd['filepath']
    resp = client.get(port, host, {'filepath': filepath}, timeout=timeout)
    print(json.dumps({'ok': True, 'result': str(resp)}))

elif action == 'add_shots':
    results = []
    for filepath in cmd['filepaths']:
        try:
            resp = client.get(port, host, {'filepath': filepath}, timeout=timeout)
            results.append({'path': filepath, 'ok': True})
        except Exception as e:
            results.append({'path': filepath, 'ok': False, 'error': str(e)})
    n_ok = sum(1 for r in results if r['ok'])
    print(json.dumps({'ok': True, 'result': n_ok}))

else:
    print(json.dumps({'ok': False, 'error': f'unknown action: {action}'}))
    sys.exit(1)

"""Reuse current SWE/MemAgent runners; replace only the model HTTP transport."""
import hashlib,json,os,runpy,sys
from pathlib import Path
R=Path(__file__).resolve().parent;G=R.parent/'gemma_full_20261017'
sys.path[:0]=[str(R),str(G/'harness'),str(R.parents[1]/'mini-swe-agent/src'),str(R.parents[1]/'external/SWE-bench')]
import tebench.backends as backends
from goal_client import GoalSessionBackend
old=backends.create
def create(config,method):
    if method!='dram_backed_kv':return old(config,method)
    release=Path(os.environ['GOAL_RELEASE'])
    config['goal_runtime']={'release':str(release),'manifest_sha256':hashlib.sha256((release/'runtime-manifest.json').read_bytes()).hexdigest()}
    return GoalSessionBackend(config)
backends.create=create
import tebench.methods as methods
snapshot=methods.policy_snapshot
def actual_snapshot(method,config):
    if method!='dram_backed_kv':return snapshot(method,config)
    startup=json.loads(Path(os.environ['GOAL_STARTUP']).read_text())
    return {'method':'goal_query','protocol':'kv_session_v1','selection_contract':startup['selection_contract'],
            'engine_overrides':startup['effective_server_config'],'runtime_manifest_sha256':hashlib.sha256((Path(os.environ['GOAL_RELEASE'])/'runtime-manifest.json').read_bytes()).hexdigest()}
methods.policy_snapshot=actual_snapshot
bench=sys.argv.pop(1)
runpy.run_path(str(G/bench/('run_swe.py' if bench=='swebench' else 'run_memagent.py')),run_name='__main__')

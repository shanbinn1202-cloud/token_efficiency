"""Lightweight source/config identities, excluding weights, caches and secrets."""
import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent
GOAL_FILES=['grade_goal.py','job_time.py','goal_server.py','goal_client.py','rewriter_service.py','run_goal_harness.py','validate_goal_gpu.py','stage_goal_runtime.py','goal_stage.py','goal_formal_job.sh','validation_job.sh']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def goal():
 files=[R/n for n in GOAL_FILES]+[R/'selective-prefix'/n for n in ['serve.py','install.py','INSTALL_RELEASE.json','SOURCE_SHA256SUMS.txt']]
 files+=list((R/'selective-prefix/source-overlay').rglob('*.py'))
 files+=list((R.parent/'gemma_full_20261017/harness').rglob('*.py'))
 files+=list((R/'goal/formal').rglob('*.yaml'))
 files+=[p for p in (R/'goal/formal').rglob('*manifest.json') if not p.is_symlink()]
 return {str(p):digest(p) for p in files if '__pycache__' not in p.parts}
if __name__=='__main__':Path(sys.argv[1]).write_text(json.dumps(goal(),indent=2))

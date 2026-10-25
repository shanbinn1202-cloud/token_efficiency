"""Official resumable grading; never changes inference or throughput files."""
import os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;G=R.parent/'gemma_full_20261017';model,bench=sys.argv[1:]
bundle=R/'goal/formal'/model/bench
out=bundle/('result' if bench=='swebench' else 'memagent/result')
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(G/'harness'),str(R.parents[1]/'mini-swe-agent/src'),str(R.parents[1]/'external/SWE-bench')]))
script=G/bench/('grade_swe.py' if bench=='swebench' else 'grade_memagent.py')
subprocess.run([sys.executable,str(script),'--bundle',str(bundle),'--output',str(out)],env=env,check=True)
if bench=='memagent':subprocess.run([sys.executable,str(G/'memagent/judge_memagent.py'),'--bundle',str(bundle)],env=env,check=True)

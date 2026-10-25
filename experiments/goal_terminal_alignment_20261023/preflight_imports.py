"""CPU-only imports of actual runtime/scaffold entrypoints before GPU work."""
import importlib,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;T=D.parents[1];G=D.parent/'gemma_full_20261017'
sys.path[:0]=[str(D),str(G/'harness'),str(T/'mini-swe-agent/src'),str(T/'external/SWE-bench')]
for name in ['litellm','datasets','pandas','docker','tenacity','goal_client','goal_server','experiments.benchmarks32_20261004.telemetry','baselines.common.environment','baselines.common.runner','experiments.memagent20_20261003.run']:
 importlib.import_module(name)
 print(json.dumps({'import':name,'passed':True}),flush=True)

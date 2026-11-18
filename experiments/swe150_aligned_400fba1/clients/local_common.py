"""Frozen Mini-SWE and local ARM64 tooling shared by inference and SWE-bench grading."""
import json
import os
from pathlib import Path
import re
import shlex
import sys
from types import MethodType

ROOT=Path(__file__).resolve().parents[1]
FORMAL=Path('/projects/bcjw/ssun4/swe64_standard_20261115/formal-code-20261115')
sys.path[:0]=[str(FORMAL),str(FORMAL/'mini242'),str(FORMAL/'archive-reference'),str(FORMAL/'vendor/SWE-bench')]
os.environ.update(AGENTDIET_MODE='off',REQUEST_TIMEOUT='4400',MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT='1',
                  MSWEA_GLOBAL_CONFIG_DIR=str(FORMAL/'agent-global-config'),LITELLM_LOCAL_MODEL_COST_MAP='True')
from baselines.common.environment import create_apptainer_environment
from minisweagent.environments.singularity import SingularityEnvironment

SRC=ROOT/'source/examples/swebench'


def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,default=str)+'\n')
    tmp.replace(path)


def dataset():
    rows=json.loads((ROOT/'assets/dataset-first150.json').read_text())
    pins=dict(line.split() for line in (SRC/'verified-150.txt').read_text().splitlines())
    assert len(rows)==150 and [r['instance_id'] for r in rows]==list(pins),'Task list/order does not match pinned Verified-150'
    return rows,pins


def tool_execute(self,action,cwd='',*,timeout=None):
    # The upstream Singularity wrapper launches bash -c. exec replaces it with
    # the same login-shell invocation as Docker and activates the ARM environment
    # explicitly (the isolated Apptainer home lacks Docker's profile setup).
    script='source /opt/miniconda3/bin/activate testbed || exit $?\n'+action['command']
    return SingularityEnvironment.execute(self,{**action,'command':'exec bash -lc '+shlex.quote(script)},cwd,timeout=timeout)


def task_env(image,commit):
    env=create_apptainer_environment(image,timeout=120)
    try:
        env.execute=MethodType(tool_execute,env)
        env.config.env.update(PAGER='cat',MANPAGER='cat',LESS='-R',PIP_PROGRESS_BAR='off',TQDM_DISABLE='1')
        if not re.fullmatch('[0-9a-f]{40}',commit):
            raise ValueError('Unrecognized base commit')
        check=env.execute({'command':f'git reset --hard {commit} && git diff --exit-code && git diff --cached --exit-code && test "$CONDA_DEFAULT_ENV" = testbed && shopt -q login_shell && git rev-parse HEAD && command -v python'})
        if check['returncode'] or check.get('exception_info') or commit not in check['output']:
            raise RuntimeError(f'Base checkout or testbed login failed: {check}')
        return env,check
    except BaseException:
        env.cleanup()
        raise

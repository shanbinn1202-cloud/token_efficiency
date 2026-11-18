"""Write the Harbor 0.21.0 job file of one Terminal-Bench 2.0 run (`harbor run -c <file> -y`).

The settings come from the model profile's template (job-<profile>.json) and the task list (tasks.json); the
script fills in only the job name, the jobs directory, the port of the loopback endpoint and the task names, and
refuses anything else:

- --profile: qwen3.6-27b or gemma4-31b-it (the template: Terminus 2's context limit is the profile's benchmark
  server context, 262,144 tokens for both profiles);
- --context: 65536 writes the job at 65,536 tokens instead, the context of the earlier dry runs (the server must
  then run at --max-model-len 65536: close_sessions.py check refuses another context); the default is the
  profile's 262,144;
- --tasks: trial16, smoke32 (every task with a 900-s agent timeout) or all (the 89 tasks);
- --port: the loopback port Harbor's agents connect to (1024-65535);
- --jobs-dir: an absolute path; Harbor writes the job to <jobs dir>/<job name>, which must not exist (Harbor
  resumes an existing job directory instead of starting a new run);
- --job-name: letters, digits, '.', '_' and '-' (at most 100);
- --stock: the stock vLLM control: Terminus 2 also sends chat_template_kwargs enable_thinking false with every
  call (the stock server keeps vLLM's default template switches; this runtime's servers render thinking off
  themselves).

Harbor's pydantic models ignore keys they do not know, so a misspelt key would silently change nothing; the
script checks every key of the job against the fields of Harbor 0.21.0's models (HARBOR_FIELDS) and the
template's values against the settings of the runs (check_template).

Usage:
  python3 -I -B examples/terminalbench/make_job.py --profile qwen3.6-27b --tasks trial16 --port 20000 \\
      --jobs-dir /abs/path/runs/tb-obs-1/jobs --job-name tb-obs-1 --out /abs/path/runs/tb-obs-1/job.json
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
TASKS_FILE = 'tasks.json'
PROFILES = {'qwen3.6-27b': 'job-qwen3.6-27b.json', 'gemma4-31b-it': 'job-gemma4-31b-it.json'}
# Terminus 2's context limit per profile: the --max-model-len of the profile's Terminal-Bench servers.
CONTEXT = {'qwen3.6-27b': 262144, 'gemma4-31b-it': 262144}
# --context: the profile's context (the default) or 65,536, the context of the earlier dry runs.
CONTEXT_CHOICES = (262144, 65536)
TASK_SETS = ('trial16', 'smoke32', 'all')
AGENT_TIMEOUT_SEC = 900.0          # every task of trial16 and smoke32 (their task.toml; timeout_multiplier 1.0)
STOCK_CALL_KWARGS = {'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
JOB_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z')
TASK_NAME = re.compile(r'[a-z0-9][a-z0-9-]{0,63}\Z')
# Harbor's hosted_vllm checks (src/harbor/llms/utils.py, v0.21.0): the name and the four model_info fields.
HOSTED_VLLM_NAME = re.compile(r'hosted_vllm/[A-Za-z0-9._-]{1,64}\Z')
HOSTED_VLLM_MODEL_INFO = ('max_input_tokens', 'max_output_tokens', 'input_cost_per_token', 'output_cost_per_token')
# The fields of Harbor 0.21.0's models at tag v0.21.0: JobConfig, RetryConfig and DatasetConfig
# (src/harbor/models/job/config.py), AgentConfig, EnvironmentConfig and VerifierConfig
# (src/harbor/models/trial/config.py), and the named parameters of Terminus2.__init__ (src/harbor/agents/
# terminus_2/terminus_2.py; Terminus 2 also takes **kwargs, so a misspelt agent kwarg is ignored too).
HARBOR_FIELDS = {
    'job': ('job_name', 'jobs_dir', 'n_attempts', 'install_only', 'timeout_multiplier', 'agent_timeout_multiplier',
            'verifier_timeout_multiplier', 'agent_setup_timeout_multiplier', 'environment_build_timeout_multiplier',
            'debug', 'n_concurrent_trials', 'quiet', 'retry', 'environment', 'verifier', 'metrics', 'agents',
            'datasets', 'tasks', 'artifacts', 'extra_instruction_paths', 'source_jobs'),
    'retry': ('max_retries', 'include_exceptions', 'exclude_exceptions', 'wait_multiplier', 'min_wait_sec',
              'max_wait_sec'),
    'environment': ('type', 'import_path', 'force_build', 'delete', 'cpu_enforcement_policy',
                    'memory_enforcement_policy', 'override_cpus', 'override_memory_mb', 'override_storage_mb',
                    'override_gpus', 'override_tpu', 'suppress_override_warnings', 'mounts', 'extra_docker_compose',
                    'env', 'kwargs', 'extra_allowed_hosts'),
    'verifier': ('override_timeout_sec', 'max_timeout_sec', 'include_logs', 'exclude_logs', 'env', 'import_path',
                 'kwargs', 'disable'),
    'agent': ('name', 'import_path', 'model_name', 'n_concurrent', 'concurrency_group', 'skills',
              'override_timeout_sec', 'override_setup_timeout_sec', 'max_timeout_sec', 'resume_trajectory',
              'load_trajectory', 'extra_allowed_hosts', 'include_logs', 'exclude_logs', 'kwargs', 'env',
              'mcp_servers'),
    'dataset': ('path', 'name', 'version', 'ref', 'registry_url', 'registry_path', 'repo', 'overwrite',
                'download_dir', 'task_names', 'exclude_task_names', 'n_tasks'),
    'terminus_2': ('model_name', 'max_turns', 'parser_name', 'api_base', 'temperature', 'reasoning_effort',
                   'collect_rollout_details', 'session_id', 'enable_summarize', 'proactive_summarization_threshold',
                   'max_thinking_tokens', 'model_info', 'trajectory_config', 'tmux_pane_width', 'tmux_pane_height',
                   'store_all_messages', 'record_terminal_session', 'interleaved_thinking',
                   'suppress_max_turns_warning', 'use_responses_api', 'llm_backend', 'llm_kwargs', 'llm_call_kwargs',
                   'extra_env'),
}
# The keys a job of this harness sets (a subset of HARBOR_FIELDS): the settings of the runs.
JOB_KEYS = ('job_name', 'jobs_dir', 'n_attempts', 'timeout_multiplier', 'n_concurrent_trials', 'quiet', 'retry',
            'environment', 'verifier', 'agents', 'datasets')
AGENT_KWARGS = ('api_base', 'model_info', 'temperature', 'llm_kwargs')
LLM_KWARGS = {'max_tokens': 4096, 'timeout': 3600, 'max_retries': 0, 'num_retries': 0}


class JobError(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise JobError(message)


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_fields(job):
    """Every key of the job is a field of Harbor 0.21.0's models (HARBOR_FIELDS)."""
    def keys(value, kind, where):
        need(type(value) is dict, '%s must be an object' % where)
        unknown = sorted(set(value) - set(HARBOR_FIELDS[kind]))
        need(not unknown, '%s: not a Harbor 0.21.0 field: %s' % (where, ', '.join(unknown)))
    keys(job, 'job', 'job')
    for name in ('retry', 'environment', 'verifier'):
        if name in job:
            keys(job[name], name, name)
    for k, agent in enumerate(job.get('agents') or ()):
        keys(agent, 'agent', 'agents[%d]' % k)
        keys(agent.get('kwargs', {}), 'terminus_2', 'agents[%d].kwargs' % k)
    for k, dataset in enumerate(job.get('datasets') or ()):
        keys(dataset, 'dataset', 'datasets[%d]' % k)


def check_template(template, profile, tasks):
    """The template's settings are the runs' settings: exactly JOB_KEYS, one attempt, the task timeouts unchanged
    (timeout_multiplier 1.0, no agent timeout override), -n 16, no Harbor retries, the docker environment deleted
    after each trial, verification on, Terminus 2 with the served model at temperature 0, the hosted_vllm
    model_info at the profile's context and the 4096-token reply bound, and the dataset of tasks.json."""
    check_fields(template)
    need(tuple(sorted(template)) == tuple(sorted(JOB_KEYS)), 'template keys must be exactly %s' % ', '.join(JOB_KEYS))
    need(template['n_attempts'] == 1 and template['timeout_multiplier'] == 1.0 and template['quiet'] is True
         and template['n_concurrent_trials'] == 16, 'template: n_attempts 1, timeout_multiplier 1.0, quiet, -n 16')
    need(template['retry'] == {'max_retries': 0}, 'template: retry must be {"max_retries": 0}')
    need(template['environment'] == {'type': 'docker', 'delete': True, 'force_build': False},
         'template: environment must be docker, delete, no force_build')
    need(template['verifier'] == {'disable': False}, 'template: verifier must be {"disable": false}')
    need(type(template['agents']) is list and len(template['agents']) == 1, 'template: exactly one agent')
    agent = template['agents'][0]
    need(sorted(agent) == ['kwargs', 'model_name', 'name'] and agent['name'] == 'terminus-2',
         'template: the agent is terminus-2 with model_name and kwargs only')
    need(agent['model_name'] == 'hosted_vllm/selective-prefix' and HOSTED_VLLM_NAME.match(agent['model_name']),
         'template: model_name must be hosted_vllm/selective-prefix (the served model name)')
    kwargs = agent['kwargs']
    need(tuple(sorted(kwargs)) == tuple(sorted(AGENT_KWARGS)), 'template: agent kwargs must be %s'
         % ', '.join(AGENT_KWARGS))
    need(kwargs['api_base'] == 'http://127.0.0.1:@PORT@/v1', 'template: api_base must be the loopback @PORT@')
    need(kwargs['temperature'] == 0.0 and type(kwargs['temperature']) is float, 'template: temperature 0.0')
    need(kwargs['llm_kwargs'] == LLM_KWARGS, 'template: llm_kwargs must be %s' % json.dumps(LLM_KWARGS))
    info = kwargs['model_info']
    need(type(info) is dict and tuple(sorted(info)) == tuple(sorted(HOSTED_VLLM_MODEL_INFO))
         and all(number(info[k]) for k in info), 'template: model_info must be the four numeric hosted_vllm fields')
    need(info['max_input_tokens'] == CONTEXT[profile], 'template: max_input_tokens must be %d for %s'
         % (CONTEXT[profile], profile))
    need(info['max_output_tokens'] == LLM_KWARGS['max_tokens'] and info['input_cost_per_token'] == 0
         and info['output_cost_per_token'] == 0, 'template: max_output_tokens 4096, zero costs')
    need(type(template['datasets']) is list and len(template['datasets']) == 1, 'template: exactly one dataset')
    dataset = template['datasets'][0]
    need(dataset == dict(tasks['dataset'], task_names='@TASK_NAMES@'),
         'template: the dataset must be tasks.json\'s (name, ref) with @TASK_NAMES@')
    need(template['job_name'] == '@JOB_NAME@' and template['jobs_dir'] == '@JOBS_DIR@',
         'template: job_name @JOB_NAME@, jobs_dir @JOBS_DIR@')


def check_tasks(tasks):
    """tasks.json: the dataset, its tasks and the two sets (sorted, every set task listed, 900-s timeouts)."""
    need(tasks.get('schema') == 'terminal-bench-tasks-v1', 'tasks.json: unknown schema')
    dataset = tasks.get('dataset') or {}
    need(sorted(dataset) == ['name', 'ref'] and re.fullmatch(r'[a-z0-9-]+/[a-z0-9-]+', str(dataset['name']))
         and re.fullmatch(r'sha256:[0-9a-f]{64}', str(dataset['ref'])), 'tasks.json: dataset must be name and ref')
    names = sorted(tasks.get('tasks') or {})
    need(names and all(TASK_NAME.match(n) for n in names), 'tasks.json: bad task names')
    for name in names:
        t = tasks['tasks'][name]
        need(re.fullmatch(r'sha256:[0-9a-f]{64}', str(t.get('ref'))) and re.fullmatch(r'[0-9a-f]{64}',
             str(t.get('checksum'))) and number(t.get('agent_timeout_sec')), 'tasks.json: task %s' % name)
    sets = tasks.get('sets') or {}
    need(sorted(sets) == ['smoke32', 'trial16'], 'tasks.json: the sets are trial16 and smoke32')
    for key, size in (('trial16', 16), ('smoke32', 32)):
        value = sets[key]
        need(type(value) is list and len(value) == size and value == sorted(set(value)) and set(value) <= set(names),
             'tasks.json: %s must be %d sorted tasks of the list' % (key, size))
        slow = [n for n in value if tasks['tasks'][n]['agent_timeout_sec'] != AGENT_TIMEOUT_SEC]
        need(not slow, 'tasks.json: %s tasks without a %d-s agent timeout: %s' % (key, AGENT_TIMEOUT_SEC,
                                                                                ', '.join(slow)))
    need(set(sets['trial16']) <= set(sets['smoke32']), 'tasks.json: smoke32 must contain trial16')


def task_set(tasks, name):
    need(name in TASK_SETS, 'unknown task set %r (one of %s)' % (name, ', '.join(TASK_SETS)))
    return sorted(tasks['tasks']) if name == 'all' else list(tasks['sets'][name])


def make_job(profile, tasks_name, port, jobs_dir, job_name, stock=False, here=HERE, context=None):
    """The job (a dict) and its task names (plain). context: Terminus 2's context limit (model_info's
    max_input_tokens), one of CONTEXT_CHOICES; None keeps the template's, the profile's CONTEXT."""
    need(profile in PROFILES, 'unknown profile %r (one of %s)' % (profile, ', '.join(PROFILES)))
    need(context is None or (type(context) is int and context in CONTEXT_CHOICES),
         'context must be one of %s' % ', '.join(map(str, CONTEXT_CHOICES)))
    need(re.fullmatch(r'[0-9]{4,5}', str(port)) and 1024 <= int(port) <= 65535, 'port must be 1024-65535')
    need(type(job_name) is str and JOB_NAME.match(job_name), 'job name: letters, digits, ".", "_", "-" (1-100)')
    need(type(jobs_dir) is str and Path(jobs_dir).is_absolute() and '\n' not in jobs_dir,
         'jobs dir must be an absolute path')
    need(not (Path(jobs_dir) / job_name).exists(), 'job directory exists (Harbor would resume it): %s'
         % (Path(jobs_dir) / job_name))
    tasks = load(Path(here) / TASKS_FILE)
    check_tasks(tasks)
    names = task_set(tasks, tasks_name)
    template = load(Path(here) / PROFILES[profile])
    check_template(template, profile, tasks)
    org = tasks['dataset']['name'].split('/')[0]
    job = json.loads(json.dumps(template))
    job.update(job_name=job_name, jobs_dir=jobs_dir)
    kwargs = job['agents'][0]['kwargs']
    kwargs['api_base'] = 'http://127.0.0.1:%d/v1' % int(port)
    if context is not None:
        kwargs['model_info']['max_input_tokens'] = context
    if stock:
        kwargs['llm_call_kwargs'] = json.loads(json.dumps(STOCK_CALL_KWARGS))
    job['datasets'][0]['task_names'] = ['%s/%s' % (org, n) for n in names]
    check_fields(job)
    return job, names


def encode(job):
    """The job file's bytes (sorted keys, two-space indent, as the runs wrote it)."""
    return (json.dumps(job, indent=2, sort_keys=True) + '\n').encode('utf-8')


def planned(job):
    """The plain task names of a job file (its dataset's task_names without the organization)."""
    (dataset,) = job['datasets']
    org = dataset['name'].split('/')[0] + '/'
    names = dataset['task_names']
    need(type(names) is list and names and all(type(n) is str and n.startswith(org) for n in names),
         'job: task_names must be %s<task> names' % org)
    return [n[len(org):] for n in names]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--profile', required=True, choices=sorted(PROFILES))
    p.add_argument('--tasks', required=True, choices=TASK_SETS)
    p.add_argument('--port', required=True)
    p.add_argument('--jobs-dir', required=True)
    p.add_argument('--job-name', required=True)
    p.add_argument('--stock', action='store_true')
    p.add_argument('--context', type=int, choices=CONTEXT_CHOICES,
                   help="Terminus 2's context limit: the profile's 262144 (default) or 65536, the context of the "
                        "earlier dry runs; the server's --max-model-len must be the same")
    p.add_argument('--out', required=True, help="the job file to write (must not exist), or '-' for stdout")
    a = p.parse_args(argv)
    try:
        job, names = make_job(a.profile, a.tasks, a.port, a.jobs_dir, a.job_name, a.stock, context=a.context)
        raw = encode(job)
        if a.out == '-':
            sys.stdout.write(raw.decode('utf-8'))
            return 0
        need(Path(a.out).parent.is_dir() and not os.path.lexists(a.out), 'out must be a new file in an existing '
             'directory: %s' % a.out)
        with open(a.out, 'xb') as f:
            f.write(raw)
    except JobError as error:
        print(json.dumps(dict(status='JOB_REFUSED', error=str(error))), file=sys.stderr)
        return 2
    print(json.dumps(dict(status='JOB_WRITTEN', job=a.out, sha256=hashlib.sha256(raw).hexdigest(),
                          profile=a.profile, stock=a.stock, task_set=a.tasks, tasks=len(names),
                          context=job['agents'][0]['kwargs']['model_info']['max_input_tokens'],
                          job_dir=str(Path(a.jobs_dir) / a.job_name)), sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())

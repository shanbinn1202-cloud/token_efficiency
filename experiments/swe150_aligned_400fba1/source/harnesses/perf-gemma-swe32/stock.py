"""The same-node stock vLLM 0.24.0 control of perf-gemma-swe32 (stdlib only; the GPU job and the verifier).

perf-cudagraph's stock server (plain `python -I -B -m vllm.entrypoints.openai.api_server` from the reused clean env, no
runtime source, no KV connector; job 3321538's environment) serving the chat API to the same agents: the argv is
controller.stock_argv (config.stock, PROVISIONAL Vanilla settings), parsed once by vLLM's own parser on the GPU node
before the server starts (`argv_gate`: vLLM infers the device while building its parser, so the login node cannot).
READY = GET /health 200 and /v1/models listing the served name. Stop: SIGTERM to the APIServer parent only, a process
group signal only as a failed escalation. The server writes no runtime evidence: its record is the server log
(vLLM's engine config, async scheduling, CUDA-graph capture, the KV size), the /metrics samples and the agents'
trajectories on tbench (stock has no session close route: the pool skips the close).
"""
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER_MODULE = 'vllm.entrypoints.openai.api_server'
ARGV_PROBE = r'''
import json, sys
spec = json.load(open(sys.argv[1]))
import vllm
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.engine.arg_utils import AsyncEngineArgs
args = make_arg_parser(FlexibleArgumentParser(description='gswe32 stock argv gate')).parse_args(spec['argv'])
validate_parsed_serve_args(args)
engine = AsyncEngineArgs.from_cli_args(args)
values = vars(args)
import enum


def plain(value):
    # an enum destination (--attention-backend: AttentionBackendEnum) is compared by its member name
    return value.name if isinstance(value, enum.Enum) else value


out = dict(version=vllm.__version__, file=vllm.__file__, serve_args={k: plain(values.get(k, '<absent>')) for k in spec['keys']},
           engine={k: repr(getattr(engine, k, '<absent>')) for k in ('enforce_eager', 'async_scheduling',
                                                                     'compilation_config', 'kv_cache_memory_bytes')})
print(json.dumps(out, default=repr, sort_keys=True))
'''
LOG_PATTERNS = dict(
    kv_cache_size_tokens=re.compile(r'GPU KV cache size: ([\d,]+) tokens'),
    async_scheduling=re.compile(r'Asynchronous scheduling is (enabled|disabled)'),
    graph_capture=re.compile(r'Graph capturing finished in ([\d.]+) secs?, took ([\d.]+) GiB'),
    engine_version=re.compile(r'Initializing a V1 LLM engine \(v([0-9.]+)\)'),
    model_load=re.compile(r'Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds'))
CUDAGRAPH_MODE = re.compile(r"'cudagraph_mode': <CUDAGraphMode\.([A-Z_]+): [^>]*>")
ENFORCE_EAGER = re.compile(r'\benforce_eager=(True|False)')


def server_command(python, argv):
    return [str(python), '-I', '-B', '-m', SERVER_MODULE] + [str(a) for a in argv]


def server_env(environ, stage, extra=None):
    """job 3321538's stock environment: no REMOTE_RA_* variable, the runtime switch off, spawn workers, the V1 runner,
    stage-local vLLM config and caches, no usage statistics, offline; extra: config.stock.env (Gemma stage 0:
    VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0)."""
    env = {k: v for k, v in environ.items() if not k.startswith('REMOTE_RA_')}
    env.pop('VLLM_POLICY_RUNTIME_MANIFEST', None)
    root = Path(stage) / 'cache' / 'stock'
    env.update(VLLM_POLICY_RUNTIME_V1='0', VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_USE_V2_MODEL_RUNNER='0',
               VLLM_NO_USAGE_STATS='1', VLLM_CACHE_ROOT=str(root / 'vllm'),
               VLLM_CONFIG_ROOT=str(Path(stage) / 'home' / '.config' / 'vllm'), TRITON_CACHE_DIR=str(root / 'triton'),
               TORCHINDUCTOR_CACHE_DIR=str(root / 'inductor'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', VLLM_USE_FLASHINFER_SAMPLER='0')  # the native sampler, as our servers
    env.update({str(k): str(v) for k, v in (extra or {}).items()})
    return env


def argv_gate(stage, python, argv, expected, *, runner=subprocess.run, env=None):
    """vLLM's own parser reads the stock argv back as pinned (perf-cudagraph job.argv_gate): every expected destination
    and the engine's KV bytes; mode default leaves enforce_eager False. Returns the record (status PASS or FAIL)."""
    spec = Path(stage) / 'tmp' / 'stock-argv-spec.json'
    spec.write_text(json.dumps(dict(argv=[str(a) for a in argv], keys=sorted(expected))), encoding='utf-8')
    result = runner([str(python), '-I', '-B', '-c', ARGV_PROBE, str(spec)], capture_output=True, text=True,
                    timeout=900, env=env)
    lines = [line for line in (result.stdout or '').splitlines() if line.startswith('{')]
    value = json.loads(lines[-1]) if lines else {}
    got = value.get('serve_args') or {}
    differ = {k: (got.get(k), v) for k, v in expected.items() if got.get(k) != v}
    ok = result.returncode == 0 and value.get('version') == '0.24.0' and not differ
    return dict(status='STOCK_ARGV_GATE_PASS' if ok else 'STOCK_ARGV_GATE_FAIL', returncode=result.returncode,
                differ=differ, probe=value, stderr=(result.stderr or '')[-3000:])


def http_get(url, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def wait_ready(base_url, model, process, deadline, *, get=http_get, clock=time.monotonic, sleep=time.sleep):
    """READY: GET /health is 200 and GET /v1/models lists the served model."""
    while True:
        if process.poll() is not None:
            raise RuntimeError('stock server exited before READY (exit %s)' % process.poll())
        try:
            status, _ = get(base_url + '/health', 5)
            if status == 200:
                status, raw = get(base_url + '/v1/models', 5)
                if status == 200 and model in [m.get('id') for m in json.loads(raw).get('data', [])]:
                    return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        if clock() >= deadline:
            raise RuntimeError('stock server not READY within the time budget')
        sleep(2)


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_server(process, receipt, wait=60):
    """SIGTERM to the APIServer parent only; process-group signals only as a failed escalation. The stock server's own
    exit code is recorded, never required to be 0."""
    receipt.update(pid=process.pid, target='APIServer parent only', signal=None, signal_sent=False,
                   pre_signal_exitcode=process.poll(), escalated=False, group_leftovers=False)
    if receipt['pre_signal_exitcode'] is None:
        os.kill(process.pid, signal.SIGTERM)
        receipt.update(signal='SIGTERM', signal_sent=True)
        try:
            receipt['exitcode'] = process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            receipt['escalated'] = True
            for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
                try:
                    os.killpg(process.pid, sig)
                    receipt['exitcode'] = process.wait(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
                except ProcessLookupError:
                    break
    else:
        receipt['exitcode'] = receipt['pre_signal_exitcode']
    deadline = time.monotonic() + 20
    while group_alive(process.pid) and time.monotonic() < deadline:
        time.sleep(.5)
    if group_alive(process.pid):
        receipt.update(group_leftovers=True, escalated=True)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return receipt


def startup_info(log_text):
    """What the stock server log says about its engine (first match each; absent -> None)."""
    info = {}
    for key, pattern in LOG_PATTERNS.items():
        found = pattern.search(log_text)
        info[key] = found.groups() if found else None
    mode = CUDAGRAPH_MODE.search(log_text)
    eager = ENFORCE_EAGER.search(log_text)
    info.update(cudagraph_mode=mode.group(1) if mode else None,
                enforce_eager=(eager.group(1) == 'True') if eager else None,
                tracebacks=log_text.count('Traceback (most recent call last):'))
    return info


def check_server(out, server_log, stock_config, outcomes, language_model_only=None):
    """Checks of the stock server (`stock_config` = config.stock): the engine mode (default: CUDA graphs on, i.e.
    enforce_eager False), async scheduling as the one switch says (every 'Asynchronous scheduling is ...' line), the
    kernel set of our servers, the /metrics samples readable and vLLM's preemption counter 0 in every sample, and every
    instance that ran either done or truncated; with `language_model_only` (vLLM --language-model-only, as our servers)
    no multi-modal startup line in the log. Returns dict(status, checks, failed, findings, ...) like swe_checks."""
    expected_mode = stock_config['mode']
    import measure
    import smoke_checks as sc
    report = sc.Report()
    text = Path(server_log).read_text(encoding='utf-8', errors='replace') if Path(server_log).exists() else ''
    info = startup_info(text)

    def mode():
        sc.need(info['enforce_eager'] is (expected_mode == 'eager'), 'stock server enforce_eager %r for mode %s'
                % (info['enforce_eager'], expected_mode))
        return info
    report.run('stock-mode', mode)

    def async_scheduling():
        found = sorted(set(LOG_PATTERNS['async_scheduling'].findall(text)))
        wanted = ['enabled'] if stock_config['async_scheduling'] else ['disabled']
        sc.need(found == wanted, 'stock server async scheduling %s, config.stock.async_scheduling %r'
                % (found, stock_config['async_scheduling']))
        return dict(async_scheduling=found[0])
    report.run('stock-async-scheduling', async_scheduling)
    import swe_checks
    report.run('kernel-alignment', lambda: swe_checks.check_kernels(server_log))  # the same set as our servers
    if language_model_only is not None:
        report.run('language-model-only', lambda: swe_checks.check_text_only_log(server_log, language_model_only))
    report.run('no-preemption', lambda: sc.check_no_preemption(out))

    def instances():
        bad = {i: v.get('status') for i, v in (outcomes or {}).items() if v.get('status') not in ('done', 'truncated')}
        sc.need(not bad, 'instances without a trajectory: %s' % bad)
        return dict(instances=len(outcomes or {}))
    report.run('instances', instances)
    timeline, error = measure.load_timeline(out)
    return dict(policy='stock', status='PASS' if not report.failed else 'FAIL', failed=report.failed,
                findings=report.findings, checks=report.checks, startup=info,
                batch=measure.batch_profile(timeline), batch_error=error, rows=None)

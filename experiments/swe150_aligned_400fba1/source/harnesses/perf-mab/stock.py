"""The stock vLLM 0.24.0 servers of perf-mab (stdlib only; the GPU job and the verifier).

perf-decode2's stock control (plain `python -I -B -m vllm.entrypoints.openai.api_server` from the reused clean env, no
runtime source, no KV connector), one server per config.servers entry of kind `stock`: the argv is
controller.stock_argv (the G1c replay server eager and synchronous, the Vanilla server with vLLM's defaults), the
environment perf-swe16's. Every planned stock argv is parsed once by vLLM's own parser on the GPU node before the first
server starts (`argv_gate`); a server whose argv the parser rejects or misreads does not start. READY = GET /health
200 and /v1/models listing the served name. Stop: SIGTERM to the APIServer parent only, a process-group signal only as
a failed escalation. The stock servers share one compile cache ($stage/cache/stock).
"""
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER_MODULE = 'vllm.entrypoints.openai.api_server'
ARGV_PROBE = r'''
import json, sys
specs = json.load(open(sys.argv[1]))
import vllm
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.engine.arg_utils import AsyncEngineArgs
out = dict(version=vllm.__version__, file=vllm.__file__, servers={})
for label, spec in sorted(specs.items()):
    try:
        args = make_arg_parser(FlexibleArgumentParser(description='pmab stock argv gate')).parse_args(spec['argv'])
        validate_parsed_serve_args(args)
        engine = AsyncEngineArgs.from_cli_args(args)
        values = vars(args)
        out['servers'][label] = dict(serve_args={k: values.get(k, '<absent>') for k in spec['keys']},
            engine={k: repr(getattr(engine, k, '<absent>')) for k in ('enforce_eager', 'async_scheduling',
                                                                     'compilation_config', 'kv_cache_memory_bytes',
                                                                     'profiler_config')}, error=None)
    except BaseException as exc:
        out['servers'][label] = dict(error='%s: %s' % (type(exc).__name__, exc))
print(json.dumps(out, default=repr, sort_keys=True))
'''


def server_command(python, argv):
    return [str(python), '-I', '-B', '-m', SERVER_MODULE] + [str(a) for a in argv]


def server_env(environ, stage, extra=None):
    """job 3321538's stock environment: no REMOTE_RA_* variable, the runtime switch off, spawn workers, the V1 runner,
    stage-local vLLM config and caches (shared by the stock servers), no usage statistics, offline, the native
    sampler; then the server's own additions (none in perf-mab)."""
    env = {k: v for k, v in environ.items() if not k.startswith('REMOTE_RA_')}
    env.pop('VLLM_POLICY_RUNTIME_MANIFEST', None)
    env.pop('VLLM_KV_CACHE_LAYOUT', None)
    env.pop('VLLM_CUSTOM_SCOPES_FOR_PROFILING', None)
    root = Path(stage) / 'cache' / 'stock'
    env.update(VLLM_POLICY_RUNTIME_V1='0', VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_USE_V2_MODEL_RUNNER='0',
               VLLM_NO_USAGE_STATS='1', VLLM_CACHE_ROOT=str(root / 'vllm'),
               VLLM_CONFIG_ROOT=str(Path(stage) / 'home' / '.config' / 'vllm'), TRITON_CACHE_DIR=str(root / 'triton'),
               TORCHINDUCTOR_CACHE_DIR=str(root / 'inductor'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', VLLM_USE_FLASHINFER_SAMPLER='0')
    env.update(extra or {})
    return env


def argv_gate(stage, python, specs, *, runner=subprocess.run, env=None):
    """vLLM's own parser reads every planned stock argv back as pinned: `specs` {label: dict(argv, expected)}. Returns
    the record: status PASS only if every server parsed and matched; per server `ok`, `differ` and the parser's
    error."""
    path = Path(stage) / 'tmp' / 'stock-argv-specs.json'
    path.write_text(json.dumps({label: dict(argv=[str(a) for a in s['argv']], keys=sorted(s['expected']))
                                for label, s in specs.items()}), encoding='utf-8')
    result = runner([str(python), '-I', '-B', '-c', ARGV_PROBE, str(path)], capture_output=True, text=True,
                    timeout=900, env=env)
    lines = [line for line in (result.stdout or '').splitlines() if line.startswith('{')]
    value = json.loads(lines[-1]) if lines else {}
    servers = {}
    for label, spec in specs.items():
        got = (value.get('servers') or {}).get(label) or {}
        differ = {k: ((got.get('serve_args') or {}).get(k), v) for k, v in spec['expected'].items()
                  if (got.get('serve_args') or {}).get(k) != v}
        ok = result.returncode == 0 and value.get('version') == '0.24.0' and not got.get('error') and not differ \
            and bool(got)
        servers[label] = dict(ok=ok, differ=differ, error=got.get('error'), engine=got.get('engine'))
    status = 'STOCK_ARGV_GATE_PASS' if servers and all(s['ok'] for s in servers.values()) else 'STOCK_ARGV_GATE_FAIL'
    return dict(status=status, returncode=result.returncode, version=value.get('version'), file=value.get('file'),
                servers=servers, stderr=(result.stderr or '')[-3000:])


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _opener():
    return _OPENER


def http_get(url, timeout):
    try:
        with _opener().open(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def http_post(url, timeout=600):
    request = urllib.request.Request(url, data=b'', method='POST')
    try:
        with _opener().open(request, timeout=timeout) as response:
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

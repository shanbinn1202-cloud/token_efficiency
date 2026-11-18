"""The same-node stock vLLM 0.24.0 control of perf-swe32 (stdlib only; the GPU job and the verifier).

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
args = make_arg_parser(FlexibleArgumentParser(description='swe32 stock argv gate')).parse_args(spec['argv'])
validate_parsed_serve_args(args)
engine = AsyncEngineArgs.from_cli_args(args)
values = vars(args)
out = dict(version=vllm.__version__, file=vllm.__file__, serve_args={k: values.get(k, '<absent>') for k in spec['keys']},
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
# vLLM 0.24.0's API server echo of its arguments (job 3429193: "[api_utils.py:273] non-default args: {...,
# 'max_model_len': 131072, ..., 'kv_cache_memory_bytes': 34359738368, ...}") and the engine's concurrency line
NON_DEFAULT_ARGS = re.compile(r'non-default args: (\{.*\})')
CONCURRENCY = re.compile(r'Maximum concurrency for ([\d,]+) tokens per request')
PREEMPTED = 'STOCK_PREEMPTED'
# 2026-11-12 (the stock control with thinking ON, controller.stock_thinking): the job's probe of the live server after
# READY, written into the server's output directory: the same two messages with chat_template_kwargs enable_thinking
# false and true (the agents' own kwargs), max_tokens 1, greedy. The template's thinking-on generation header
# (<|im_start|>assistant\n<think>\n, 5 tokens) is 2 tokens shorter than the thinking-off one (... <think>\n\n</think>\n\n,
# 7 tokens), so usage.prompt_tokens(on) = prompt_tokens(off) - 2 exactly when the server applied the kwargs.
THINKING_PROBE = 'stock-thinking-probe.json'
THINKING_PROBE_MESSAGES = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                           {'role': 'user', 'content': 'Reply with one word.'}]
HEADER_TOKENS_ON_MINUS_OFF = -2
# 2026-11-16 (the stock keep-reasoning plans, controller.stock_contract history reasoning_kept): the job's probe of the
# live server after READY and the thinking probe, before any agent. One conversation with an earlier reply (content
# 'Done.', reasoning KEEP_PROBE_REASONING) sent three ways, max_tokens 1, greedy, the agents' kwargs:
#   kept           the reply carries `reasoning` (and reasoning_content), as the keep model class sends it;
#   content_only   the reply carries only reasoning_content (Mini-SWE-Agent through litellm alone): vLLM drops it;
#   not_preserved  as kept, with the request's chat_template_kwargs preserve_thinking false over the server default.
# With the server's --default-chat-template-kwargs {"preserve_thinking": true} the pinned template renders the reply as
# <think>\n R \n</think>\n\n Done. (kept), <think>\n\n</think>\n\n Done. (content_only: the empty block) and Done.
# (not_preserved). The pinned tokenizer and template (postrun/stock_history.py, the token-exact test in
# test_real_tokenizer.py): kept - content_only = 14 and content_only - not_preserved = 4 prompt tokens. PASS iff all
# three answer 200 with exactly these differences: the server applies the default kwargs, reads `reasoning`, and the
# template keeps it.
KEEP_PROBE = 'stock-keep-probe.json'
KEEP_PROBE_REASONING = 'The user wants one word, so I answer with one word.'
KEEP_PROBE_DIFFERENCES = dict(kept_minus_content_only=14, content_only_minus_not_preserved=4)


def keep_probe_messages(variant):
    reply = dict(role='assistant', content='Done.', reasoning_content=KEEP_PROBE_REASONING)
    if variant in ('kept', 'not_preserved'):
        reply['reasoning'] = KEEP_PROBE_REASONING
    return THINKING_PROBE_MESSAGES + [reply, {'role': 'user', 'content': 'Reply with one more word.'}]


def server_command(python, argv):
    return [str(python), '-I', '-B', '-m', SERVER_MODULE] + [str(a) for a in argv]


def server_env(environ, stage):
    """job 3321538's stock environment: no REMOTE_RA_* variable, the runtime switch off, spawn workers, the V1 runner,
    stage-local vLLM config and caches, no usage statistics, offline."""
    env = {k: v for k, v in environ.items() if not k.startswith('REMOTE_RA_')}
    env.pop('VLLM_POLICY_RUNTIME_MANIFEST', None)
    root = Path(stage) / 'cache' / 'stock'
    env.update(VLLM_POLICY_RUNTIME_V1='0', VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_USE_V2_MODEL_RUNNER='0',
               VLLM_NO_USAGE_STATS='1', VLLM_CACHE_ROOT=str(root / 'vllm'),
               VLLM_CONFIG_ROOT=str(Path(stage) / 'home' / '.config' / 'vllm'), TRITON_CACHE_DIR=str(root / 'triton'),
               TORCHINDUCTOR_CACHE_DIR=str(root / 'inductor'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', VLLM_USE_FLASHINFER_SAMPLER='0')  # the native sampler, as our servers
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


def http_post(url, body, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'},
                                     method='POST')
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def thinking_probe(base_url, model, contract, *, post=http_post, timeout=120):
    """The live stock server's generation header with the agents' thinking-on kwargs (THINKING_PROBE): two requests,
    enable_thinking false and the contract's chat_template_kwargs, max_tokens 1, temperature 0. PASS iff both answer
    200 with usage.prompt_tokens and on = off - 2. The one-token replies' fields are recorded (content, reasoning,
    reasoning_content), not judged: a reply cut inside its reasoning is the parser's own case."""
    kwargs = contract['stock_model_kwargs']['extra_body']['chat_template_kwargs']
    record = dict(schema='swe32-stock-thinking-probe-v1', messages=THINKING_PROBE_MESSAGES, kwargs_on=kwargs,
                  expected_difference=HEADER_TOKENS_ON_MINUS_OFF)
    for key, template in (('off', dict(kwargs, enable_thinking=False)), ('on', dict(kwargs))):
        body = dict(model=model, messages=THINKING_PROBE_MESSAGES, max_tokens=1, temperature=0.0,
                    chat_template_kwargs=template)
        try:
            status, raw = post(base_url + '/v1/chat/completions', body, timeout)
            value = json.loads(raw) if status == 200 else {}
        except (OSError, ValueError, urllib.error.URLError) as exc:
            status, value = None, dict(error='%s: %s' % (type(exc).__name__, exc))
        message = ((value.get('choices') or [{}])[0].get('message') or {})
        record[key] = dict(http=status, prompt_tokens=(value.get('usage') or {}).get('prompt_tokens'),
                           content=message.get('content'), reasoning=message.get('reasoning'),
                           reasoning_content=message.get('reasoning_content'), error=value.get('error'))
    on, off = record['on'].get('prompt_tokens'), record['off'].get('prompt_tokens')
    record['difference'] = on - off if type(on) is int and type(off) is int else None
    ok = record['on']['http'] == record['off']['http'] == 200 and record['difference'] == HEADER_TOKENS_ON_MINUS_OFF
    record.update(status='PASS' if ok else 'FAIL', reason=None if ok else
                  'prompt tokens on %r / off %r (http %r / %r): the thinking-on header is not 2 tokens shorter'
                  % (on, off, record['on']['http'], record['off']['http']))
    return record


def keep_probe(base_url, model, contract, *, post=http_post, timeout=120):
    """The live stock server's history with the earlier reasoning kept (KEEP_PROBE, 2026-11-16): the three requests of
    keep_probe_messages, max_tokens 1, temperature 0, the agents' chat_template_kwargs (not_preserved: with
    preserve_thinking false over the server default). PASS iff all three answer 200 and the prompt-token differences
    equal KEEP_PROBE_DIFFERENCES."""
    kwargs = contract['stock_model_kwargs']['extra_body']['chat_template_kwargs']
    record = dict(schema='swe32-stock-keep-probe-v1', reasoning=KEEP_PROBE_REASONING, kwargs=kwargs,
                  server_default=(contract.get('kept') or {}).get('default_chat_template_kwargs'),
                  expected=dict(KEEP_PROBE_DIFFERENCES))
    for variant in ('kept', 'content_only', 'not_preserved'):
        template = dict(kwargs, preserve_thinking=False) if variant == 'not_preserved' else dict(kwargs)
        body = dict(model=model, messages=keep_probe_messages(variant), max_tokens=1, temperature=0.0,
                    chat_template_kwargs=template)
        try:
            status, raw = post(base_url + '/v1/chat/completions', body, timeout)
            value = json.loads(raw) if status == 200 else {}
        except (OSError, ValueError, urllib.error.URLError) as exc:
            status, value = None, dict(error='%s: %s' % (type(exc).__name__, exc))
        record[variant] = dict(http=status, prompt_tokens=(value.get('usage') or {}).get('prompt_tokens'),
                               error=value.get('error'))
    tokens = [record[v]['prompt_tokens'] for v in ('kept', 'content_only', 'not_preserved')]
    got = dict(kept_minus_content_only=tokens[0] - tokens[1], content_only_minus_not_preserved=tokens[1] - tokens[2]) \
        if all(type(t) is int for t in tokens) else None
    record['differences'] = got
    ok = all(record[v]['http'] == 200 for v in ('kept', 'content_only', 'not_preserved')) \
        and got == KEEP_PROBE_DIFFERENCES
    record.update(status='PASS' if ok else 'FAIL', reason=None if ok else
                  'prompt tokens kept / content_only / not_preserved %r (http %r): differences %r, expected %r'
                  % (tokens, [record[v]['http'] for v in ('kept', 'content_only', 'not_preserved')], got,
                     KEEP_PROBE_DIFFERENCES))
    return record


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


def planned_settings(text, stock_config):
    """stock25: vLLM's own echo of the plan's settings in the server log: the API server's non-default args carry its
    --max-model-len and --kv-cache-memory-bytes, and the engine's concurrency line names the same context."""
    import smoke_checks as sc
    found = NON_DEFAULT_ARGS.search(text)
    sc.need(found, 'no "non-default args" line in the stock server log')
    got = {}
    for key in ('max_model_len', 'kv_cache_memory_bytes'):
        value = re.search(r"'%s': (\d+)" % key, found.group(1))
        got[key] = int(value.group(1)) if value else None
    want = dict(max_model_len=stock_config['max_model_len'], kv_cache_memory_bytes=stock_config['kv_cache_gib'] * 1024 ** 3)
    sc.need(got == want, 'stock server started with %r, the plan asks for %r' % (got, want))
    concurrency = [int(v.replace(',', '')) for v in CONCURRENCY.findall(text)]
    sc.need(concurrency and set(concurrency) == {want['max_model_len']},
            'the engine\'s concurrency line names %r tokens per request, not %d' % (concurrency, want['max_model_len']))
    kv = LOG_PATTERNS['kv_cache_size_tokens'].search(text)
    return dict(got, kv_cache_size_tokens=int(kv.group(1).replace(',', '')) if kv else None,
                preemption=stock_config.get('preemption', 'fail'))


def literal_dict(text):
    """A dict echoed in vLLM's log (Python repr; JSON accepted too), or None."""
    import ast
    for read in (ast.literal_eval, json.loads):
        try:
            value = read(text)
            return value if isinstance(value, dict) else None
        except (ValueError, SyntaxError):
            continue
    return None


def check_thinking(out, text, contract):
    """stock-thinking (2026-11-12): vLLM's echo of its non-default args names the contract's reasoning parser, and the
    job's probe after READY (THINKING_PROBE in the output directory) passed: the live server's thinking-on prompt is
    the thinking-off prompt minus 2 tokens (the thinking-on generation header)."""
    import smoke_checks as sc
    found = NON_DEFAULT_ARGS.search(text)
    sc.need(found, 'no "non-default args" line in the stock server log')
    parser = re.search(r"'reasoning_parser': '([A-Za-z0-9_.-]+)'", found.group(1))
    sc.need(parser and parser.group(1) == contract['reasoning_parser'],
            'the stock server started with reasoning parser %r, the contract asks for %r'
            % (parser and parser.group(1), contract['reasoning_parser']))
    path = Path(out) / THINKING_PROBE
    sc.need(path.is_file(), 'no %s: the job did not probe the thinking-on header' % THINKING_PROBE)
    probe = json.loads(path.read_text(encoding='utf-8'))
    again = probe.get('on', {}).get('prompt_tokens'), probe.get('off', {}).get('prompt_tokens')
    sc.need(probe.get('status') == 'PASS' and type(again[0]) is int and type(again[1]) is int
            and again[0] - again[1] == HEADER_TOKENS_ON_MINUS_OFF and probe.get('kwargs_on') ==
            contract['stock_model_kwargs']['extra_body']['chat_template_kwargs'],
            'stock thinking probe %r: %r' % (probe.get('status'), probe.get('reason')))
    value = dict(reasoning_parser=parser.group(1), history=contract['history'], prompt_tokens_on=again[0],
                 prompt_tokens_off=again[1], reply_on=dict((k, probe['on'].get(k)) for k in
                                                           ('content', 'reasoning', 'reasoning_content')))
    kept = contract.get('kept') if contract['history'] == 'reasoning_kept' else None
    if kept:  # 2026-11-16: the keep-reasoning history: vLLM's echo of the default kwargs and the job's keep probe
        defaults = re.search(r"'default_chat_template_kwargs': (\{[^{}]*\})", found.group(1))
        want = dict(kept['default_chat_template_kwargs'])
        sc.need(defaults and literal_dict(defaults.group(1)) == want, 'the stock server started with default chat '
                'template kwargs %r, the contract asks for %r' % (defaults and defaults.group(1), want))
        path = Path(out) / KEEP_PROBE
        sc.need(path.is_file(), 'no %s: the job did not probe the kept history' % KEEP_PROBE)
        keep = json.loads(path.read_text(encoding='utf-8'))
        tokens = [(keep.get(v) or {}).get('prompt_tokens') for v in ('kept', 'content_only', 'not_preserved')]
        sc.need(keep.get('status') == 'PASS' and all(type(t) is int for t in tokens)
                and dict(kept_minus_content_only=tokens[0] - tokens[1],
                         content_only_minus_not_preserved=tokens[1] - tokens[2]) == KEEP_PROBE_DIFFERENCES
                and keep.get('reasoning') == KEEP_PROBE_REASONING,
                'stock keep probe %r: %r' % (keep.get('status'), keep.get('reason')))
        value.update(default_chat_template_kwargs=defaults.group(1), keep_probe_prompt_tokens=tokens)
    return value


def check_server(out, server_log, stock_config, outcomes, language_model_only=None, settings=None, thinking=None):
    """Checks of the stock server (`stock_config` = config.stock, or a plan's own stock settings over it): the engine
    mode (default: CUDA graphs on, i.e. enforce_eager False), async scheduling as the one switch says (every
    'Asynchronous scheduling is ...' line), the kernel set of our servers, the /metrics samples readable and vLLM's
    preemption counter 0 in every sample, and every instance that ran either done or truncated; with
    `language_model_only` (vLLM --language-model-only, as our servers) no multi-modal startup line in the log. With a
    plan's own stock `settings` (stock25) also `stock-settings` (planned_settings), and with its preemption 'finding' a
    preempted request is the finding STOCK_PREEMPTED instead of a failure (the counter must still be readable). Returns
    dict(status, checks, failed, findings, ...) like swe_checks."""
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
    if settings:
        report.run('stock-settings', lambda: planned_settings(text, stock_config))
    if thinking:  # 2026-11-12: the stock thinking-on contract (controller.stock_thinking)
        report.run('stock-thinking', lambda: check_thinking(out, text, thinking))

    def preemption():
        try:
            return sc.check_no_preemption(out)
        except sc.CheckFailed as exc:  # only the preempted count itself; an unreadable counter stays a failure
            if stock_config.get('preemption', 'fail') == 'finding' and str(exc).startswith('vLLM preempted'):
                raise sc.Finding(PREEMPTED, str(exc))
            raise
    report.run('no-preemption', preemption)

    def instances():
        bad = {i: v.get('status') for i, v in (outcomes or {}).items() if v.get('status') not in ('done', 'truncated')}
        sc.need(not bad, 'instances without a trajectory: %s' % bad)
        return dict(instances=len(outcomes or {}))
    report.run('instances', instances)
    timeline, error = measure.load_timeline(out)
    return dict(policy='stock', status='PASS' if not report.failed else 'FAIL', failed=report.failed,
                findings=report.findings, checks=report.checks, startup=info,
                batch=measure.batch_profile(timeline), batch_error=error, rows=None)

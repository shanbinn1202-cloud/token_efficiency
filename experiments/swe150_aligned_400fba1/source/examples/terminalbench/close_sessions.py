"""Check a chat-text server before a Harbor run, and close the sessions the run left open after it.

Terminus 2 sends no X-Session-ID and never closes its sessions; the server keeps a finished trial's session (and
its summarization sub-calls' sessions) open until it is idle long enough to be evicted. After the run, `close`
reads every client session from GET /v1/chat-text/info and closes each one that is still open with
POST /v1/chat-text/close: {"session_id": ...} for a session under an X-Session-ID header (all of that header's
sessions at once), else {"key_sha256": ...}. A 409 (a request still queued or in flight) is retried every 2 s up
to --limit seconds after the first close (one bound for all). It then reads the info again and exits 0
("SESSIONS_CLOSED") only when no session is open, 1 otherwise ("SESSIONS_LEFT_OPEN").

The server accepts a close from a loopback peer only: run this on the server's node, or through the reverse
tunnel's relay on the agent host (the tunnel's far end connects to the server from 127.0.0.1).

`check`, before the run, refuses (exit 1, "SERVER_REFUSED") a server that does not serve the job: GET /v1/models
must list the job's model; GET /v1/chat-text/info must report the chat-text protocol, the context limit
(max_model_len) of the job's model_info.max_input_tokens, the reply bound (max_tokens) of its max_output_tokens,
at least the job's concurrent trials as sessions (max_running_sessions), the profile's model_profile, and no
client session yet (one fresh server per run). For a --stock job (a stock vLLM control) the model only, and no
chat-text route.

Usage:
  python3 -I -B examples/terminalbench/close_sessions.py check --base-url http://127.0.0.1:20000 \\
      --job <job file> --profile qwen3.6-27b
  python3 -I -B examples/terminalbench/close_sessions.py close --base-url http://127.0.0.1:20000
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_job as M  # noqa: E402

INFO = '/v1/chat-text/info'
CLOSE = '/v1/chat-text/close'
MODELS = '/v1/models'
PROTOCOL = 'chat_text_v1'
TIMEOUT_S = 30
# GET /v1/chat-text/info's options.model_profile per profile (absent on a Qwen server).
MODEL_PROFILE = {'qwen3.6-27b': None, 'gemma4-31b-it': 'gemma4-31b-it'}
# Loopback only: no proxy from the environment (http_proxy, the system settings).
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(base_url, path, body=None, timeout=TIMEOUT_S):
    """(HTTP status, parsed JSON body or None); (None, error text) when the server cannot be reached."""
    data = None if body is None else json.dumps(body).encode('utf-8')
    req = urllib.request.Request(base_url.rstrip('/') + path, data=data, method='GET' if body is None else 'POST',
                                 headers={'Content-Type': 'application/json'} if body is not None else {})
    try:
        with OPENER.open(req, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    except (urllib.error.URLError, OSError) as error:
        return None, '%s: %s' % (type(error).__name__, error)
    try:
        return status, json.loads(raw.decode('utf-8'))
    except ValueError:
        return status, None


def base_url(value):
    if not re.fullmatch(r'http://(127\.0\.0\.1|localhost|\[::1\]):[0-9]{2,5}', value.rstrip('/')):
        raise argparse.ArgumentTypeError('the base URL must be http://127.0.0.1:<port> (loopback)')
    return value.rstrip('/')


def open_sessions(info):
    return [s for s in (info or {}).get('sessions') or () if s.get('open')]


def targets(info):
    """The closes for the open sessions: one per X-Session-ID header, else one per key_sha256 (in info order)."""
    out = []
    for s in open_sessions(info):
        target = {'session_id': s['session_id']} if s.get('session_id') else {'key_sha256': s.get('key_sha256')}
        if target not in out:
            out.append(target)
    return out


def close_all(url, limit=60.0, call=request, clock=time.monotonic, sleep=time.sleep):
    status, info = call(url, INFO)
    if status != 200 or type(info) is not dict:
        return dict(status='SESSIONS_LEFT_OPEN', error='GET %s: HTTP %s %s' % (INFO, status, info))
    closes, started = [], None
    for target in targets(info):
        attempts = 0
        started = clock() if started is None else started
        while True:
            attempts += 1
            code, body = call(url, CLOSE, target)
            if code != 409 or clock() - started >= limit:
                break
            sleep(2)
        closes.append(dict(target, http=code, attempts=attempts,
                           receipts=(body or {}).get('receipts') if type(body) is dict else None))
    status, after = call(url, INFO)
    left = open_sessions(after) if status == 200 and type(after) is dict else None
    ok = left == [] and all(c['http'] == 200 for c in closes)
    value = dict(status='SESSIONS_CLOSED' if ok else 'SESSIONS_LEFT_OPEN', closed=len(closes),
                 open_before=len(open_sessions(info)), closes=closes,
                 left_open=None if left is None else [dict(key_sha256=s.get('key_sha256'),
                                                           session_id=s.get('session_id'), busy=s.get('busy'))
                                                      for s in left])
    if left is None:
        value['error'] = 'GET %s after the closes: HTTP %s' % (INFO, status)
    return value


def check(url, job, profile, call=request):
    """The reasons the server does not serve this job (empty: it does)."""
    reasons = []
    agent = job['agents'][0]
    kwargs = agent['kwargs']
    stock = 'llm_call_kwargs' in kwargs
    served = agent['model_name'].split('/', 1)[1]
    status, models = call(url, MODELS)
    ids = [m.get('id') for m in (models or {}).get('data') or ()] if type(models) is dict else []
    if status != 200 or served not in ids:
        reasons.append('GET %s: HTTP %s, models %s (the job\'s model is %s)' % (MODELS, status, ids, served))
    status, info = call(url, INFO)
    if stock:
        if status == 200:
            reasons.append('a --stock job, but the server answers %s (this runtime, not a stock vLLM control)' % INFO)
        return reasons
    if status != 200 or type(info) is not dict:
        return reasons + ['GET %s: HTTP %s %s (not a chat-text server)' % (INFO, status, info)]
    options, counts = info.get('options') or {}, info.get('counts') or {}
    model_info = kwargs['model_info']
    expected = dict(protocol=PROTOCOL, max_model_len=model_info['max_input_tokens'],
                    max_tokens=model_info['max_output_tokens'], model_profile=MODEL_PROFILE[profile])
    found = dict(protocol=info.get('protocol'), max_model_len=options.get('max_model_len'),
                 max_tokens=options.get('max_tokens'), model_profile=options.get('model_profile'))
    for key in expected:
        if found[key] != expected[key]:
            reasons.append('%s: the server has %r, the job needs %r' % (key, found[key], expected[key]))
    running = options.get('max_running_sessions')
    if type(running) is not int or running < job['n_concurrent_trials']:
        reasons.append('max_running_sessions %r is below the job\'s %d concurrent trials'
                       % (running, job['n_concurrent_trials']))
    if counts.get('client_sessions') != 0 or open_sessions(info):
        reasons.append('the server already has client sessions (%r): start a fresh server for each run'
                       % counts.get('client_sessions'))
    return reasons


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('--base-url', required=True, type=base_url)
    c.add_argument('--job', required=True, type=Path)
    c.add_argument('--profile', required=True, choices=sorted(M.PROFILES))
    x = sub.add_parser('close')
    x.add_argument('--base-url', required=True, type=base_url)
    x.add_argument('--limit', type=float, default=60.0, help='seconds of 409 retries in all (default 60)')
    a = p.parse_args(argv)
    if a.command == 'close':
        value = close_all(a.base_url, a.limit)
        print(json.dumps(value, sort_keys=True))
        return 0 if value['status'] == 'SESSIONS_CLOSED' else 1
    try:
        job = M.load(a.job)
        M.check_fields(job)
        reasons = check(a.base_url, job, a.profile)
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        reasons = ['not a job file of make_job.py: %s: %s' % (type(error).__name__, error)]
    print(json.dumps(dict(status='SERVER_REFUSED' if reasons else 'SERVER_MATCHES_JOB', reasons=reasons,
                          base_url=a.base_url, profile=a.profile), sort_keys=True))
    return 1 if reasons else 0


if __name__ == '__main__':
    sys.exit(main())

"""Stand-in for a gswe32 server at the far end of the tunnel (stdlib only): the 16-agent stress test of tbench (no GPU,
no DeltaAI). Serves the OpenAI chat route on a Unix socket (as the reverse tunnel does) or, for local tests, on
127.0.0.1 (`tcp:<port>`): every task (told apart by its task message) gets scripted Mini-SWE-Agent XML actions that do
real work in the task's container (git, grep, find, file reads, the repository's test collection, a small script),
after a model-like delay, and submits after a per-task number of steps. It answers the session close route like the
runtime (200 with a receipt; a repeated close says so) and logs every request's shape (body bytes, keys, the OpenAI
client's retry and timeout headers) so the stress test can show what litellm sends.

Usage: fake_server.py <socket path | tcp:PORT> <request log.jsonl> [min delay s] [max delay s] [min steps] [max steps]
"""
import hashlib
import json
import os
import random
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler

TARGET, LOG = sys.argv[1], sys.argv[2]
DELAY = (float(sys.argv[3]) if len(sys.argv) > 3 else 2.0, float(sys.argv[4]) if len(sys.argv) > 4 else 8.0)
STEPS = (int(sys.argv[5]) if len(sys.argv) > 5 else 18, int(sys.argv[6]) if len(sys.argv) > 6 else 26)
COMMANDS = [
    "cd /testbed && git status --short | head -20 && git log --oneline -3",
    "cd /testbed && ls && find . -name '*.py' -path '*test*' | head -40",
    "cd /testbed && grep -rn 'def ' --include=*.py . | head -400 | tail -60",
    "cd /testbed && python -c 'import sys; print(sys.version)' && pip list 2>/dev/null | wc -l",
    "cd /testbed && cat \"$(git ls-files '*.py' | head -1)\" | head -150",
    "cd /testbed && (timeout 100 python -m pytest --collect-only -q 2>&1 || timeout 100 python tests/runtests.py --help 2>&1) | tail -8",
    "cd /testbed && sed -n '1,80p' setup.cfg 2>/dev/null || ls -la",
    "cd /testbed && printf 'print(sum(range(10)))\\n' > /tmp/gswe32_probe.py && python /tmp/gswe32_probe.py",
    "cd /testbed && git grep -n 'import' -- '*.py' | head -200 | tail -40",
    "cd /testbed && du -sh . && git ls-files | wc -l",
]
LOCK = threading.Lock()
CLOSED = {}


def steps_for(task):
    rnd = random.Random(int(task[:8], 16))
    return rnd.randint(*STEPS)


def reply_for(task, step):
    total = steps_for(task)
    if step >= total:
        return ("THOUGHT: Submit the (empty) patch.\n\n<mswea_bash_command>echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
                "cat /testbed/patch.txt</mswea_bash_command>")
    if step == total - 1:
        return "THOUGHT: Write the patch.\n\n<mswea_bash_command>cd /testbed && git diff > patch.txt && wc -c patch.txt</mswea_bash_command>"
    pad = ' Let me look at more of the repository.' * (40 if step % 7 == 3 else 2)
    return 'THOUGHT: Step %d of the stress test.%s\n\n<mswea_bash_command>%s</mswea_bash_command>' % (
        step, pad, COMMANDS[(step + int(task[:4], 16)) % len(COMMANDS)])


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def address_string(self):
        return 'local'

    def send(self, status, value):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip('/') == '/v1/models':
            self.send(200, {'object': 'list', 'data': [{'id': 'selective-prefix', 'object': 'model'}]})
        elif self.path == '/health':
            self.send(200, {})
        else:
            self.send(404, {'detail': 'Not Found'})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get('Content-Length') or 0))
        body = json.loads(raw or b'{}')
        if self.path == '/v1/sessions/close':
            instance = body.get('instance_id')
            with LOCK:
                repeated = instance in CLOSED
                CLOSED.setdefault(instance, time.time())
            self.send(200, {'object': 'session.close', 'instance_id': instance, 'repeated': repeated, 'fake': True})
            return
        messages = body.get('messages') or []
        task = hashlib.sha256((messages[1].get('content', '') if len(messages) > 1 else '').encode()).hexdigest()
        step = sum(1 for m in messages if m.get('role') == 'assistant')
        delay = random.uniform(*DELAY)
        headers = {k.lower(): v for k, v in self.headers.items() if k.lower().startswith(('x-stainless', 'user-agent'))}
        record = dict(t=round(time.time(), 3), task=task[:16], step=step, bytes=len(raw), keys=sorted(body),
                      sampling={k: body.get(k) for k in sorted(body) if k != 'messages'}, headers=headers,
                      delay_s=round(delay, 2), messages=len(messages))
        with LOCK:
            with open(LOG, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, sort_keys=True) + '\n')
        time.sleep(delay)
        content = reply_for(task, step)
        self.send(200, {'id': 'chatcmpl-fake-%s-%d' % (task[:8], step), 'object': 'chat.completion',
                        'created': int(time.time()), 'model': body.get('model', 'selective-prefix'),
                        'choices': [{'index': 0, 'finish_reason': 'stop',
                                     'message': {'role': 'assistant', 'content': content}}],
                        'usage': {'prompt_tokens': len(raw) // 4, 'completion_tokens': len(content) // 4,
                                  'total_tokens': len(raw) // 4 + len(content) // 4}})

    def log_message(self, *args):
        pass


if TARGET.startswith('tcp:'):
    class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True
        allow_reuse_address = True
    server = Server(('127.0.0.1', int(TARGET[4:])), Handler)
else:
    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

        def get_request(self):
            request, _ = self.socket.accept()
            return request, ('local', 0)
    if os.path.exists(TARGET):
        os.unlink(TARGET)
    os.umask(0o177)
    server = Server(TARGET, Handler)
print('FAKE_SERVING %s' % TARGET, flush=True)
server.serve_forever()

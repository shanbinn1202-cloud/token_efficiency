"""Stand-in for a tb16 server at the far end of the tunnel (stdlib only): the stress test of tbench (real Harbor 0.21.0
and Terminus 2 against a fake model; no GPU, no DeltaAI). Serves the OpenAI chat route on a Unix socket (as the
reverse tunnel does) or, for local tests, on 127.0.0.1 (`tcp:<port>`). Every trial (told apart by its first message,
which carries the task instruction and the container's prompt) gets scripted Terminus 2 JSON replies that do real work
in the task's container (`pwd`, `ls -la`, a file listing, `uname -a`, ...) after a model-like delay, then marks the
task complete and confirms it; an interleaved summarization or questions request (a history whose first message
differs) gets a short plain answer. It logs every request's shape (body bytes and keys, the sampling fields, an
X-Session-ID header if any, the message roles), so the stress test shows what Terminus 2 / litellm send.

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
DELAY = (float(sys.argv[3]) if len(sys.argv) > 3 else 1.0, float(sys.argv[4]) if len(sys.argv) > 4 else 4.0)
STEPS = (int(sys.argv[5]) if len(sys.argv) > 5 else 2, int(sys.argv[6]) if len(sys.argv) > 6 else 4)
COMMANDS = ['pwd && ls -la\n', 'ls -la / | head -30\n', 'uname -a && cat /etc/os-release | head -3\n',
            'find . -maxdepth 2 | head -40\n', 'df -h . && free -m\n', 'env | sort | head -20\n']
LOCK = threading.Lock()


def steps_for(task):
    return random.Random(int(task[:8], 16)).randint(*STEPS)


def terminus(analysis, plan, commands, complete):
    return json.dumps(dict(analysis=analysis, plan=plan, commands=commands, task_complete=complete), indent=2)


def reply_for(task, step):
    """Terminus 2's JSON format (terminus_json_plain_parser): commands while working, then task_complete twice (the
    second answers Terminus' confirmation prompt)."""
    total = steps_for(task)
    if step >= total:
        return terminus('The stress test is done.', 'Mark the task complete.', [], True)
    cmd = COMMANDS[(step + int(task[:4], 16)) % len(COMMANDS)]
    return terminus('Stress-test step %d of %d.' % (step + 1, total), 'Run one inspection command.',
                    [dict(keystrokes=cmd, duration=1.0)], False)


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
        if self.path != '/v1/chat/completions':
            self.send(404, {'detail': 'Not Found'})
            return
        messages = body.get('messages') or []
        first = messages[0].get('content', '') if messages else ''
        task = hashlib.sha256(str(first).encode()).hexdigest()
        step = sum(1 for m in messages if m.get('role') == 'assistant')
        delay = random.uniform(*DELAY)
        headers = {k.lower(): v for k, v in self.headers.items()
                   if k.lower().startswith(('x-stainless', 'user-agent', 'x-session-id'))}
        aside = 'Terminal Output' not in str(first) and 'Current Terminal Screen' not in str(first) \
            and 'tasked with solving command-line tasks' not in str(first)
        record = dict(t=round(time.time(), 3), task=task[:16], step=step, bytes=len(raw), keys=sorted(body),
                      sampling={k: body.get(k) for k in sorted(body) if k != 'messages'}, headers=headers,
                      roles=[m.get('role') for m in messages][:4], delay_s=round(delay, 2), messages=len(messages),
                      aside=aside)
        with LOCK:
            with open(LOG, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, sort_keys=True) + '\n')
        time.sleep(delay)
        content = 'A short summary of the work so far.' if aside else reply_for(task, step)
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

"""Bounded authenticated duplex frames, independent of public HTTP inputs."""
import base64
import hashlib
import hmac
import json
import threading

MAX_FRAME = 4 * 1024 * 1024
MAX_CORE = 1024 * 1024
MAX_SEQUENCE = 8192
DIRECTIONS = {'frontend-engine', 'engine-frontend', 'engine-worker', 'worker-engine'}
FIELDS = {
    'capability': {'challenge'},
    'capability_result': {'challenge', 'native_ready', 'live_multi_turn_supported', 'missing'},
    'add': {'request', 'session', 'turn', 'generation', 'core', 'source'},
    'cancel': {'request', 'session', 'turn', 'generation'},
    'ack': {'request', 'session', 'turn', 'generation', 'state', 'binding'},
    'worker_command': {'request', 'session', 'turn', 'generation', 'operation', 'binding', 'payload'},
    'worker_receipt': {'request', 'session', 'turn', 'generation', 'operation', 'binding', 'receipt'},
}
ALLOWED = {
    'frontend-engine': {'capability', 'add', 'cancel'},
    'engine-frontend': {'capability_result', 'ack'},
    'engine-worker': {'worker_command'}, 'worker-engine': {'worker_receipt'},
}


class TransportError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise TransportError(message)


def encoded(obj):
    # Preserve insertion order in trusted template arguments: sorting could change rendering.
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON field')
        result[key] = value
    return result


def decoded(raw):
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TransportError('invalid bounded JSON') from exc


def hex64(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def identity(body):
    for key in ('request', 'session', 'generation'):
        require(type(body[key]) is str and 0 < len(body[key]) <= 256, 'invalid private identity')
    require(type(body['turn']) is int and 0 <= body['turn'] < 256, 'invalid private turn')
    return tuple(body[k] for k in ('request', 'session', 'turn', 'generation'))


class Channel:
    """One direction, one owner thread; sequence consumed before dispatch errors."""
    def __init__(self, key, epoch, source, direction):
        require(type(key) is bytes and len(key) == 32 and hex64(epoch) and hex64(source)
                and direction in DIRECTIONS, 'startup credential required')
        self.key = hmac.new(key, direction.encode(), hashlib.sha256).digest()
        self.epoch, self.source, self.direction = epoch, source, direction
        self.sent = self.received = 0
        self.thread = threading.get_ident()

    def owner(self):
        require(threading.get_ident() == self.thread, 'wrong private channel owner thread')

    def validate(self, kind, body):
        require(kind in ALLOWED[self.direction] and type(body) is dict
                and set(body) == FIELDS[kind], 'invalid direction or exact command schema')
        if kind in ('capability', 'capability_result'):
            require(hex64(body['challenge']), 'challenge required')
        else:
            identity(body)
        if kind == 'capability_result':
            require(type(body['native_ready']) is bool and type(body['live_multi_turn_supported']) is bool
                    and type(body['missing']) is list and len(body['missing']) <= 32
                    and all(type(v) is str and len(v) <= 256 for v in body['missing']), 'invalid capability')
        if kind == 'add':
            require(type(body['core']) is str and type(body['source']) is dict, 'invalid add payload')
            try:
                raw = base64.b64decode(body['core'], validate=True)
            except (ValueError, TypeError) as exc:
                raise TransportError('invalid core encoding') from exc
            require(0 < len(raw) <= MAX_CORE, 'core size limit')
        if kind == 'ack':
            require(body['state'] in {'accepted', 'complete', 'cancelled', 'rejected'}
                    and (body['binding'] == '' or hex64(body['binding'])), 'invalid acknowledgement')
        if kind.startswith('worker_'):
            require(body['operation'] in {'prepare', 'append', 'complete', 'cancel'}
                    and hex64(body['binding']), 'invalid worker command')
            require(type(body['payload' if kind == 'worker_command' else 'receipt']) is dict,
                    'invalid worker payload')

    def send(self, kind, body):
        self.owner(); self.validate(kind, body)
        require(self.sent < MAX_SEQUENCE, 'channel sequence exhausted')
        raw = encoded(dict(version=2, epoch=self.epoch, source=self.source,
                           direction=self.direction, sequence=self.sent, kind=kind, body=body))
        frame = encoded(dict(payload=base64.b64encode(raw).decode(),
                             mac=hmac.new(self.key, raw, hashlib.sha256).hexdigest()))
        require(len(frame) <= MAX_FRAME, 'frame size limit')
        self.sent += 1
        return frame

    def receive(self, frame):
        self.owner()
        require(type(frame) is bytes and 0 < len(frame) <= MAX_FRAME, 'frame size limit')
        outer = decoded(frame)
        require(type(outer) is dict and set(outer) == {'payload', 'mac'}
                and type(outer['payload']) is str and hex64(outer['mac']), 'invalid envelope')
        try:
            raw = base64.b64decode(outer['payload'], validate=True)
        except (ValueError, TypeError) as exc:
            raise TransportError('invalid envelope encoding') from exc
        require(hmac.compare_digest(outer['mac'], hmac.new(self.key, raw, hashlib.sha256).hexdigest()),
                'authentication failed')
        obj = decoded(raw)
        require(type(obj) is dict and set(obj) == {'version','epoch','source','direction','sequence','kind','body'}
                and type(obj['version']) is int and obj['version'] == 2
                and obj['epoch'] == self.epoch and obj['source'] == self.source
                and obj['direction'] == self.direction, 'channel identity mismatch')
        # Replies can be lost when their bounded caller expires. Commands must
        # remain contiguous; replies use a monotonic high-water mark.
        reply = self.direction in {'engine-frontend', 'worker-engine'}
        require(type(obj['sequence']) is int and 0 <= obj['sequence'] < MAX_SEQUENCE
                and (obj['sequence'] >= self.received if reply else obj['sequence'] == self.received),
                'replay or out-of-order frame')
        self.validate(obj['kind'], obj['body'])
        self.received = obj['sequence'] + 1
        return obj['kind'], obj['body']

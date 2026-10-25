"""Multi-session stage 1: one server, N startup-fixed tasks, one session per task.

The contract shared by the frontend and the engine
(development/multi-session/STAGE1_CONTRACT.md):

- The startup config lists the tasks. Session k serves ``tasks[k]`` for the
  whole process; nothing at runtime adds, removes or renames a session.
- Session ids derive from the server id, so the frontend and the engine agree
  without another channel. Session 0 keeps the server id, so a one-task server
  behaves exactly as before.
- A request is routed by its task message (``messages[1]``). HTTP never names a
  session, so routing requires tasks whose problem statements cannot be
  confused: no statement may contain another.

``MAX_SESSIONS`` is the one cap on sessions per server; every per-process
session cap on the live path reads it (startup task list, session ids,
EngineLifecycle, the native adapter). Stage 1 set it to 8; it is 16 since
branch ms16 (not yet validated on a GPU above 5 sessions).

Session pool (branch ms-pool; startup option ``--max-running-sessions K``,
pinned as ``max_running_sessions``): the startup config registers up to
``MAX_TASKS`` tasks, still one session id per task and still routed by the
task message, but at most K of them are open at a time (K <= MAX_SESSIONS,
``max_num_seqs`` = K). A session opens at its task's first request (turn 0),
waiting first while K sessions are open, and closes when the harness closes
it (its agent finished) or at shutdown; a closed session never reopens. The
pinned task list is what integrity covers, exactly as before: tasks are only
opened and closed, never added. Without the option nothing here changes.
"""
from bisect import bisect_right

from .contract import require

MAX_SESSIONS = 16
# Registered tasks of a session-pool server (startup option --max-running-sessions).
MAX_TASKS = 1024
# Registered slots of a chat-text server (serve.py --protocol chat-text,
# chat_text_protocol.MAX_SLOTS): its sessions bind lazily, one slot per client
# session and per rebind, so it registers more than the other pools. Only its own
# startup config (protocol chat_text_v1) reads this cap.
CHAT_TEXT_PROTOCOL = 'chat_text_v1'
MAX_CHAT_TEXT_SLOTS = 1024
# Open (running) sessions of a chat-text pool: Harbor's 16 concurrent Terminus trials
# plus spare sessions for finished trials (no close call arrives) and Terminus's
# summary sub-calls, so that no running trial is evicted while a spare exists
# (K = 20 is the Terminal-Bench setting). Every other server keeps MAX_SESSIONS.
MAX_CHAT_TEXT_RUNNING = 24
TASK_KEYS = frozenset({'instance_id', 'problem_statement'})
# Lifetime request bound of one engine process (lifecycle entries, request
# tombstones, minted render bindings; none is ever evicted): every session's own
# request bound, MAX_TURNS, summed over the sessions (1,024 up to 4 sessions, as
# before). MAX_TURNS is the per-session bound the frontend controller (max_turns),
# the native adapter (turn ordinal) and the session authority (MAX_GENERATIONS)
# enforce, so the lifetime bound can never refuse a request a session could still
# make: 120-step agents (at most 120 requests per task; a CAPACITY_WAIT retry takes
# no record) fit at any task count. It was 128 per session (4,096 at 32 tasks, 6.7 %
# above 32 x 120).
MIN_REQUEST_LIMIT = 1024
MAX_TURNS = 256
REQUESTS_PER_SESSION = MAX_TURNS


def task_limit(pooled, protocol=None):
    """Startup tasks per server: MAX_SESSIONS, or MAX_TASKS on a session-pool server
    (MAX_CHAT_TEXT_SLOTS on a chat-text one, whose pool registers its slots)."""
    if pooled and protocol == CHAT_TEXT_PROTOCOL:
        return MAX_CHAT_TEXT_SLOTS
    return MAX_TASKS if pooled else MAX_SESSIONS


def request_limit(count, limit=MAX_SESSIONS):
    """Lifetime requests of a server with ``count`` sessions: max(1,024, 256 x count).

    A session-pool server passes its registered task count and ``limit=MAX_TASKS``
    (a chat-text server ``MAX_CHAT_TEXT_SLOTS``)."""
    require(type(count) is int and 1 <= count <= limit, 'session count required')
    return max(MIN_REQUEST_LIMIT, REQUESTS_PER_SESSION * count)


def validate_tasks(tasks, limit=MAX_SESSIONS):
    """Startup task list: 1..limit frozen issues, unambiguous to route."""
    require(type(tasks) is list and 1 <= len(tasks) <= limit,
            'startup task list of 1..%d tasks required' % limit)
    for task in tasks:
        require(type(task) is dict and set(task) == TASK_KEYS and
                all(type(v) is str and v.strip() for v in task.values()),
                'each task needs instance_id and problem_statement')
    ids = [t['instance_id'] for t in tasks]
    require(len(set(ids)) == len(ids), 'task instance ids must be distinct')
    statements = [t['problem_statement'] for t in tasks]
    # A statement can only lie inside a longer one or equal another, so only longer
    # statements are searched (the same answer as every ordered pair, without the
    # quadratic scan over equally long ones: a chat-text server's up to 1,024 slot
    # statements, which config.load checks on every call, have one length).
    distinct = sorted(set(statements), key=len)
    lengths = [len(s) for s in distinct]
    require(len(distinct) == len(statements) and
            all(a not in b for a in distinct for b in distinct[bisect_right(lengths, len(a)):]),
            'no problem statement may contain another: routing would be ambiguous')
    return tasks


def running_cap(protocol=None):
    """The largest K of a session pool: MAX_CHAT_TEXT_RUNNING on a chat-text server,
    MAX_SESSIONS on every other one."""
    return MAX_CHAT_TEXT_RUNNING if protocol == CHAT_TEXT_PROTOCOL else MAX_SESSIONS


def running_limit(value, tasks, protocol=None):
    """A session-pool server's K (``max_running_sessions``) for its startup tasks:
    1 <= K <= min(running_cap(protocol), len(tasks)); None (no option) keeps every
    task open."""
    if value is None:
        return None
    cap = running_cap(protocol)
    require(type(value) is int and type(tasks) is list and
            1 <= value <= min(cap, len(tasks)),
            'max_running_sessions must be 1..%d and at most the number of tasks' % cap)
    return value


def session_ids(server_id, count, limit=MAX_SESSIONS):
    """Session k's id: the server id for k = 0, ``<server id>-s<k>`` after that."""
    require(type(server_id) is str and server_id and type(count) is int and
            1 <= count <= limit, 'server id and session count required')
    return [server_id] + ['%s-s%d' % (server_id, k) for k in range(1, count)]


def session_index(server_id, count, session_id, limit=MAX_SESSIONS):
    """The task index a session id belongs to; rejects ids this server never issued."""
    ids = session_ids(server_id, count, limit)
    require(session_id in ids, 'session not initialized by loaded server')
    return ids.index(session_id)


def route(tasks, messages):
    """The task index of a request: the one task whose statement is in messages[1]."""
    require(type(messages) is list and len(messages) >= 2 and type(messages[1]) is dict and
            messages[1].get('role') == 'user' and type(messages[1].get('content')) is str,
            'task message required')
    content = messages[1]['content']
    hits = [k for k, task in enumerate(tasks) if task['problem_statement'] in content]
    require(len(hits) == 1, 'request matches no startup task' if not hits
            else 'request matches several startup tasks')
    return hits[0]

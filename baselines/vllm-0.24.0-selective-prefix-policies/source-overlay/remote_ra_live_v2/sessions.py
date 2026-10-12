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
"""
from .contract import require

MAX_SESSIONS = 8
TASK_KEYS = frozenset({'instance_id', 'problem_statement'})


def validate_tasks(tasks):
    """Startup task list: 1..MAX_SESSIONS frozen issues, unambiguous to route."""
    require(type(tasks) is list and 1 <= len(tasks) <= MAX_SESSIONS,
            'startup task list of 1..%d tasks required' % MAX_SESSIONS)
    for task in tasks:
        require(type(task) is dict and set(task) == TASK_KEYS and
                all(type(v) is str and v.strip() for v in task.values()),
                'each task needs instance_id and problem_statement')
    ids = [t['instance_id'] for t in tasks]
    require(len(set(ids)) == len(ids), 'task instance ids must be distinct')
    statements = [t['problem_statement'] for t in tasks]
    require(all(a not in b for i, a in enumerate(statements)
                for j, b in enumerate(statements) if i != j),
            'no problem statement may contain another: routing would be ambiguous')
    return tasks


def session_ids(server_id, count):
    """Session k's id: the server id for k = 0, ``<server id>-s<k>`` after that."""
    require(type(server_id) is str and server_id and type(count) is int and
            1 <= count <= MAX_SESSIONS, 'server id and session count required')
    return [server_id] + ['%s-s%d' % (server_id, k) for k in range(1, count)]


def session_index(server_id, count, session_id):
    """The task index a session id belongs to; rejects ids this server never issued."""
    ids = session_ids(server_id, count)
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

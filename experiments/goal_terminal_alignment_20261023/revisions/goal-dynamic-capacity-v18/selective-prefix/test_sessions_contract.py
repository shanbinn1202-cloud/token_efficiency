"""Multi-session stage-1 contract: task list, session ids, routing, config. CPU only."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
from remote_ra_live_v2 import sessions  # noqa: E402

A = dict(instance_id='repo__a-1', problem_statement='Fix the off-by-one error in window_sum.')
B = dict(instance_id='repo__b-2', problem_statement='Make parse_date accept ISO week dates.')


def chat(task, extra=()):
    return [dict(role='system', content='You are a helpful assistant.'),
            dict(role='user', content='<pr_description>\n%s\n</pr_description>' % task['problem_statement']),
            *extra]


class Tasks(unittest.TestCase):
    def test_valid_lists(self):
        self.assertEqual(sessions.validate_tasks([A]), [A])
        self.assertEqual(sessions.validate_tasks([A, B]), [A, B])

    def test_rejects_bad_lists(self):
        bad = [[], [A] * 2, [A, dict(B, instance_id=A['instance_id'])],
               [A, dict(B, problem_statement=A['problem_statement'] + ' Also add a test.')],
               [dict(A, extra='x')], [dict(A, problem_statement='  ')], (A,),
               [dict(A, instance_id='t%d' % k, problem_statement='Task number %03d.' % k)
                for k in range(sessions.MAX_SESSIONS + 1)]]
        for tasks in bad:
            with self.assertRaises(RuntimeError, msg=repr(tasks)[:80]):
                sessions.validate_tasks(tasks)


class Ids(unittest.TestCase):
    def test_session_zero_keeps_the_server_id(self):
        self.assertEqual(sessions.session_ids('http-abc', 1), ['http-abc'])
        self.assertEqual(sessions.session_ids('http-abc', 3), ['http-abc', 'http-abc-s1', 'http-abc-s2'])

    def test_index_round_trip_and_foreign_ids(self):
        ids = sessions.session_ids('http-abc', 3)
        self.assertEqual([sessions.session_index('http-abc', 3, s) for s in ids], [0, 1, 2])
        for foreign in ('http-abc-s3', 'http-abd', 'http-abc-s0', ''):
            with self.assertRaises(RuntimeError):
                sessions.session_index('http-abc', 3, foreign)
        with self.assertRaises(RuntimeError):
            sessions.session_ids('http-abc', 0)


class Routing(unittest.TestCase):
    def test_routes_by_task_message(self):
        tasks = [A, B]
        self.assertEqual(sessions.route(tasks, chat(B)), 1)
        later = chat(A, [dict(role='assistant', content='ls'), dict(role='user', content=B['problem_statement'])])
        self.assertEqual(sessions.route(tasks, later), 0)   # only messages[1] decides

    def test_rejects_unknown_or_malformed(self):
        with self.assertRaises(RuntimeError):
            sessions.route([A], chat(B))
        with self.assertRaises(RuntimeError):
            sessions.route([A, B], [dict(role='system', content='x')])
        with self.assertRaises(RuntimeError):
            sessions.route([A, B], [dict(role='system', content='x'), dict(role='assistant', content=A['problem_statement'])])


class Config(unittest.TestCase):
    """remote_ra_http.config: a ``tasks`` list is optional; ``task`` stays tasks[0]."""

    def load(self, **changes):
        from remote_ra_http import config
        value = dict(schema='remote-ra-http-config-v2', policy='full', protocol='mswea_xml_v1',
                     selection_contract=None, task=A, page_tokens=784, selection_seed=42,
                     selection_fraction=.5, warmup_requests=0, output_directory=str(Path(tempfile.gettempdir()).resolve()),
                     input_pins_sha256='0' * 64, task_sha256='0' * 64, common_config_sha256='0' * 64,
                     issue_sha256=sha256(A['problem_statement'].encode()).hexdigest())
        value.update(changes)
        raw = json.dumps(value).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'startup.json'
            path.write_bytes(raw)
            env = dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                       REMOTE_RA_HTTP_CONFIG_SHA256=sha256(raw).hexdigest())
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                return config.load()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_single_task_config_is_normalized(self):
        self.assertEqual(self.load()['tasks'], [A])

    def test_task_list(self):
        from remote_ra_http.config import digest
        value = self.load(tasks=[A, B], tasks_sha256=digest([A, B]))
        self.assertEqual(value['tasks'], [A, B])
        self.assertEqual(value['task'], A)

    def test_task_list_checks(self):
        from remote_ra_http.config import digest
        for changes in (dict(tasks=[B, A], tasks_sha256=digest([B, A])),     # tasks[0] must be task
                        dict(tasks=[A, B]),                                  # pin required
                        dict(tasks=[A, B], tasks_sha256=digest([A])),        # pin must match
                        dict(tasks=[A, A], tasks_sha256=digest([A, A]))):    # validate_tasks
            with self.assertRaises(RuntimeError, msg=str(changes)[:80]):
                self.load(**changes)


if __name__ == '__main__':
    unittest.main()

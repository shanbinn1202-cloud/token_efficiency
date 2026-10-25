"""chat-text replay of Terminal-Bench 2.0 Terminus 2 trajectories (ATIF), protocol level. CPU only.

Opt-in: set REMOTE_RA_TB_ATIF_DIR to a directory of Harbor job directories
(``<job>/<trial>/agent/trajectory.json``, ATIF-v1.7, e.g. the August Terminus 2 runs
2026-08-19__18-08-51 and 2026-08-20__09-40-12 copied from tbench) and run it where the
pinned tokenizer is available (Transformers, the remote-ra-tok venv). The trajectories are
not part of this repository (they hold task prompts and terminal output).

Each trial is replayed by test_chat_text_protocol's Terminus client against the
protocol-level server, with the server limits the TB plan fixes (65,536-token context,
4,096-token replies, the client's model_info max_input_tokens 65,536 and its 8,000-token
summary threshold): the first message is the trial's first prompt, each main reply is the
trial's reply rebuilt as Terminus JSON (analysis, plan, commands), and each observation is
the trial's next prompt. Summary, questions, answers and short-summary sub-calls get short
scripted replies. The token counter is the pinned tokenizer's count of the message texts, exact and 20 % low
(SCALES).

Checked for every request: a refusal (400 with the context text) changes nothing; every
served prompt equals the pinned template's render of its messages (enable_thinking False,
preserve_thinking True), unless an earlier reply of its session was cut at its bound and does
not re-encode to its own ids; the main session stays on one slot unless it is rebound, the
questions and short-summary calls take their own slots, and the summary hand-off and the
unwound histories are rebuilds. A summary of the counts is printed.
"""
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_chat_text_protocol as T  # noqa: E402

ATIF = os.environ.get('REMOTE_RA_TB_ATIF_DIR')
CONTEXT, BOUND, THRESHOLD = 65536, 4096, 8000
# The client's token estimate: exact, and 20 % low (litellm's token_counter for a model it does not
# know undercounts Qwen tokens; the August build-pov-ray run hit the server's context limit).
SCALES = (1.0, 0.8)


def trajectories(root):
    return sorted(Path(root).glob('*/*/agent/trajectory.json'))


def reply_of(step):
    """A main reply rebuilt as Terminus JSON from the ATIF agent step."""
    message = step.get('message') or ''
    analysis, _, plan = message.partition('\nPlan: ')
    analysis = analysis[len('Analysis: '):] if analysis.startswith('Analysis: ') else analysis
    commands = [dict(keystrokes=c.get('arguments', {}).get('keystrokes', ''),
                     duration=c.get('arguments', {}).get('duration', 1.0))
                for c in step.get('tool_calls') or () if c.get('function_name') == 'bash_command']
    complete = any(c.get('function_name') == 'mark_task_complete' for c in step.get('tool_calls') or ())
    return json.dumps(dict(analysis=analysis, plan=plan, commands=commands, task_complete=complete), indent=2,
                      ensure_ascii=False)


def episode(path):
    """(first prompt, [reply], [observation]) of one trial: the observation after reply k is
    the prompt of episode k + 1."""
    steps = json.loads(path.read_text(encoding='utf-8'))['steps']
    first = steps[0]['message']
    replies, observations = [], []
    for step in steps[1:]:
        if step.get('source') != 'agent':
            continue
        replies.append(reply_of(step))
        results = (step.get('observation') or {}).get('results') or []
        observations.append(''.join(r.get('content') or '' for r in results) or 'New Terminal Output:\n')
    return first, replies, observations


SUB_REPLIES = {
    'You are about to hand off': 'Summary: I explored the environment, ran the build and checked its output.',
    'You are picking up work': '1. Which files changed?\n2. Does the build pass?\n3. What failed last?',
    'The next agent has a few questions': 'Answers: the files are under /app; the build fails at the link step.',
    'Briefly continue this task': 'Continue with the build and check the logs.',
}


def replay(kit, path, scale=1.0):
    first, replies, observations = episode(path)
    script = iter(replies)
    counts = dict(requests=0, refused=0, rebuilds=0, asides=0, mismatches=0, cut=0, max_prompt=0)

    def model(messages, slot):
        last = messages[-1]['content']
        for prefix, text in SUB_REPLIES.items():
            if last.startswith(prefix) or messages[0]['content'].startswith(prefix):
                return text
        return next(script, T.action(len(messages), complete=True))
    server = T.ProtocolServer(kit, slots=256, limit=CONTEXT, bound=BOUND, model=model)
    count = lambda messages: int(scale * sum(len(kit.encode(m['content'])) + 4 for m in messages))
    agent = T.Terminus('instruction of %s' % path.parent.parent.name, send=server.send, count=count,
                       context_limit=CONTEXT, threshold=THRESHOLD)
    prompt = first
    for k in range(len(replies)):
        T.run(agent.step(prompt))
        prompt = observations[k]
    cut_slots = set()       # slots holding a reply cut at its bound (kept as its own ids)
    for entry in server.log:
        counts['requests'] += 1
        counts['rebuilds'] += entry['rebuild']
        counts['asides'] += entry['binding'] == 'bind' and entry['slot'] != server.log[0]['slot']
        counts['max_prompt'] = max(counts['max_prompt'], len(entry['prompt']))
        if entry['rebuild'] or entry['binding'] is not None:
            cut_slots.discard(entry['slot'])
        if entry['prompt'] != kit.reference(entry['messages']):
            counts['mismatches'] += 1
            assert entry['slot'] in cut_slots, '%s: a prompt differs from the template render' % path
        if entry['finish'] == 'length':
            cut_slots.add(entry['slot'])
    counts['refused'] = sum(1 for _, status, _ in agent.requests if status != 200)
    counts['cut'] = agent.events.count('output_length')
    counts['events'] = sorted(set(agent.events))
    return counts, server, agent


@unittest.skipUnless(ATIF, 'set REMOTE_RA_TB_ATIF_DIR to the copied ATIF job directories (not in the repository)')
class AtifReplayTests(unittest.TestCase):
    maxDiff = None

    def test_every_trial(self):
        kit = T.real_kit()
        if kit is None:
            self.skipTest('the replay needs the pinned tokenizer (Transformers, remote-ra-tok venv)')
        paths = trajectories(ATIF)
        only = os.environ.get('REMOTE_RA_TB_ATIF_ONLY')
        if only:
            paths = [p for p in paths if only in str(p)]
        self.assertTrue(paths)
        total = dict(trials=0, requests=0, refused=0, rebuilds=0, asides=0, mismatches=0, cut=0)
        for path, scale in [(p, s) for p in paths for s in SCALES]:
            with self.subTest(trial=str(path.parent.parent.relative_to(ATIF)), scale=scale):
                counts, server, agent = replay(kit, path, scale)
                total['trials'] += 1
                for name in ('requests', 'refused', 'rebuilds', 'asides', 'mismatches', 'cut'):
                    total[name] += counts[name]
                if counts['refused'] or counts['rebuilds'] or counts['asides']:
                    print('%-55s x%.2f %s' % (path.parent.parent.relative_to(ATIF), scale, counts))
                # A refusal is always the context text; every other request was served.
                for messages, status, body in agent.requests:
                    if status != 200:
                        self.assertEqual(status, 400)
                        self.assertTrue(T.harbor_context_error(body), body)
                self.assertLessEqual(counts['max_prompt'], CONTEXT)
        print('ATIF replay totals:', total)


if __name__ == '__main__':
    unittest.main()

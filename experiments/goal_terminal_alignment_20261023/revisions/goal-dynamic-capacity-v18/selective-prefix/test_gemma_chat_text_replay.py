"""chat-text replay of Terminal-Bench 2.0 Terminus 2 trajectories (ATIF) with Gemma's turns. CPU only.

Opt-in, as test_chat_text_replay (the Qwen replay): REMOTE_RA_TB_ATIF_DIR names a directory of
Harbor job directories (``<job>/<trial>/agent/trajectory.json``, e.g. the August Terminus 2
runs), and the pinned Gemma tokenizer and template are needed (REMOTE_RA_GEMMA4_METADATA and
Transformers, the remote-ra-tok venv). The trajectories are not part of this repository.

Each trial is replayed by test_chat_text_protocol's Terminus client against the
protocol-level server with Gemma's turns (test_gemma_chat_text.GemmaKit: the pinned
tokenizer, the template through Transformers with the content as OpenAI text parts and
thinking off) and the limits of the TB plan (65,536-token context, 4,096-token replies,
Terminus's 8,000-token summary threshold); each main reply is the trial's reply rebuilt as
Terminus JSON and ends with ``<turn|>``. The client's token estimate is the pinned
tokenizer's count, exact and 20 % low.

Checked for every request: a refusal (400 with the context text) changes nothing; every
served prompt is the template's render of its messages with the empty thought channel kept
after every past model turn, and without those channels it is stock vLLM's prompt token for
token, unless an earlier reply of its session was cut at its bound and does not re-encode to
its own ids; the questions and short-summary calls take their own slots, and the summary
hand-off and the unwound histories are rebuilds. A summary of the counts is printed.
"""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_chat_text_protocol as T  # noqa: E402
import test_chat_text_replay as Q  # noqa: E402  ATIF parsing, sub-call replies, the plan's limits
import test_gemma_chat_text as GT  # noqa: E402

ATIF = os.environ.get('REMOTE_RA_TB_ATIF_DIR')


def replay(kit, path, scale=1.0):
    first, replies, observations = Q.episode(path)
    script = iter(replies)
    counts = dict(requests=0, refused=0, rebuilds=0, asides=0, mismatches=0, stock_mismatches=0, cut=0, markers=0,
                  max_prompt=0, channels=0)

    def model(messages, slot):
        last = messages[-1]['content']
        for prefix, text in Q.SUB_REPLIES.items():
            if last.startswith(prefix) or messages[0]['content'].startswith(prefix):
                return text
        return next(script, T.action(len(messages), complete=True))
    server = T.ProtocolServer(kit, slots=256, limit=Q.CONTEXT, bound=Q.BOUND, model=model)
    count = lambda messages: int(scale * sum(len(kit.encode(m['content'])) + 4 for m in messages))
    agent = T.Terminus('instruction of %s' % path.parent.parent.name, send=server.send, count=count,
                       context_limit=Q.CONTEXT, threshold=Q.THRESHOLD)
    prompt = first
    for k in range(len(replies)):
        T.run(agent.step(prompt))
        prompt = observations[k]
    cut_slots = set()
    for entry in server.log:
        counts['requests'] += 1
        counts['rebuilds'] += entry['rebuild']
        counts['asides'] += entry['binding'] == 'bind' and entry['slot'] != server.log[0]['slot']
        counts['max_prompt'] = max(counts['max_prompt'], len(entry['prompt']))
        if entry['rebuild'] or entry['binding'] is not None:
            cut_slots.discard(entry['slot'])
        if any('<|turn>' in m['content'] for m in entry['messages']):
            counts['markers'] += 1          # turn markers inside content: no text-level reference
        else:
            if entry['prompt'] != kit.reference(entry['messages']):
                counts['mismatches'] += 1
                assert entry['slot'] in cut_slots, '%s: a prompt differs from the template render' % path
            stock = kit.stock(entry['messages'])
            if kit.without_channels(entry['prompt']) != stock:
                counts['stock_mismatches'] += 1
                assert entry['slot'] in cut_slots, '%s: a prompt differs from stock vLLM' % path
            counts['channels'] += len(entry['prompt']) - len(stock)
        if entry['finish'] == 'length':
            cut_slots.add(entry['slot'])
    counts['refused'] = sum(1 for _, status, _ in agent.requests if status != 200)
    counts['cut'] = agent.events.count('output_length')
    counts['events'] = sorted(set(agent.events))
    return counts, server, agent


@unittest.skipUnless(ATIF, 'set REMOTE_RA_TB_ATIF_DIR to the copied ATIF job directories (not in the repository)')
class GemmaAtifReplayTests(unittest.TestCase):
    maxDiff = None

    def test_every_trial(self):
        kit = GT.real_kit()
        if kit is None:
            self.skipTest('the replay needs the pinned Gemma tokenizer (%s, Transformers)' % GT.G.ENV)
        paths = Q.trajectories(ATIF)
        only = os.environ.get('REMOTE_RA_TB_ATIF_ONLY')
        if only:
            paths = [p for p in paths if only in str(p)]
        self.assertTrue(paths)
        names = ('requests', 'refused', 'rebuilds', 'asides', 'mismatches', 'stock_mismatches', 'cut', 'markers',
                 'channels')
        total = dict(trials=0, max_prompt=0, **{name: 0 for name in names})
        for path, scale in [(p, s) for p in paths for s in Q.SCALES]:
            with self.subTest(trial=str(path.parent.parent.relative_to(ATIF)), scale=scale):
                counts, server, agent = replay(kit, path, scale)
                total['trials'] += 1
                for name in names:
                    total[name] += counts[name]
                total['max_prompt'] = max(total['max_prompt'], counts['max_prompt'])
                if counts['refused'] or counts['rebuilds'] or counts['asides']:
                    print('%-55s x%.2f %s' % (path.parent.parent.relative_to(ATIF), scale, counts))
                for messages, status, body in agent.requests:
                    if status != 200:
                        self.assertEqual(status, 400)
                        self.assertTrue(T.harbor_context_error(body), body)
                self.assertLessEqual(counts['max_prompt'], Q.CONTEXT)
        print('Gemma ATIF replay totals:', total)


if __name__ == '__main__':
    unittest.main()

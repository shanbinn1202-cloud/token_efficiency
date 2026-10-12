"""XML protocol token stream against the upstream dram-backed-kv server algorithm; CPU only."""
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
from remote_ra_live_v2 import mswea_protocol as xml
from remote_ra_http.config import validate_wire
from remote_ra_http.mswea_route import finish_response, prepare_request

SPECIAL = {'<|im_start|>': 1, '<|im_end|>': 2, '<|endoftext|>': 3}
EOS, PAD = 2, 3


class Tokenizer:
    """Parses added special tokens inside text, like the HF fast tokenizer."""
    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for token, value in SPECIAL.items():
                if text.startswith(token, i):
                    ids.append(value); i += len(token); break
            else:
                ids.append(ord(text[i]) + 100); i += 1
        return ids

    def decode(self, ids, skip_special_tokens=True):
        names = {v: k for k, v in SPECIAL.items()}
        return ''.join('' if t in names and skip_special_tokens else names.get(t) or chr(t - 100)
                       for t in ids)


TOK = Tokenizer()


def render_initial(messages):
    # Stand-in for the pinned template with enable_thinking=False.
    return TOK.encode(''.join(xml.message_text(m['role'], m['content']) for m in messages))


def transcript(allow_rebuild=False):
    return xml.Transcript(encode=TOK.encode, render_initial=render_initial, decode=TOK.decode,
                          stop_ids=frozenset({EOS, PAD}), allow_rebuild=allow_rebuild)


class Upstream:
    """Verbatim token bookkeeping of the upstream server (a70701e):
    OnlineSelectorSession._render/accepts/query/_generate/_append_boundary and
    the session rebuild in main().current_session."""

    def __init__(self):
        self.new_session()
        self.sessions = 0

    def new_session(self):
        self.logical, self.serializations, self.turn_id = [], [], 0

    def render(self, messages, add_prompt):
        ids = render_initial(messages[:2])
        exact_assistants = list(self.serializations)
        for message in messages[2:]:
            role, content = str(message['role']), str(message.get('content', ''))
            assert role in {'assistant', 'user'}
            exact_index = next((index for index, (returned, _ids) in enumerate(exact_assistants)
                                if returned.strip() == content.strip()), None) if role == 'assistant' else None
            if exact_index is None:
                ids.extend(TOK.encode(f'<|im_start|>{role}\n{content}<|im_end|>\n')); continue
            _returned, exact_ids = exact_assistants.pop(exact_index)
            ids.extend(exact_ids)
        if add_prompt:
            assert messages[-1]['role'] == 'user'
            ids.extend(TOK.encode('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        return ids

    def accepts(self, messages):
        if not self.logical:
            return True
        complete = self.render(messages, False)
        return len(complete) > len(self.logical) and complete[:len(self.logical)] == self.logical

    def query(self, messages, generated):
        new_instance = len(messages) == 2 and self.turn_id > 0
        if new_instance or (self.turn_id > 0 and not self.accepts(messages)):
            self.new_session()
        if self.turn_id == 0:
            self.sessions += 1
        self.turn_id += 1
        complete = self.render(messages, False)
        with_prompt = self.render(messages, True)
        assert complete[:len(self.logical)] == self.logical
        # _generate appends every token; the marker check precedes the EOS check.
        self.logical = list(with_prompt) + list(generated)
        text = TOK.decode(generated)
        eos = bool(generated) and generated[-1] in (EOS, PAD) and xml.MARKER not in text
        self.logical += TOK.encode('\n' if eos else '<|im_end|>\n')
        # assistant_start is the end of the prompt-free render on every turn.
        self.serializations.append((text, tuple(self.logical[len(complete):])))
        return with_prompt, len(complete), text


def reply(text, eos=False):
    return TOK.encode(text) + ([EOS] if eos else [])


def conversation(turns):
    messages = [dict(role='system', content='You are a helpful assistant.'),
                dict(role='user', content='<pr_description>\nFix the index bug in target.py\n</pr_description>')]
    for action, observation in turns:
        messages += [dict(role='assistant', content=action), dict(role='user', content=observation)]
    return messages


class TranscriptTests(unittest.TestCase):
    def test_turns_match_upstream_render_and_boundaries(self):
        ours, upstream = transcript(), Upstream()
        actions = ['THOUGHT: look\n<mswea_bash_command>ls</mswea_bash_command>',
                   'Done.',   # no marker: ends on EOS -> boundary is just "\n"
                   'THOUGHT: edit <|im_start|> literal\n<mswea_bash_command>sed -i x</mswea_bash_command>']
        eos = [False, True, False]
        observations = ['<returncode>0</returncode>\n<output>\na.py\n</output>',
                        'Please continue.',
                        '<returncode>1</returncode>\n<output>\nboom\n</output>']
        history = []
        for index in range(3):
            messages = conversation(history)
            tokens, stable, span = ours.render(messages)
            generated = reply(actions[index], eos[index])
            previous = len(upstream.logical)
            expected, prompt_end, text = upstream.query(messages, generated)
            self.assertEqual(list(tokens), expected)
            # Our stable prefix ends before the observation; upstream's
            # prompt-free render ends after it.
            self.assertEqual(span[1] if span else stable, prompt_end)
            self.assertEqual(stable, previous if span else prompt_end)
            committed = ours.commit(tokens, generated)
            self.assertEqual(committed['text'], text)
            self.assertEqual(list(ours.stream), upstream.logical)
            self.assertEqual(committed['boundary'], tuple(TOK.encode('\n' if eos[index] else '<|im_end|>\n')))
            if index:
                observed = TOK.encode(xml.message_text('user', history[-1][1]))
                self.assertEqual(span, (stable, stable + len(observed)))
                self.assertEqual(tokens[span[0]:span[1]], tuple(observed))
            else:
                self.assertIsNone(span)
                self.assertEqual(ours.initial_length, stable)
            # LiteLLM/Mini-SWE may trim the stored reply; exact IDs are still reused.
            history.append((' ' + committed['text'] + '\n', observations[index]))
        self.assertEqual(upstream.sessions, 1)
        self.assertEqual(ours.previous_action(), text)

    def test_non_extension_fails_closed_or_rebuilds_like_upstream(self):
        good = 'a <mswea_bash_command>ls</mswea_bash_command>'
        ours = transcript()
        tokens, _s, _p = ours.render(conversation([]))
        ours.commit(tokens, reply(good))
        for messages in (conversation([('something else', 'obs')]),          # rewritten reply
                         conversation([]),                                    # resend
                         conversation([(good, 'obs')])[:-2] + [dict(role='user', content='Format error')]):
            with self.subTest(messages=len(messages)), self.assertRaisesRegex(Exception, 'SESSION_REBUILD_REQUIRED'):
                ours.render(messages)
        # With rebuilds allowed the result is upstream's cold session.
        ours, upstream = transcript(allow_rebuild=True), Upstream()
        first = conversation([])
        tokens, _s, _p = ours.render(first)
        upstream.query(first, reply(good))
        ours.commit(tokens, reply(good))
        retry = first + [dict(role='user', content='Format error: expected exactly 1 action')]
        tokens, stable, span = ours.render(retry)
        expected, prompt_end, _text = upstream.query(retry, reply(good))
        self.assertEqual((list(tokens), stable, span), (expected, prompt_end, None))
        self.assertEqual(ours.commit(tokens, reply(good))['rebuild'], True)
        self.assertEqual((ours.turn, ours.sessions, upstream.sessions), (1, 2, 2))
        self.assertEqual(ours.initial_length, len(render_initial(first)))

    def test_commit_requires_the_rendered_prompt(self):
        ours = transcript()
        tokens, _s, _p = ours.render(conversation([]))
        with self.assertRaisesRegex(Exception, 'differs'):
            ours.commit(tokens + (5,), [200])
        ours.abandon()
        with self.assertRaisesRegex(Exception, 'differs'):
            ours.commit(tokens, [200])
        with self.assertRaisesRegex(Exception, 'end with a user'):
            ours.render(conversation([('a', 'b')])[:-1])

    def test_objective_returncode_and_sampling_contract(self):
        self.assertEqual(xml.root_objective(conversation([])), 'Fix the index bug in target.py')
        self.assertEqual(xml.return_code('x <returncode>-9</returncode> y'), -9)
        self.assertIsNone(xml.return_code('no code'))
        good = NS(n=1, temperature=0.0, max_tokens=1536, min_tokens=0, stop=[], ignore_eos=False,
                  stop_token_ids=[PAD], eos_token_id=EOS, repetition_penalty=1.0, presence_penalty=0.0,
                  frequency_penalty=0.0, logit_bias=None, allowed_token_ids=None, bad_words=[],
                  structured_outputs=None, logits_processors=None, skip_special_tokens=True)
        self.assertTrue(xml.check_sampling(good, frozenset({EOS, PAD})))
        for field, value in (('temperature', 0.6), ('max_tokens', 16), ('stop', ['</x>']),
                             ('stop_token_ids', [PAD, 9]), ('repetition_penalty', 1.05)):
            # The rejection names the differing field (smoke job 3265224 only saw the contract name).
            with self.subTest(field=field), self.assertRaisesRegex(Exception, 'greedy generation.*' + field):
                xml.check_sampling(NS(**{**vars(good), field: value}), frozenset({EOS, PAD}))
        # vLLM 0.24's Qwen3 parser engine sets skip_special_tokens=False while rendering; it
        # only shapes vLLM's own text, which the route replaces with the upstream decode.
        self.assertTrue(xml.check_sampling(NS(**{**vars(good), 'skip_special_tokens': False}),
                                           frozenset({EOS, PAD})))

    def test_engine_marker_stop(self):
        status = types.ModuleType('vllm.v1.request')
        status.RequestStatus = NS(FINISHED_STOPPED='FINISHED_STOPPED')
        modules = {'vllm': types.ModuleType('vllm'), 'vllm.v1': types.ModuleType('vllm.v1'),
                   'vllm.v1.request': status}
        saved = {k: sys.modules.get(k) for k in modules}
        sys.modules.update(modules)
        try:
            stop = xml.MarkerStop(TOK.decode)
            request = NS(output_token_ids=reply('x <mswea_bash_command>ls</mswea_bash_'), status=None,
                         stop_reason=None)
            self.assertFalse(stop(request))
            request.output_token_ids += TOK.encode('command>')
            self.assertTrue(stop(request))
            self.assertEqual((request.status, request.stop_reason), ('FINISHED_STOPPED', xml.MARKER))
        finally:
            for key, value in saved.items():
                if value is None: sys.modules.pop(key, None)
                else: sys.modules[key] = value


class RouteTests(unittest.TestCase):
    def test_wire_contract(self):
        task = dict(instance_id='t', problem_statement='Fix the index bug in target.py')
        body = dict(messages=conversation([('a', '<returncode>0</returncode>')]))
        self.assertTrue(validate_wire(body, task, 1, 'mswea_xml_v1'))
        with self.assertRaisesRegex(Exception, 'tools'):
            validate_wire(dict(body, tools=[{'type': 'function'}]), task, 1, 'mswea_xml_v1')
        with self.assertRaisesRegex(Exception, 'first request'):
            validate_wire(body, task, 0, 'mswea_xml_v1')
        bad = dict(messages=conversation([]))
        bad['messages'][1]['content'] = 'another issue'
        with self.assertRaisesRegex(Exception, 'frozen issue'):
            validate_wire(bad, task, 0, 'mswea_xml_v1')
        extra = dict(messages=[dict(m, extra={'k': 1}) for m in conversation([])])
        self.assertTrue(validate_wire(extra, task, 0, 'mswea_xml_v1'))   # transport metadata ignored

    def test_request_contract_and_reply_rebuilt_from_token_ids(self):
        request = NS(tools=None, stream=False, n=1, max_tokens=64, max_completion_tokens=None,
                     temperature=0.7, top_p=0.9, min_tokens=0, repetition_penalty=1.1, presence_penalty=1.5,
                     frequency_penalty=0.0, stop=['x'], stop_token_ids=[], include_stop_str_in_output=True,
                     ignore_eos=False, skip_special_tokens=True, logprobs=False, top_logprobs=None,
                     logit_bias=None, bad_words=None, allowed_token_ids=None, response_format=None,
                     tool_choice=None, return_token_ids=None)
        prepare_request(request, frozenset({EOS, PAD}))
        self.assertEqual((request.max_tokens, request.temperature, request.repetition_penalty,
                          request.presence_penalty, request.stop, sorted(request.stop_token_ids),
                          request.return_token_ids), (1536, 0.0, 1.0, 0.0, [], [EOS, PAD], True))
        ours = transcript()
        tokens, _s, _p = ours.render(conversation([]))
        # vLLM truncates at a stop string; upstream returns the whole final token.
        generated = TOK.encode('run <mswea_bash_command>ls</mswea_bash_command>\n')
        message = NS(content='truncated', tool_calls=[1], reasoning='r')
        result = NS(choices=[NS(message=message, token_ids=list(generated), finish_reason='length',
                                stop_reason='x')], prompt_token_ids=[1])
        finish_response(result, NS(capture=NS(tokens=tokens)), ours)
        self.assertEqual(message.content, 'run <mswea_bash_command>ls</mswea_bash_command>\n')
        self.assertEqual((message.tool_calls, message.reasoning, result.choices[0].finish_reason,
                          result.choices[0].token_ids, result.prompt_token_ids), ([], None, 'stop', None, None))
        self.assertEqual(ours.turn, 1)


def requests(messages):
    """The requests Mini-SWE-Agent sent: every prefix ending at a user message
    that got a model reply.  A format-error retry drops the rejected reply, so
    its text is not in the trajectory; a placeholder stands in for it."""
    plain = [dict(role=m['role'], content=m['content']) for m in messages if m['role'] != 'exit']
    for index, message in enumerate(plain):
        if index < 1 or message['role'] != 'user' or index + 1 >= len(plain):
            continue
        following = plain[index + 1]
        text = (following['content'] if following['role'] == 'assistant'
                else 'THOUGHT: rejected reply without an action')
        yield plain[:index + 1], text


class CohortTrajectoryTests(unittest.TestCase):
    """Replay the upstream cohort-2 trajectories (if exported) through both algorithms."""
    DIR = ROOT.parent.parent/'development'/'selection-contract-v7'/'upstream-cohort2'

    def test_real_trajectories_render_like_upstream(self):
        files = sorted(self.DIR.glob('*.traj.json')) if self.DIR.is_dir() else []
        if not files:
            self.skipTest('upstream cohort-2 trajectories not exported next to this checkout')
        for path in files:
            with self.subTest(trajectory=path.name):
                messages = json.loads(path.read_text(encoding='utf-8'))['messages']
                metrics = [json.loads(line) for line in
                           path.with_name(path.name.replace('.traj.json', '.selector_metrics.jsonl'))
                           .read_text(encoding='utf-8').splitlines()]
                ours, upstream = transcript(allow_rebuild=True), Upstream()
                turn_ids, rebuilds = [], []
                for prefix, text in requests(messages):
                    tokens, stable, span = ours.render(prefix)
                    expected, prompt_end, _t = upstream.query(prefix, TOK.encode(text))
                    self.assertEqual((list(tokens), span[1] if span else stable), (expected, prompt_end))
                    committed = ours.commit(tokens, TOK.encode(text))
                    self.assertEqual(list(ours.stream), upstream.logical)
                    turn_ids.append(ours.turn); rebuilds.append(committed['rebuild'])
                # Same turn numbering and session rebuilds as the recorded run.
                self.assertEqual(turn_ids, [row['turn_id'] for row in metrics])
                self.assertEqual(rebuilds, [row['resume_reason'] != 'fresh_instance'
                                            and row['turn_id'] == 1 for row in metrics])


if __name__ == '__main__':
    unittest.main()

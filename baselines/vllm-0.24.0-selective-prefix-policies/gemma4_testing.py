"""Shared fixtures of the Gemma 4 stage-1 CPU tests (not a test module).

* ``config()`` builds a Gemma 4 31B IT ``config.json`` value with the pinned geometry.
* ``metadata_dir()`` is the directory named by ``REMOTE_RA_GEMMA4_METADATA`` that holds
  the pinned metadata files of the checkpoint (config.json, generation_config.json,
  tokenizer.json, tokenizer_config.json, chat_template.jinja), or None; tests that need
  the real files skip without it. The files are checked against their pins first.
* ``Stage0`` re-creates the synthetic XML session of the stage-0 stock-vLLM reference
  (perf-gemma0 ``xmlsession``, job 3364469, server t1-triton): system and task messages,
  the deterministic ``cat -n`` observations, and the recorded completion ids with the
  prompt hashes of every turn (``testdata/gemma4-stage0-append.json``).
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source-overlay'))

from remote_ra_live_v2 import model_profile as MP  # noqa: E402

ENV = 'REMOTE_RA_GEMMA4_METADATA'
STAGE0 = HERE / 'testdata' / 'gemma4-stage0-append.json'


def config():
    """A config.json value with exactly the fields gemma4_geometry checks (pinned values)."""
    types = ['full_attention' if i % 6 == 5 else 'sliding_attention' for i in range(60)]
    text = dict(model_type='gemma4_text', num_hidden_layers=60, layer_types=types, hidden_size=5376,
                sliding_window=1024, num_key_value_heads=16, head_dim=256, num_global_key_value_heads=4,
                global_head_dim=512, num_kv_shared_layers=0, attention_k_eq_v=True, enable_moe_block=False,
                hidden_size_per_layer_input=0, dtype='bfloat16', max_position_embeddings=262144)
    return dict(model_type='gemma4', architectures=['Gemma4ForConditionalGeneration'], text_config=text)


def metadata_dir():
    """The pinned Gemma metadata directory, or None when REMOTE_RA_GEMMA4_METADATA is unset."""
    value = os.environ.get(ENV)
    if not value:
        return None
    root = Path(value)
    for name, pin in MP.GEMMA4_MODEL_PINS.items():
        raw = (root / name).read_bytes()
        if sha256(raw).hexdigest() != pin:
            raise AssertionError('%s: %s differs from its pin' % (ENV, name))
    return root


def tokenizer():
    """Transformers' tokenizer of the pinned metadata directory, or None (no files or no
    transformers/tokenizers in this interpreter)."""
    root = metadata_dir()
    if root is None:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    return AutoTokenizer.from_pretrained(str(root), local_files_only=True)


# ---------------------------------------------------------------- the stage-0 XML session
# perf-gemma0 xmlsession.py (smoke-dp1 session.py of job 3308015, task variant 0), byte for byte.
FN, MODULE = 'window_sum', 'stats/rolling.py'
STATEMENT = ('`{fn}` in `{module}` drops the final window: `{fn}({values}, {window})` returns `{wrong}` instead of '
             '`{right}`. Fix the off-by-one error in the loop bound so the final element is preserved. '
             'Windows longer than the input must still raise `ValueError`.')
SYSTEM = ('You are a helpful assistant that can interact with a computer shell to solve programming tasks. '
          'Reply with a short THOUGHT and exactly one <mswea_bash_command>...</mswea_bash_command> block; '
          'keep replies under 60 words. Write exactly one THOUGHT line of at most 25 words, then the block '
          'with one short shell command, and nothing after the closing tag.')
INSTRUCTIONS = ('<instructions>\n'
                'You are in /testbed. This is a scripted smoke session: every command returns a synthetic listing.\n'
                'Each reply: one line starting with THOUGHT: (at most 25 words), then exactly one '
                '<mswea_bash_command>...</mswea_bash_command> block with one short shell command. '
                'Nothing after the closing tag. Under 60 words in total.\n'
                '</instructions>')
FUNCTIONS = ('window_sum', 'window_mean', 'rolling_max', 'prefix_total', 'trailing_count')
LISTING_SEED = 0
TEMPLATES = (
    'def {fn}_{a}(values, window={b}):',
    '    """Return the rolling {fn} of {b} items (see issue {c})."""',
    '    total_{a} = sum(values[{a}:{a} + window])',
    '    if len(values) < window:',
    '        raise ValueError("window {b} exceeds %d values" % len(values))',
    '    for index in range(len(values) - window + {d}):',
    '        yield values[index:index + window]  # bound check {c}',
    '    return [{fn}_{a}(chunk) for chunk in _chunks(values, {b})]',
    'class Rolling{cls}{a}(BaseWindow):',
    '    cache_key = "{fn}:{a}:{b}"',
    '',
    'import itertools  # stats/rolling.py line {c}',
    'assert {fn}_{a}([1, 2, 3], window={d}) == [{a}, {b}]',
    '    return total_{a} / max(1, window - {d})',
)


def problem_statement():
    values, window = list(range(1, 6)), 2
    right = [sum(values[i:i + window]) for i in range(len(values) - window + 1)]
    return STATEMENT.format(fn=FN, module=MODULE, values=values, window=window, wrong=right[:-1], right=right)


def task_message():
    return '<pr_description>\n' + problem_statement() + '\n</pr_description>\n\n' + INSTRUCTIONS


def initial_messages():
    return [dict(role='system', content=SYSTEM), dict(role='user', content=task_message())]


def listing(turn, lines):
    state = (2654435761 * (turn + 1) + LISTING_SEED) & 0xFFFFFFFF
    out = ['# synthetic listing %d of %s (%d lines)' % (turn, MODULE, lines)]
    for number in range(1, lines + 1):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        fn = FUNCTIONS[(state >> 8) % len(FUNCTIONS)]
        code = TEMPLATES[(state >> 12) % len(TEMPLATES)].format(
            fn=fn, cls=''.join(p.title() for p in fn.split('_')), a=(state >> 4) % 97,
            b=(state >> 9) % 31 + 2, c=(state >> 3) % 9973, d=(state >> 15) % 3)
        out.append('%6d\t%s' % (number + 1000 * turn, code))
    return '\n'.join(out)


def observation(turn, lines):
    return '<returncode>%d</returncode>\n<output>\n%s\n</output>' % (1 if turn % 3 == 2 else 0, listing(turn, lines))


def ids_sha(ids):
    """perf-gemma0 emulate.prompt_sha256: SHA-256 of the comma-joined decimal ids."""
    return sha256(','.join(str(i) for i in ids).encode()).hexdigest()


class Stage0:
    """The stage-0 append reference: per turn the prompt length and hash, the completion
    ids, the boundary ids and the next user message's length and hash."""

    def __init__(self, path=STAGE0):
        value = json.loads(Path(path).read_bytes())
        self.turns = value['turns']
        self.observation_lines = value['observation_lines']
        self.stop_ids = frozenset(value['stop_ids'])
        self.header_ids = value['header_ids']
        self.initial_tokens = value['initial_tokens']
        self.source = value['source']

    def observation(self, turn):
        return observation(turn, self.observation_lines)

    def messages(self, turns, decode):
        """The client's messages after ``turns`` completed turns (replies decoded from their ids)."""
        messages = initial_messages()
        for k in range(turns):
            messages += [dict(role='assistant', content=decode(self.turns[k]['completion_ids'])),
                         dict(role='user', content=self.observation(k + 1))]
        return messages

"""goal_query: the Goal rewriter on vLLM (serve.py --goal-rewriter-engine vllm). CPU only.

remote_ra_live_v2.goal_rewriter_vllm (parent transport) and goal_rewriter_vllm_child (child);
docs/rewriter-vllm/DESIGN.md. No GPU and no vLLM: the child runs as a real subprocess, with the
command and environment the parent builds (-I -S -B -u, only the given site-packages on
sys.path), against a fake ``vllm`` package and a fake ``transformers`` written to a temporary
site directory. The fake AsyncLLM decodes with test_goal_rewriter_batch's fake model and
enforcer (so its tokens are those of the Hugging Face child's fakes), waits a per-request delay
(so answers complete out of order), and scripts failures by words in the prompt (an engine
error, a dead engine, a hang, a crash, a slow call). The fake tokenizer is
test_goal_rewriter_batch's (one token per character).

Child:
- the prompt ids and the JSON schema are the upstream generate's (goal_rewriter_child.prepare),
  errors before the model call too; decode and parse as upstream (a reply cut at max_new_tokens
  answers the upstream GenerationFailed with its token counts); xgrammar's schema without
  uniqueItems;
- run as a subprocess: stdout carries only protocol lines (the fake vLLM's logger and a C-level
  write land on stderr), the readiness record is the upstream one, every answer equals the
  upstream generate's for that request alone (latency aside), bad lines and envelopes answer as
  upstream, an abort of an unknown request is silent, the end of stdin shuts the engine down;
- isolation: the child's sys.path has the given site and no overlay, its vllm is the site's, it
  loaded no overlay package (only the two scripts by file path) and no torch, its environment has
  no REMOTE_RA_*, VLLM_POLICY_* or SELECTIVE_* variable and VLLM_BATCH_INVARIANT only with batch
  invariance; the engine gets exactly the pinned engine arguments, each request temperature 0,
  max_tokens, the stop ids, the schema, no detokenization and the final output only;
- lm-format-enforcer on vLLM 0.24 (job rw2): the fake site has vLLM 0.24's module layout and
  lm-format-enforcer 0.11.3's integrations, and the fake engine builds its backend in a separate
  interpreter set up as a spawned EngineCore (multiprocessing.spawn.prepare). The backend gets the
  Hugging Face child's tokenizer data (no padding id) in a bitmask of the model's ids; the startup
  check (one guided request before the readiness record) refuses an unguided reply and a stray id;
  mutations (no module, the module in the child only, no Transformers alias, the data over every
  model id, no startup check: every call fails as in rw2) are caught.

Parent (the real child):
- sixteen sessions' Goals at once (three turns each): the Goals and their fallbacks are the
  Hugging Face child's (test_goal_rewriter_batch's in-process upstream child), the engine got the
  same prompts, several calls were in flight and they completed out of order, each exchange has
  the upstream timing keys and no batch keys;
- an engine error fails its call only; a timeout fails its call, aborts that request in the child
  and keeps the child (the other calls go on); a child exit fails every call in flight and the
  next call starts a new child inside it; an engine that died ends its child likewise; shutdown
  fails the call in flight and ends the child; a child that does not start fails the startup
  (exit, invalid readiness, startup timeout); the child runs in its own session, and (POSIX)
  ``stop`` ends its whole process group;
- the Goal control hash, the startup check, the tap, and the authority (it starts the vLLM
  rewriter only with the pin).

End to end: test_gemma_goal_rewriter's sixteen goal_query sessions on the Gemma 4 window layout's
engine and worker fakes with goal_rewriter_engine pinned in the startup config: the authority
(private_endpoint) starts the real child, and every turn decides, selects, attends and replies as with
the Hugging Face child's fakes, the exchanges equal and without batch keys.

Launcher, startup config and evidence: the flags pin goal_rewriter_engine and nothing else
(Qwen and Gemma 4 profiles; the vLLM command unchanged), their refusals, config.load's check of
the pin, the dry run, the child command; ``hf`` (the default) adds nothing and the Hugging Face
child's command is unchanged; worker-attached.json and ready.json gain the record and the
child's start time only with vllm.
"""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / 'source-overlay'
sys.path.insert(0, str(OVERLAY))
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    torch = None

import test_goal_rewriter_batch as RB     # noqa: E402
from remote_ra_live_v2 import goal_async as GA, goal_rewriter_child as C, query_selection     # noqa: E402
from remote_ra_live_v2 import goal_rewriter_vllm as V, goal_rewriter_vllm_child as VC, selection_contract  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error     # noqa: E402

query_selection.upstream()      # the packaged dram_kv namespace
from dram_kv import schema      # noqa: E402
from dram_kv.goals import maintained, rewriter as R     # noqa: E402

U = C.upstream()
READY = '{"protocol":"dram_kv_rewriter_ndjson","ready":true,"version":1}\n'
WAIT = 60.0                     # a subprocess's answer under load (the suite runs one OpenMP thread)

# ---- the fake site: vllm and transformers ---------------------------------------------------------

FAKE = {
    'vllm/__init__.py': r'''"""A fake vLLM 0.24 for the Goal rewriter's vLLM child (test_goal_rewriter_vllm)."""
import json
import os
import sys
import threading

_LOCK = threading.Lock()


def log(**record):
    path = os.environ.get('FAKE_VLLM_LOG')
    if path:
        record['pid'] = os.getpid()
        with _LOCK, open(path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')


WATCHED = ('remote_ra', 'dram_kv', 'policy_runtime', 'vllm', 'transformers', 'torch')
if not os.environ.get('FAKE_ENGINE_CORE'):          # the child, not the fake EngineCore
    log(event='import', file=__file__, path=list(sys.path), env=sorted(os.environ),
        modules={name: getattr(module, '__file__', None) for name, module in sorted(sys.modules.items())
                 if name.split('.')[0].startswith(WATCHED)},
        batch_invariant=os.environ.get('VLLM_BATCH_INVARIANT'))
    print('INFO fake vllm: the logger writes to stdout')
    os.write(1, b'fake vllm: a C-level write to fd 1\n')

from vllm.sampling_params import SamplingParams  # noqa: E402
''',
    # vLLM 0.24 has no vllm/transformers_utils/tokenizer.py: its tokenizers moved to vllm/tokenizers/.
    'vllm/transformers_utils/__init__.py': '',
    'vllm/tokenizers/__init__.py': '',
    'vllm/tokenizers/mistral.py': r'''class MistralTokenizer:
    """vLLM 0.24's Mistral tokenizer (formerly vllm.transformers_utils.tokenizer.MistralTokenizer)."""
''',
    'vllm/v1/structured_output/__init__.py': r'''"""A fake vLLM 0.24 structured-output manager: the backend is built for the first structured request
(StructuredOutputManager.grammar_init) in the EngineCore process (fake_engine_core: a spawned process)."""
from fake_engine_core import build  # noqa: F401
''',
    'fake_engine_core.py': r'''"""The fake vLLM's EngineCore for the structured-output backend: a new interpreter that
multiprocessing.spawn.prepare sets up as a spawned EngineCore (the child's sys.path, its script re-run as
__mp_main__); its answer comes back as one JSON line."""
import builtins
import json
import multiprocessing.spawn
import os
import subprocess
import sys
import threading

MODEL_VOCAB = 160       # the checkpoint's padded vocabulary (config vocab_size), past the tokenizer's 137 ids
ENGINE_CORE = 'FAKE_ENGINE_CORE'
PROGRAM = ("import json,sys;data=json.loads(sys.argv[1]);sys.path[:]=data['sys_path'];"
           "import multiprocessing.spawn as spawn;spawn.prepare(data['prepare']);"
           "import fake_engine_core;fake_engine_core.answer(data['model'],data['backend'])")


def summary(model, backend):
    """In the EngineCore: the backend's tokenizer data, and what the test checks of it."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    if backend != 'lm-format-enforcer':
        return dict(backend=backend)
    from vllm.v1.structured_output.backend_lm_format_enforcer import tokenizer_data
    data = tokenizer_data(tokenizer, MODEL_VOCAB)
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    hf = build_token_enforcer_tokenizer_data(tokenizer)                         # the Hugging Face child's
    own = build_token_enforcer_tokenizer_data(tokenizer, True, MODEL_VOCAB)     # lm-format-enforcer's vLLM builder's
    size = len(tokenizer)
    beyond = lambda tokens: [i for i, text, _ in tokens if i >= size and text == '']
    return dict(backend=backend, tokenizer_len=size, width=data.vocab_size, use_bitmask=data.use_bitmask,
                eos=data.eos_token_id, same_as_hf=data.regular_tokens == hf.regular_tokens,
                max_id=max(i for i, _, _ in data.regular_tokens), root_beyond=beyond(data.regular_tokens),
                own_root_beyond=beyond(own.regular_tokens))


def answer(model, backend):
    try:
        line = dict(ok=summary(model, backend))
    except Exception as error:
        line = dict(error=[type(error).__name__, str(error)])
    sys.__stdout__.write(json.dumps(line) + '\n')


_LOCK = threading.Lock()        # one build at a time (vLLM builds it on the scheduler's thread)


def build(model, backend):
    """The backend of the first structured request, built in the EngineCore; its error re-raised here."""
    prepare = multiprocessing.spawn.get_preparation_data('fake-engine-core')
    assert prepare.get('init_main_from_path'), 'spawn would not re-run the child script'
    data = dict(sys_path=sys.path, model=model, backend=backend,
                prepare={key: prepare[key] for key in ('sys_path', 'init_main_from_path')})
    with _LOCK:     # stdin not inherited: Windows blocks duplicating it while the child's reader waits on it
        run = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', PROGRAM, json.dumps(data)], capture_output=True,
                             stdin=subprocess.DEVNULL, text=True, encoding='utf-8', timeout=60,
                             env=dict(os.environ, **{ENGINE_CORE: '1'}))
    lines = run.stdout.splitlines()
    if run.returncode or not lines:
        raise RuntimeError('the fake EngineCore failed: %s' % run.stderr[-2000:])
    value = json.loads(lines[-1])
    if 'error' in value:
        kind, message = value['error']
        error = getattr(builtins, kind, None)
        raise (error if isinstance(error, type) and issubclass(error, Exception) else RuntimeError)(message)
    return value['ok']
''',
    'vllm/v1/structured_output/backend_lm_format_enforcer.py': r'''"""A fake vLLM 0.24 lm-format-enforcer backend: the real one's lazy load of
lmformatenforcer.integrations.vllm (LazyLoader) and its tokenizer-data call."""
import importlib


def tokenizer_data(tokenizer, vocab_size):
    lmfe_vllm = importlib.import_module('lmformatenforcer.integrations.vllm')
    return lmfe_vllm.build_vllm_token_enforcer_tokenizer_data(tokenizer, use_bitmask=True, vocab_size=vocab_size)
''',
    'torch/__init__.py': '"""A fake torch: lm-format-enforcer\'s vLLM integration imports it (the EngineCore has it)."""\n',
    'lmformatenforcer/__init__.py': r'''"""A fake lm-format-enforcer 0.11.3: the tokenizer data."""


class TokenEnforcerTokenizerData:
    def __init__(self, regular_tokens, decoder, eos_token_id, use_bitmask=False, vocab_size=None):
        self.regular_tokens, self.decoder, self.eos_token_id = regular_tokens, decoder, eos_token_id
        self.use_bitmask, self.vocab_size = use_bitmask, vocab_size
''',
    'lmformatenforcer/integrations/__init__.py': '',
    'lmformatenforcer/integrations/vllm.py': r'''"""A fake lm-format-enforcer 0.11.3 vLLM integration: its import block and its builder, as released."""
try:
    import torch
    import vllm
    from vllm.transformers_utils.tokenizer import MistralTokenizer
    from transformers import PreTrainedTokenizerBase
except ImportError:
    raise ImportError('vllm is not installed. Please install it with "pip install vllm"')
from lmformatenforcer import TokenEnforcerTokenizerData
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data


def build_vllm_token_enforcer_tokenizer_data(tokenizer, use_bitmask=False, vocab_size=None):
    if vocab_size is None and hasattr(tokenizer, 'llm_engine'):
        vocab_size = tokenizer.llm_engine.get_model_config().get_vocab_size()
    if hasattr(tokenizer, 'get_tokenizer'):
        tokenizer = tokenizer.get_tokenizer()
    if isinstance(tokenizer, MistralTokenizer):
        return build_token_enforcer_tokenizer_data(tokenizer, use_bitmask, vocab_size)
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer
    return build_token_enforcer_tokenizer_data(tokenizer, use_bitmask, vocab_size)
''',
    'lmformatenforcer/integrations/transformers.py': r'''"""A fake lm-format-enforcer 0.11.3 Transformers integration: its import block and its builder, as
released (Transformers 5 has no PreTrainedTokenizerBase in transformers.tokenization_utils)."""
import functools
try:
    from transformers.tokenization_utils import PreTrainedTokenizerBase
except ImportError:
    raise ImportError('transformers is not installed. Please install it with "pip install transformers[torch]"')
from lmformatenforcer import TokenEnforcerTokenizerData


def _build_regular_tokens_list(tokenizer, vocab_size):
    token_0 = tokenizer.encode("0")[-1]
    regular_tokens = []
    for token_idx in range(vocab_size):
        if token_idx in tokenizer.all_special_ids:
            continue
        decoded_after_0 = tokenizer.decode([token_0, token_idx])[1:]
        decoded_regular = tokenizer.decode([token_idx])
        is_word_start_token = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_idx, decoded_after_0, is_word_start_token))
    return regular_tokens


def _decode_function(tokenizer, tokens):
    decoded = tokenizer.decode(tokens)
    return decoded.rstrip('�')


def build_token_enforcer_tokenizer_data(tokenizer, use_bitmask=False, vocab_size=None):
    vocab_size = vocab_size or len(tokenizer)
    regular_tokens = _build_regular_tokens_list(tokenizer, vocab_size)
    decode_fn = functools.partial(_decode_function, tokenizer)
    return TokenEnforcerTokenizerData(regular_tokens, decode_fn, tokenizer.eos_token_id, use_bitmask, vocab_size)
''',
    'vllm/sampling_params.py': r'''import enum


class RequestOutputKind(enum.Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


class StructuredOutputsParams:
    def __init__(self, json=None):
        self.json = json
        self.kwargs = dict(json=json)


class SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
''',
    'vllm/inputs.py': r'''def TokensPrompt(**fields):
    return dict(fields)
''',
    'vllm/config.py': r'''class CompilationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StructuredOutputsConfig:
    def __init__(self, backend='auto'):
        self.kwargs = dict(backend=backend)
''',
    'vllm/engine/__init__.py': '',
    'vllm/engine/arg_utils.py': r'''class AsyncEngineArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
''',
    'vllm/v1/__init__.py': '',
    'vllm/v1/engine/__init__.py': '',
    'vllm/v1/engine/async_llm.py': r'''"""A scripted AsyncLLM: test_goal_rewriter_batch's fake model and enforcer for each request, a
per-request delay; words in the prompt script failures."""
import asyncio
from hashlib import sha256
import json
import os
import re
import sys

from vllm import log
from vllm.v1.structured_output import build as build_backend

EOS, OFFSET = 1, 10
VOCAB = tuple(range(32 + OFFSET, 127 + OFFSET))
STRAY = 150             # a padding id past the tokenizer (FAKE_VLLM_STRAY_ID)


def instance(schema):
    """The only value of a schema of required properties, single enums and fixed-size arrays: what
    guided decoding must produce (the child's startup check)."""
    if schema['type'] == 'object':
        return {key: instance(schema['properties'][key]) for key in schema['required']}
    if schema['type'] == 'array':
        return [instance(schema['items'])] * schema['minItems']
    (value,) = schema['enum']
    return value


def text_of(ids):
    return ''.join(chr(i - OFFSET) for i in ids if i >= OFFSET)


def target(prompt):
    """test_goal_rewriter_batch.Model.target."""
    request = json.loads(prompt[prompt.index('<user>') + 6:prompt.index('<assistant>')])['request']
    tag = sha256(prompt.encode()).hexdigest()[:6]
    observation = request['latest_observation']
    if 'TRUNCATE' in observation:
        return '{"description":"' + 'a' * 400
    body = dict(description='Retrieve %s %s XZ' % (request['trigger'], tag), phase_hint='hypothesis',
                files=sorted(set(re.findall(r'[a-z]+/[a-z_]+\.py', observation)))[:3])
    if 'DUPLICATE' in observation:
        body['symbols'] = ['parse', 'parse']
    if request['allow_constraint_updates']:
        body['constraints'] = ['keep Z']
    return json.dumps(body, separators=(',', ':'))


def pick(goal, generated, allowed):
    """test_goal_rewriter_batch.Model.pick."""
    p = 0
    for token in generated:
        if token < OFFSET:
            break
        ch = chr(token - OFFSET)
        while p < len(goal) and goal[p] != ch:
            p += 1
        p += 1
    for ch in goal[p:]:
        if ord(ch) + OFFSET in allowed:
            return ord(ch) + OFFSET
    return EOS if EOS in allowed else min(allowed)


def decode(goal, forbidden, max_tokens, stops):
    """Greedy decoding under test_goal_rewriter_batch's enforcer: no forbidden character, only EOS
    once the object closed. The token ids (the stop token included) and the finish reason."""
    allowed_open = {t for t in VOCAB if chr(t - OFFSET) not in forbidden}
    generated = []
    for _ in range(max_tokens):
        depth, opened = 0, False
        for ch in text_of(generated):
            depth += ch == '{'
            depth -= ch == '}'
            opened = opened or ch == '{'
        token = pick(goal, generated, {EOS} if opened and depth == 0 else allowed_open)
        generated.append(token)
        if token in stops:
            return generated, 'stop'
    return generated, 'length'


def plain(value):
    if hasattr(value, 'kwargs'):
        return {key: plain(item) for key, item in value.kwargs.items()}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, 'name') and hasattr(value, 'value') and not isinstance(value, (str, int, float)):
        return value.name
    return value


class Completion:
    def __init__(self, token_ids, finish_reason):
        self.token_ids, self.finish_reason = token_ids, finish_reason


class Output:
    def __init__(self, request_id, token_ids, finish_reason):
        self.request_id, self.finished = request_id, True
        self.outputs = [Completion(token_ids, finish_reason)]


class AsyncLLM:
    def __init__(self, args):
        self.args, self.errored, self.active, self.peak = args, False, 0, 0
        self.backend = None     # built for the first structured request; a failed build is retried (vLLM)
        log(event='engine', args=plain(args), batch_invariant=os.environ.get('VLLM_BATCH_INVARIANT'))

    @classmethod
    def from_engine_args(cls, args):
        return cls(args)

    async def guided(self, schema):
        """The structured-output backend: built once (its build error fails the request, as vLLM's
        EngineCore does); xgrammar refuses uniqueItems."""
        name = self.args.kwargs['structured_outputs_config'].kwargs['backend']
        if self.backend is None:
            self.backend = await asyncio.to_thread(build_backend, self.args.kwargs['model'], name)
            log(event='backend', **self.backend)
        if name == 'xgrammar' and 'uniqueItems' in json.dumps(schema):
            raise ValueError('xgrammar refuses the schema keyword uniqueItems')

    async def generate(self, prompt, sampling_params, request_id):
        ids = list(prompt['prompt_token_ids'])
        words = text_of(ids)
        self.active += 1
        self.peak = max(self.peak, self.active)
        log(event='request', request_id=request_id, prompt_ids=ids, sampling=plain(sampling_params))
        try:
            schema = sampling_params.structured_outputs.json
            try:
                await self.guided(schema)
            except ImportError as error:        # vLLM 0.24 (job rw2): the EngineCore logs it, ends the request
                sys.stderr.write('ERROR fake EngineCore: Unexpected error pre-processing request %s: %s: %s\n'
                                 % (request_id, type(error).__name__, error))
                log(event='finish', request_id=request_id, peak=self.peak, finish_reason='error')
                yield Output(request_id, [], 'error')       # no token: the child's parse fails
                return
            if 'required' in schema:            # a schema with one value (the startup check)
                text = 'Hello.' if os.environ.get('FAKE_VLLM_UNGUIDED') else json.dumps(instance(schema))
                tokens = [ord(c) + OFFSET for c in text] + ([STRAY] if os.environ.get('FAKE_VLLM_STRAY_ID') else [])
                log(event='finish', request_id=request_id, peak=self.peak, finish_reason='stop')
                yield Output(request_id, tokens + [EOS], 'stop')
                return
            if 'ENGINE_ERROR' in words:
                raise ValueError('the fake engine refused this request')
            if 'ENGINE_DEAD' in words:
                self.errored = True
                raise RuntimeError('EngineDeadError: the fake engine died')
            if 'CRASH' in words:
                log(event='crash', request_id=request_id)
                os._exit(3)
            if 'HANG' in words:
                await asyncio.sleep(600)
            delay = 2.0 if 'SLOW' in words else int(sha256(words.encode()).hexdigest()[:4], 16) % 8 * 0.025
            await asyncio.sleep(delay)
            schema = sampling_params.structured_outputs.json
            forbidden = {'X'} | (set() if 'constraints' in schema['properties'] else {'Z'})
            tokens, reason = decode(target(words), forbidden, sampling_params.max_tokens,
                                    set(sampling_params.stop_token_ids))
            log(event='finish', request_id=request_id, peak=self.peak, finish_reason=reason)
            yield Output(request_id, tokens, reason)
        except asyncio.CancelledError:
            log(event='abort', request_id=request_id)
            raise
        finally:
            self.active -= 1

    def shutdown(self):
        log(event='shutdown')
''',
    'transformers/tokenization_utils.py': '"""Transformers 5: PreTrainedTokenizerBase moved to tokenization_utils_base."""\n',
    'transformers/tokenization_utils_base.py': 'class PreTrainedTokenizerBase:\n    pass\n',
    'transformers/__init__.py': r'''"""A fake Transformers for the Goal rewriter's vLLM child: test_goal_rewriter_batch's tokenizer (one
token per character, ASCII: 137 ids; an id past them decodes to nothing, as in tokenizers), without torch."""
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

EOS, OFFSET = 1, 10
SIZE = 127 + OFFSET


class _Tensor:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def tolist(self):
        return [list(row) for row in self.rows]


class _Encoding(dict):
    def to(self, device):
        return self


class _Tokenizer(PreTrainedTokenizerBase):
    eos_token_id = EOS
    unk_token_id = None
    all_special_ids = [EOS]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return ''.join('<%s>%s' % (m['role'], m['content']) for m in messages) + '<assistant>'

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [ord(c) + OFFSET for c in text]
        if return_tensors == 'pt':
            return _Encoding(input_ids=_Tensor([ids]), attention_mask=_Tensor([[1] * len(ids)]))
        return dict(input_ids=ids)

    def __len__(self):
        return SIZE

    def encode(self, text):
        return [ord(c) + OFFSET for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(int(i) - OFFSET) if OFFSET <= int(i) < SIZE else '' for i in ids)

    def convert_tokens_to_ids(self, token):
        return EOS if token == '<|im_end|>' else None


class AutoTokenizer:
    @staticmethod
    def from_pretrained(path, local_files_only=False, trust_remote_code=False):
        assert local_files_only is True and trust_remote_code is True
        return _Tokenizer()


class GenerationConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    @classmethod
    def from_pretrained(cls, path, local_files_only=False):
        assert local_files_only is True
        return cls([EOS, 2])
''',
}

STATE = NS(directory=None, root=None, site=None, empty=None, checkpoint=None, n=0)


def setUpModule():
    STATE.directory = tempfile.TemporaryDirectory(prefix='rewriter-vllm-')
    STATE.root = Path(STATE.directory.name)
    STATE.site = STATE.root / 'site'
    for name, text in FAKE.items():
        path = STATE.site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    STATE.empty = STATE.root / 'empty-site'
    STATE.empty.mkdir()
    STATE.checkpoint = STATE.root / 'qwen3-0.6b'
    STATE.checkpoint.mkdir()
    (STATE.checkpoint / 'config.json').write_bytes(b'{"model_type":"qwen3"}')


def tearDownModule():
    STATE.directory.cleanup()


def fresh(name):
    STATE.n += 1
    return STATE.root / ('%s-%d' % (name, STATE.n))


def pin(max_new_tokens=RB.MAX_NEW, timeout=30.0, startup=60.0):
    return selection_contract.rewriter_pin(STATE.checkpoint, max_new_tokens=max_new_tokens, timeout_seconds=timeout,
                                           startup_timeout_seconds=startup)


def events(log, event=None, probe=False):
    """The fake vLLM's log rows (of ``event``); the child's startup check's rows only with ``probe``."""
    if not log.exists():
        return []
    rows = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]
    return [row for row in rows if (event is None or row['event'] == event) and
            (probe or row.get('request_id') != VC.PROBE_ID)]


def until(condition, timeout=WAIT):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


# Variables the server sets that the child must not see (and one it must only see on request).
SERVER_ENV = dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG='/x/startup.json', REMOTE_RA_DYNAMIC_STARTUP='1',
                  VLLM_POLICY_RUNTIME_V1='1', SELECTIVE_PREFIX_ARCHIVE_TEST='1', VLLM_BATCH_INVARIANT='1')


def launch(test, value=None, rewriter_pin=None, site=None, environ=None, popen=subprocess.Popen):
    """The upstream instance with the vLLM transport installed over it, its child real (the fake
    site): (inner, transport, log)."""
    rewriter_pin = pin() if rewriter_pin is None else rewriter_pin
    value = V.record(rewriter_pin['model']) if value is None else value
    argv = V.child_argv(rewriter_pin, value, OVERLAY, STATE.site if site is None else site, sys.executable)
    inner = R.PersistentCommandGoalRewriter(argv, timeout_seconds=rewriter_pin['timeout_seconds'],
                                            startup_timeout_seconds=rewriter_pin['startup_timeout_seconds'])
    log = fresh('vllm').with_suffix('.jsonl')
    env = V.child_env(value, dict(os.environ, FAKE_VLLM_LOG=str(log), **SERVER_ENV, **(environ or {})))
    stderr = open(fresh('stderr'), 'w', encoding='utf-8')
    test.addCleanup(stderr.close)
    transport = V.VllmRewriter(inner, argv, value, env=env, stderr=stderr, popen=popen)
    transport.install()
    test.addCleanup(transport.close)
    return inner, transport, log


def first_turn(k, word=''):
    """Session k's first turn (INITIALIZE: one rewriter call); its task names ``word``."""
    text = 'Fix the failing check %02d in module m%02d. %s' % (k, k, word)
    return schema.TurnContext('w%02d' % k, 'repo__w%02d' % k, 1, text, None, None, text, message_source='user',
                              structured_state={'tool_name': 'bash', 'return_code': None})


def goal(own, tap, k, word=''):
    work = GA.GoalWork('r%02d' % k, own, first_turn(k, word), tap=tap)
    return work.run(), work.rewrites


def upstream_answers(requests, max_new_tokens):
    """The upstream generate of each request alone on test_goal_rewriter_batch's fakes, as answers."""
    g = RB.generator(max_new_tokens)
    return [RB.timeless(RB.as_answer(str(k), C.alone(g, request))) for k, request in enumerate(requests)]


# ---- the child, in this process ---------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for the fake tokenizer')
class ChildFunctionTests(unittest.TestCase):
    maxDiff = None

    def test_the_prompt_and_schema_are_the_upstream_generates(self):
        prompts = VC.Prompts(RB.Tokenizer(), 384)
        g = RB.generator(384)
        requests = RB.MIXED + [RB.payload(9, 'user_request', 'keep Z', True), RB.payload(8, 'user_request')]
        for request in requests:
            with self.subTest(trigger=request['trigger'], constraints=request['allow_constraint_updates']):
                ids, value = prompts(request)
                hf = C.prepare(g, request)
                self.assertEqual(ids, hf['input_ids'][0].tolist())
                allowed = request['trigger'] == 'user_request' and request['allow_constraint_updates'] is True
                self.assertEqual(value, U.candidate_schema(constraints_allowed=allowed))
        # An error before the model call is the upstream generate's.
        bad = dict(RB.payload(7), unknown_field=1)
        with self.assertRaises(ValueError) as vllm_error:
            prompts(bad)
        with self.assertRaises(ValueError) as hf_error:
            C.prepare(g, bad)
        self.assertEqual(str(vllm_error.exception), str(hf_error.exception))
        with self.assertRaisesRegex(ValueError, 'must be a RewriteRequest object'):
            prompts(['not', 'a', 'mapping'])

    def test_decode_and_parse_are_the_upstream_generates(self):
        server = VC.Server(None, None, None, RB.Tokenizer(), VC.Prompts(RB.Tokenizer(), 384), [1, 2], True,
                           lambda value: None)
        g = RB.generator(384)
        texts = {'ok': '{"description":"Retrieve evidence","files":["a.py"]}', 'cut': '{"description":"' + 'a' * 300,
                 'prose': 'Sure: {"description":"x"}', 'unknown': '{"plan":"x"}', 'list': '[1]'}
        for name, text in texts.items():
            with self.subTest(name=name):
                tokens = RB.Tokenizer()(text)['input_ids'] + ([RB.EOS] if name != 'cut' else [])
                got = server.finish('7', tokens, 1234, 5.0)
                expected = RB.as_answer('7', C.finish(g, torch.tensor(tokens), 1234, 5.0))
                self.assertEqual(got, expected)
                self.assertEqual(json.loads(json.dumps(got)), got)
        self.assertEqual(server.finish('7', RB.Tokenizer()(texts['cut'])['input_ids'], 9, 1.0)['telemetry'],
                         dict(input_tokens=9, output_tokens=len(texts['cut']), generation_latency_ms=1.0))

    def test_the_xgrammar_schema_drops_unique_items_only(self):
        for allowed in (False, True):
            value = U.candidate_schema(constraints_allowed=allowed)
            stripped = VC.without_unique_items(value)
            self.assertNotIn('uniqueItems', json.dumps(stripped))
            self.assertIn('"uniqueItems": true', json.dumps(value))
            self.assertEqual(json.dumps(stripped), json.dumps(value).replace(', "uniqueItems": true', ''))
        self.assertEqual(V.record('/m', 'xgrammar')['schema_unique_items'], False)
        self.assertEqual(V.record('/m')['schema_unique_items'], True)

    def test_the_stop_ids_are_where_transformers_stops(self):
        tokenizer = NS(eos_token_id=7, unk_token_id=0, convert_tokens_to_ids=lambda token: 9)
        fake = NS(GenerationConfig=NS(from_pretrained=lambda path, local_files_only: NS(eos_token_id=[7, 8])))
        with patch.dict(sys.modules, transformers=fake):
            self.assertEqual(VC.stop_token_ids('/m', tokenizer), [7, 8, 9])
        tokenizer = NS(eos_token_id=7, unk_token_id=0, convert_tokens_to_ids=lambda token: 0)
        broken = NS(GenerationConfig=NS(from_pretrained=lambda path, local_files_only: 1 / 0))
        with patch.dict(sys.modules, transformers=broken), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(VC.stop_token_ids('/m', tokenizer), [7])       # the unknown token is not a stop


# ---- the child, as a subprocess ---------------------------------------------------------------------

class Child:
    """The vLLM child run directly, its command and environment as the parent builds them."""

    def __init__(self, test, value, rewriter_pin, site=None, overlay=OVERLAY, environ=None):
        self.log = fresh('child').with_suffix('.jsonl')
        argv = V.child_argv(rewriter_pin, value, overlay, STATE.site if site is None else site, sys.executable)
        env = V.child_env(value, dict(os.environ, FAKE_VLLM_LOG=str(self.log), **SERVER_ENV, **(environ or {})))
        self.stderr_path = fresh('child-stderr')
        self.stderr = open(self.stderr_path, 'w', encoding='utf-8')
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                        text=True, encoding='utf-8', bufsize=1, env=env)
        self.lines, self.seen = queue.Queue(), []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        test.addCleanup(self.close)

    def _read(self):
        for line in iter(self.process.stdout.readline, ''):
            self.seen.append(line)
            self.lines.put(line)
        self.lines.put(None)

    def send(self, value):
        self.process.stdin.write((value if isinstance(value, str) else V.canonical_text(value)) + '\n')
        self.process.stdin.flush()

    def read(self):
        return self.lines.get(timeout=WAIT)

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(WAIT)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(WAIT)
        self.process.stdout.close()
        self.stderr.close()


@unittest.skipUnless(torch is not None, 'CPU torch required for the upstream reference')
class ChildProcessTests(unittest.TestCase):
    maxDiff = None

    def test_protocol_lines_only_and_the_upstream_answers(self):
        rewriter_pin = pin(max_new_tokens=384)
        value = V.record(rewriter_pin['model'])
        child = Child(self, value, rewriter_pin)
        self.assertEqual(child.read(), READY)                       # the upstream readiness record, once up
        for k, request in enumerate(RB.MIXED):
            child.send(RB.envelope(k, request))
        child.send({V.ABORT: 'unknown'})                            # nothing in flight: no answer
        child.send('not json')
        child.send(dict(request_id='x'))
        child.send(dict(request_id=5, request={}))
        answers = [json.loads(child.read()) for _ in range(len(RB.MIXED) + 3)]
        by_id = {a['request_id']: a for a in answers if a['request_id'] in {str(k) for k in range(len(RB.MIXED))}}
        self.assertEqual([RB.timeless(by_id[str(k)]) for k in range(len(RB.MIXED))],
                         upstream_answers(RB.MIXED, 384))
        # The cut reply: the upstream GenerationFailed with the pinned max_new_tokens.
        self.assertTrue(by_id['3']['error'].startswith('GenerationFailed: JSONDecodeError'))
        self.assertEqual(by_id['3']['telemetry']['output_tokens'], 384)
        try:
            json.loads('not json')
        except ValueError as error:
            decode_error = C.error_answer('', error)
        others = [a for a in answers if a['request_id'] not in by_id]
        self.assertEqual(sorted(map(V.canonical_text, others)), sorted(map(V.canonical_text, [
            decode_error, dict(request_id='', error='ValueError: request envelope has unknown or missing fields'),
            dict(request_id=5, error='ValueError: request_id must be a non-empty string')])))
        child.close()
        self.assertEqual(child.process.returncode, 0)
        self.assertIsNone(child.lines.get(timeout=WAIT))
        # stdout: protocol lines only, each canonical JSON; vLLM's output went to stderr.
        self.assertEqual(len(child.seen), 1 + len(answers))
        for line in child.seen:
            self.assertEqual(V.canonical_text(json.loads(line)) + '\n', line)
        noise = child.stderr_path.read_text(encoding='utf-8')
        self.assertIn('INFO fake vllm: the logger writes to stdout', noise)
        self.assertIn('fake vllm: a C-level write to fd 1', noise)
        self.assertEqual([e['event'] for e in events(child.log)][-1], 'shutdown')       # the end of stdin

    def test_isolation_engine_arguments_and_sampling(self):
        rewriter_pin = pin(max_new_tokens=384)
        for backend, invariant in (('lm-format-enforcer', False), ('xgrammar', True)):
            with self.subTest(backend=backend, invariant=invariant):
                value = V.record(rewriter_pin['model'], backend, 4, invariant)
                child = Child(self, value, rewriter_pin)
                self.assertEqual(child.read(), READY)
                child.send(RB.envelope(0, RB.MIXED[2]))
                self.assertEqual(json.loads(child.read())['request_id'], '0')
                child.close()
                (imported,) = events(child.log, 'import')
                # sys.path: the given site first, then the interpreter's own library; no overlay.
                self.assertEqual(Path(imported['path'][0]), STATE.site)
                resolved = [Path(p).resolve() for p in imported['path'] if p]
                self.assertNotIn(OVERLAY.resolve(), resolved)
                self.assertTrue(Path(imported['file']).resolve().is_relative_to(STATE.site.resolve()))
                # Of the overlay only the two scripts, by file path; no torch, the site's transformers.
                modules = imported['modules']
                self.assertEqual({n for n in modules if n.startswith(('remote_ra', 'dram_kv', 'policy_runtime'))},
                                 {'remote_ra_goal_rewriter_child', 'remote_ra_qwen_goal_rewriter'})
                self.assertEqual(Path(modules['remote_ra_goal_rewriter_child']),
                                 OVERLAY / 'remote_ra_live_v2' / 'goal_rewriter_child.py')
                self.assertEqual(Path(modules['remote_ra_qwen_goal_rewriter']),
                                 OVERLAY / 'dram_kv_scripts' / 'qwen_goal_rewriter.py')
                self.assertFalse([n for n in modules if n.split('.')[0] == 'torch'])
                self.assertTrue(Path(modules['transformers']).resolve().is_relative_to(STATE.site.resolve()))
                # The environment: none of the runtime's variables; batch invariance only on request.
                self.assertFalse([k for k in imported['env'] if k.upper().startswith(V.SCRUBBED)])
                self.assertIn('FAKE_VLLM_LOG', imported['env'])
                self.assertEqual(imported['batch_invariant'], '1' if invariant else None)
                # The engine: exactly the pinned arguments.
                (engine,) = events(child.log, 'engine')
                self.assertEqual(engine['args'], value['engine_args'])
                self.assertEqual(engine['args']['structured_outputs_config'], dict(backend=backend))
                self.assertEqual(engine['args']['kv_cache_memory_bytes'], 4 << 30)
                # The request: greedy, the pinned max_new_tokens, the stop ids, the schema.
                (request,) = events(child.log, 'request')
                expected = U.candidate_schema(constraints_allowed=True)
                if backend == 'xgrammar':
                    expected = VC.without_unique_items(expected)
                self.assertEqual(request['sampling'], dict(temperature=0.0, max_tokens=384, stop_token_ids=[1, 2],
                                                           structured_outputs=dict(json=expected), detokenize=False,
                                                           output_kind='FINAL_ONLY'))
                self.assertEqual(request['prompt_ids'], VC.Prompts(RB.Tokenizer(), 384)(RB.MIXED[2])[0])
                # The startup check came first: one guided request, its schema as every request's for the
                # backend; the backend's tokenizer data was built in the (spawned) EngineCore.
                (probe, call) = events(child.log, 'request', probe=True)
                self.assertEqual((probe['request_id'], call), (VC.PROBE_ID, request))
                schema = VC.PROBE_SCHEMA if backend != 'xgrammar' else VC.without_unique_items(VC.PROBE_SCHEMA)
                self.assertEqual(probe['sampling'], dict(request['sampling'], max_tokens=VC.PROBE_TOKENS,
                                                         structured_outputs=dict(json=schema)))
                self.assertEqual(text_of(probe['prompt_ids']), '<user>%s<assistant>' % VC.PROBE_PROMPT)
                (built,) = events(child.log, 'backend')
                check_backend(self, built, backend)


    def test_the_child_refuses_another_record_or_the_overlay_on_its_path(self):
        rewriter_pin = pin()
        value = V.record(rewriter_pin['model'])
        argv = V.child_argv(rewriter_pin, value, OVERLAY, STATE.site, sys.executable)
        # The record of another engine; the overlay put on sys.path next to the site.
        other = argv[:-1] + (V.canonical_text(dict(value, engine='hf')),)
        overlay = argv[:6] + (V.PROGRAM.replace('sys.path[:0]=[sys.argv[1]]',
                                                'sys.path[:0]=[sys.argv[1],%r]' % str(OVERLAY)),) + argv[7:]
        for command, message in ((other, 'the vLLM child needs the goal_rewriter_engine record'),
                                 (overlay, 'must not see the runtime overlay')):
            with self.subTest(message=message):
                run = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                     encoding='utf-8', timeout=WAIT, env=V.child_env(value))
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(run.stdout, '')
                self.assertIn(message, run.stderr)


def check_backend(test, row, backend):
    """The backend the fake EngineCore built: for lm-format-enforcer, the Hugging Face child's tokenizer
    data (the tokenizer's 137 ids, none of the 23 padding ids that lm-format-enforcer's own vLLM builder
    would allow at every step), in a bitmask of the model's 160 ids."""
    if backend != 'lm-format-enforcer':
        test.assertEqual(row['backend'], backend)
        return
    test.assertEqual({k: v for k, v in row.items() if k not in ('event', 'pid')},
                     dict(backend=backend, tokenizer_len=137, width=160, use_bitmask=True, eos=1, same_as_hf=True,
                          max_id=136, root_beyond=[], own_root_beyond=list(range(137, 160))))


# Mutations of the child (each replacement occurs once): no module under lm-format-enforcer's vLLM name
# (anywhere, or only in a spawned EngineCore), no Transformers alias, the data over every model id, no
# startup check.
NO_MODULE = ("    install_enforcer()\nif __name__ == '__main__':", "    pass\nif __name__ == '__main__':")
NO_MP_MAIN = ("if __name__ in ('__main__', '__mp_main__'):", "if __name__ == '__main__':")
NO_ALIAS = ("    transformers_alias()\n    from lmformatenforcer import", "    from lmformatenforcer import")
ALL_IDS = ('_build_regular_tokens_list(tokenizer, size)', '_build_regular_tokens_list(tokenizer, width)')
NO_CHECK = ('        probe = server.self_test()\n', '        probe = (0, 0.0)\n')


def mutated_overlay(*replacements):
    """The three overlay files the child loads, copied, ``replacements`` applied to the child."""
    root = fresh('overlay')
    for name in ('remote_ra_live_v2/goal_rewriter_vllm_child.py', 'remote_ra_live_v2/goal_rewriter_child.py',
                 'dram_kv_scripts/qwen_goal_rewriter.py'):
        text = (OVERLAY / name).read_text(encoding='utf-8')
        if name.endswith('vllm_child.py'):
            for old, new in replacements:
                assert text.count(old) == 1, old
                text = text.replace(old, new)
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding='utf-8')
    return root


@unittest.skipUnless(torch is not None, 'CPU torch required for the upstream reference')
class GuidedBackendTests(unittest.TestCase):
    """lm-format-enforcer on vLLM 0.24 (job rw2: every lm-format-enforcer call failed) and the startup check."""
    maxDiff = None
    REFUSED = 'refused: the guided decoding (%s) failed its startup check: '

    def refused(self, child, backend, *messages):
        self.assertIsNone(child.read())                         # stdout closed without the readiness record
        child.close()
        self.assertEqual(child.process.returncode, 1)
        noise = child.stderr_path.read_text(encoding='utf-8')
        self.assertIn(self.REFUSED % backend, noise)
        for message in messages:
            self.assertIn(message, noise)

    def test_without_the_module_or_the_alias_the_child_does_not_start(self):
        rewriter_pin = pin(max_new_tokens=384)
        value = V.record(rewriter_pin['model'])
        # The EngineCore logs the backend's error and ends the request without a token (as in rw2).
        empty = "the reply did not end at a stop token (error): ''"
        engine = 'Unexpected error pre-processing request %s: ' % VC.PROBE_ID
        missing = engine + 'ImportError: vllm is not installed. Please install it with "pip install vllm"'
        for mutation, message in ((NO_MODULE, missing), (NO_MP_MAIN, missing),
                                  (NO_ALIAS, engine + 'ImportError: transformers is not installed')):
            with self.subTest(message=message):
                self.refused(Child(self, value, rewriter_pin, overlay=mutated_overlay(mutation)),
                             'lm-format-enforcer', empty, message)
        # xgrammar does not load lm-format-enforcer: that child starts without the module.
        child = Child(self, V.record(rewriter_pin['model'], 'xgrammar'), rewriter_pin,
                      overlay=mutated_overlay(NO_MODULE))
        self.assertEqual(child.read(), READY)

    def test_without_the_check_every_call_fails_as_in_rw2(self):
        rewriter_pin = pin(max_new_tokens=384)
        child = Child(self, V.record(rewriter_pin['model']), rewriter_pin,
                      overlay=mutated_overlay(NO_MODULE, NO_CHECK))
        self.assertEqual(child.read(), READY)
        for k, request in enumerate(RB.MIXED[:3]):
            child.send(RB.envelope(k, request))
        answers = [json.loads(child.read()) for _ in range(3)]
        self.assertEqual(sorted(a['request_id'] for a in answers), ['0', '1', '2'])
        for answer in answers:      # rw2's answers: no token, a parse failure
            self.assertEqual(answer['error'], 'GenerationFailed: JSONDecodeError: Expecting value: line 1 column 1 '
                                              '(char 0)')
            self.assertEqual(answer['telemetry']['output_tokens'], 0)
        self.assertEqual(events(child.log, 'backend'), [])
        child.close()
        self.assertEqual(child.stderr_path.read_text(encoding='utf-8').count('ImportError: vllm is not installed'), 3)

    def test_the_padding_ids_stay_disallowed(self):
        rewriter_pin = pin(max_new_tokens=384)
        child = Child(self, V.record(rewriter_pin['model']), rewriter_pin, overlay=mutated_overlay(ALL_IDS))
        self.assertEqual(child.read(), READY)
        child.close()
        (built,) = events(child.log, 'backend')
        self.assertEqual((built['same_as_hf'], built['root_beyond']), (False, list(range(137, 160))))
        with self.assertRaises(AssertionError):
            check_backend(self, built, 'lm-format-enforcer')

    def test_an_unguided_or_stray_reply_fails_the_check(self):
        rewriter_pin = pin(max_new_tokens=384)
        for backend in V.BACKENDS:
            value = V.record(rewriter_pin['model'], backend)
            for knob, message in (('FAKE_VLLM_UNGUIDED', "the reply is not the schema's only value: 'Hello.'"),
                                  ('FAKE_VLLM_STRAY_ID', "the reply holds ids past the tokenizer's 137: [150]")):
                with self.subTest(backend=backend, knob=knob):
                    self.refused(Child(self, value, rewriter_pin, environ={knob: '1'}), backend, message)


# ---- the parent, with the real child ------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for the upstream reference')
class TransportTests(unittest.TestCase):
    maxDiff = None

    def test_sixteen_sessions_get_the_hugging_face_goals_with_calls_in_flight(self):
        reference, hf_child = RB.unbatched(delay=lambda size: 0.0)
        self.addCleanup(reference.close)
        alone = RB.sessions_at_once(reference)
        inner, transport, log = launch(self)
        got = RB.sessions_at_once(inner)
        self.assertEqual(set(got), set(alone))
        for key in sorted(alone):
            self.assertEqual(RB.comparable(got[key][0]), RB.comparable(alone[key][0]), key)
            self.assertEqual([(x['request'], x['error'], x['input_tokens'], x['output_tokens']) for x in got[key][1]],
                             [(x['request'], x['error'], x['input_tokens'], x['output_tokens']) for x in alone[key][1]],
                             key)
        # The fallbacks of the script, per call, as with the Hugging Face child.
        exchanges = [x for _, rewrites in got.values() for x in rewrites]
        reasons = {x['error'] for x in exchanges if x['error']}
        self.assertTrue(any('GenerationFailed' in r for r in reasons))
        self.assertTrue(any('symbols must be unique' in r for r in reasons))
        # The engine got the prompts the Hugging Face child built, one request per call.
        prompts = VC.Prompts(RB.Tokenizer(), RB.MAX_NEW)
        requests = events(log, 'request')
        self.assertEqual(sorted(r['prompt_ids'] for r in requests),
                         sorted(prompts(json.loads(line)['request'])[0] for line in hf_child.lines))
        self.assertEqual(len(requests), len(exchanges))
        # Several calls in flight; answered out of order and routed to their calls.
        finished = events(log, 'finish')
        self.assertGreaterEqual(max(e['peak'] for e in finished), 2)
        self.assertNotEqual([e['request_id'] for e in finished], [r['request_id'] for r in requests])
        for x in exchanges:
            self.assertEqual(set(x) & set(RB.goal_text.BATCH), set())
            self.assertTrue(all(type(x[k]) is float and x[k] >= 0 for k in RB.goal_text.TIMING), x)
            self.assertIsNotNone(x['response'])
        # The calls did not queue behind each other in the parent: wait_ms is not the calls' time.
        self.assertLess(sum(x['wait_ms'] for x in exchanges), sum(x['call_ms'] for x in exchanges))
        self.assertEqual(transport.starts, 1)

    def test_an_engine_error_fails_its_call_only(self):
        inner, transport, _ = launch(self)
        own, tap = RB.constructors(inner)
        failed, rewrites = goal(own, tap, 0, 'ENGINE_ERROR')
        self.assertEqual(failed.fallback_reason, 'RewriteCommandError: persistent rewrite failed: ValueError: the '
                                                 'fake engine refused this request')
        self.assertEqual(json.loads(rewrites[0]['response'])['error'], 'ValueError: the fake engine refused this request')
        self.assertTrue(goal(own, tap, 1)[0].slm_succeeded)
        self.assertEqual(transport.starts, 1)

    def test_a_timeout_aborts_its_request_and_keeps_the_child(self):
        inner, transport, log = launch(self, rewriter_pin=pin(timeout=5.0))
        own, tap = RB.constructors(inner)
        pid = transport._child.process.pid
        with ThreadPoolExecutor(2) as pool:
            hung = pool.submit(goal, own, tap, 0, 'HANG')
            self.assertTrue(until(lambda: events(log, 'request')))
            other = pool.submit(goal, own, tap, 1)
            fine, _ = other.result(WAIT)
            self.assertFalse(hung.done())                       # the hang holds no other call
            timed_out, rewrites = hung.result(WAIT)
        self.assertTrue(fine.slm_succeeded)
        self.assertEqual(timed_out.fallback_reason, 'RewriteCommandError: persistent rewrite process timed out')
        (x,) = rewrites
        self.assertEqual((x['response'], x['input_tokens']), (None, None))
        self.assertGreaterEqual(x['call_ms'], 5000 * 0.9)
        hang_id = next(r['request_id'] for r in events(log, 'request') if 'HANG' in text_of(r['prompt_ids']))
        self.assertTrue(until(lambda: [e for e in events(log, 'abort') if e['request_id'] == hang_id]))
        # The child keeps running: the next call goes to it.
        self.assertTrue(goal(own, tap, 2)[0].slm_succeeded)
        self.assertEqual((transport.starts, transport._child.process.pid), (1, pid))
        self.assertEqual(transport._routes, {})

    def test_a_child_exit_fails_the_calls_in_flight_and_the_next_call_restarts_it(self):
        inner, transport, log = launch(self)
        own, tap = RB.constructors(inner)
        first = transport._child.process
        with ThreadPoolExecutor(2) as pool:
            slow = pool.submit(goal, own, tap, 0, 'SLOW')
            self.assertTrue(until(lambda: events(log, 'request')))
            crashed, _ = goal(own, tap, 1, 'CRASH')
            slowed, _ = slow.result(WAIT)
        reason = 'RewriteCommandError: persistent rewrite process closed stdout (exit=3)'
        self.assertEqual((crashed.fallback_reason, slowed.fallback_reason), (reason, reason))
        self.assertEqual(first.wait(WAIT), 3)
        restarted, rewrites = goal(own, tap, 2)
        self.assertTrue(restarted.slm_succeeded)
        self.assertEqual(transport.starts, 2)
        self.assertNotEqual(transport._child.process.pid, first.pid)
        self.assertGreater(rewrites[0]['start_ms'], 0)              # the restart inside this call
        self.assertGreater(restarted.diagnostics['slm_startup_latency_ms'], 0)
        # An engine that died answers its call and ends its child; the next call restarts it.
        dead, _ = goal(own, tap, 3, 'ENGINE_DEAD')
        self.assertEqual(dead.fallback_reason, 'RewriteCommandError: persistent rewrite failed: RuntimeError: '
                                               'EngineDeadError: the fake engine died')
        self.assertTrue(until(lambda: transport._child.ended.is_set()))
        self.assertTrue(goal(own, tap, 4)[0].slm_succeeded)
        self.assertEqual(transport.starts, 3)

    def test_shutdown_fails_the_call_in_flight_and_ends_the_child(self):
        inner, transport, log = launch(self)
        own, tap = RB.constructors(inner)
        process = transport._child.process
        with ThreadPoolExecutor(1) as pool:
            slow = pool.submit(goal, own, tap, 0, 'SLOW')
            self.assertTrue(until(lambda: events(log, 'request')))
            began = time.monotonic()
            inner.close()
            stopped, _ = slow.result(WAIT)
        self.assertLess(time.monotonic() - began, V.CLOSE_SECONDS)
        self.assertEqual(stopped.fallback_reason, 'RewriteCommandError: persistent rewrite process is not running')
        self.assertEqual(process.poll(), 0)                     # it shut its engine down and exited
        self.assertIn('shutdown', [e['event'] for e in events(log)])
        self.assertFalse(transport.alive())
        self.assertEqual(goal(own, tap, 1)[0].fallback_reason,
                         'RewriteCommandError: persistent rewrite process is not running')
        inner.close()                                           # idempotent

    def test_a_child_that_does_not_start_fails_the_startup(self):
        rewriter_pin = pin(startup=3.0)
        value = V.record(rewriter_pin['model'])
        with self.assertRaisesRegex(R.RewriteCommandError, r'closed stdout \(exit=1\)'):
            launch(self, value, rewriter_pin, site=STATE.empty)          # no vllm, no transformers
        for argv, message in (((sys.executable, '-I', '-c', 'print("{}")'), 'invalid readiness record'),
                              ((sys.executable, '-I', '-c', 'import time; time.sleep(60)'), 'timed out')):
            with self.subTest(message=message):
                inner = R.PersistentCommandGoalRewriter(argv, timeout_seconds=5.0, startup_timeout_seconds=3.0)
                spawned = []

                def popen(*args, **kwargs):
                    spawned.append(subprocess.Popen(*args, **kwargs))
                    return spawned[-1]
                transport = V.VllmRewriter(inner, argv, value, popen=popen)
                with self.assertRaisesRegex(R.RewriteCommandError, message):
                    transport.install()
                self.assertIsNotNone(spawned[0].poll())             # ended, not left running
                self.assertFalse(hasattr(inner, '_remote_ra_vllm'))

    def test_its_own_session_scrubbed_environment_and_control_hash(self):
        calls = []

        def popen(*args, **kwargs):
            calls.append(kwargs)
            return subprocess.Popen(*args, **kwargs)
        inner, transport, _ = launch(self, popen=popen)
        (kwargs,) = calls
        self.assertIs(kwargs['start_new_session'], True)
        self.assertFalse([k for k in kwargs['env'] if k.upper().startswith(V.SCRUBBED)])
        self.assertNotIn(V.BATCH_INVARIANT_ENV, kwargs['env'])
        self.assertEqual({k: kwargs['env'][k] for k in V.OFFLINE}, V.OFFLINE)
        # The constructor keeps an upstream instance: its type and the Goal control hash are unchanged.
        reference = R.PersistentCommandGoalRewriter(('unused',))
        self.assertIs(type(inner), R.PersistentCommandGoalRewriter)
        hashed = lambda rewriter: selection_contract.constructor_control_hash(
            maintained.MaintainedGoalConstructor(rewriter, entity_drift_steps=3, initialize_with_rewriter=True))
        self.assertEqual(hashed(inner), hashed(reference))
        value = transport.record
        self.assertIs(V.check(inner, value), value)
        self.assertIsNone(V.check(reference, None))
        with self.assertRaisesRegex(LiveV2Error, 'with the pinned record'):
            V.check(inner, V.record(value['engine_args']['model'], 'xgrammar'))
        with self.assertRaisesRegex(LiveV2Error, 'with the pinned record'):
            V.check(reference, value)
        with self.assertRaisesRegex(LiveV2Error, 'not pinned'):
            V.check(inner, None)
        own, tap = RB.constructors(inner)
        self.assertEqual(transport.stamp, tap.stamp)
        self.assertEqual(V.attached(inner, value), {V.STARTUP_KEY: transport.startup_ms})
        self.assertGreater(transport.startup_ms, 0)
        with self.assertRaisesRegex(LiveV2Error, 'already running, batched or on vLLM'):
            V.VllmRewriter(inner, transport.argv, value).install()

    def test_the_authority_starts_the_vllm_rewriter_only_with_the_pin(self):
        import test_selection_stageb as SEL
        calls = []
        log = fresh('authority').with_suffix('.jsonl')

        def start(rewriter_pin, **kwargs):
            calls.append(kwargs)
            if not kwargs:
                inner, _ = RB.unbatched()
                self.addCleanup(inner.close)
                return inner
            stderr = open(fresh('stderr'), 'w', encoding='utf-8')
            self.addCleanup(stderr.close)
            with patch.dict(os.environ, FAKE_VLLM_LOG=str(log)):
                inner = V.start(R, rewriter_pin, kwargs['engine'], OVERLAY, STATE.site, sys.executable, stderr=stderr)
            self.addCleanup(inner.close)
            return inner
        rewriter_pin = pin()
        value = V.record(rewriter_pin['model'])
        contract = selection_contract.default_contract(SEL.P, goal_rewriter=rewriter_pin)
        with patch.object(query_selection, 'goal_rewriter', start):
            auth, _ = SEL.authority('goal_query', contract, policy='goal_query')
            self.assertIsNone(auth.rewriter_engine)
            self.assertIsNone(getattr(auth.rewriter, '_remote_ra_vllm', None))
            auth, _ = SEL.authority('goal_query', contract, policy='goal_query', rewriter_engine=value)
            self.assertEqual(auth.rewriter_engine, value)
            self.assertEqual(auth.rewriter._remote_ra_vllm.stamp, auth.tap.stamp)
            self.assertTrue(auth.constructor.construct(first_turn(0)).slm_succeeded)
            with self.assertRaisesRegex(LiveV2Error, 'goal_query with a pinned Goal rewriter'):
                SEL.authority('observation_query', policy='observation_query', rewriter_engine=value)
            with self.assertRaisesRegex(LiveV2Error, 'excludes --goal-rewriter-batch'):
                SEL.authority('goal_query', contract, policy='goal_query', rewriter_engine=value, rewriter_batch=4)
        self.assertEqual(calls, [{}, {'engine': value}])
        self.assertEqual(len(events(log, 'engine')), 1)


def text_of(ids):
    """The fake tokenizer's text of token ids."""
    return ''.join(chr(i - RB.OFFSET) for i in ids if i >= RB.OFFSET)


# ---- end to end: the Gemma 4 window layout's engine and worker fakes with the vLLM child ------------

class EngineRewriter:
    """query_selection.goal_rewriter as the e2e harness patches it: test_gemma_goal_rewriter's in-process
    Hugging Face child without an engine record; with one, the vLLM transport and its real child (built
    from the modules the engine imports; max_new_tokens as the in-process child's)."""

    def __init__(self, test, log):
        self.test, self.log, self.pin, self.engines = test, log, None, []
        self.hf = test.GGR.RewriterChild(test)

    @property
    def children(self):
        return self.hf.children

    def __call__(self, rewriter_pin, batch=None, tokens=0, engine=None):
        self.test.assertEqual(rewriter_pin, self.pin)
        if engine is None:
            self.hf.pin = self.pin
            return self.hf(rewriter_pin, batch, tokens)
        import importlib
        live = self.test.GGR.GO.live('goal_rewriter_vllm')
        stderr = open(fresh('e2e-stderr'), 'w', encoding='utf-8')
        self.test.addCleanup(stderr.close)
        with patch.dict(os.environ, FAKE_VLLM_LOG=str(self.log)):
            rewriter = live.start(importlib.import_module('dram_kv.goals.rewriter'),
                                  dict(rewriter_pin, max_new_tokens=RB.MAX_NEW), engine, OVERLAY, STATE.site,
                                  sys.executable, stderr=stderr)
        self.test.addCleanup(rewriter.close)
        self.engines.append(engine)
        return rewriter


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows and the fake rewriter model')
class GemmaSessionsOnVllmTests(unittest.TestCase):
    """test_gemma_goal_rewriter's sixteen goal_query sessions (the Gemma 4 window layout,
    --admission-overlap 16, Goals parked on their sessions' threads, --release-idle-pages), every
    turn's Goal a rewrite. With goal_rewriter_engine pinned in the startup config the authority
    (private_endpoint) starts the vLLM child, and every turn decides, selects, attends and replies as
    with the Hugging Face child's fakes, its exchanges equal and without batch keys."""
    maxDiff = None
    TURNS = 3

    @classmethod
    def setUpClass(cls):
        import test_gemma_goal_rewriter as GGR
        cls.GGR = GGR
        base = GGR.SixteenBatchedSessionsTests
        for name in ('check', 'boundary', 'window_of', 'finish', 'watch', 'goal_checks', 'pool', 'server', 'done',
                     'run_case', 'TEST_GOAL'):
            setattr(cls, name, base.__dict__[name])
        cls.checkpoint = fresh('gemma-checkpoint')
        cls.checkpoint.mkdir()
        (cls.checkpoint / 'config.json').write_bytes(b'{"model_type":"qwen3"}')

    def setUp(self):
        GGR = self.GGR
        self.enterContext(patch.dict(sys.modules, GGR.W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.enterContext(patch.object(GGR.GE, 'GOAL', self.TEST_GOAL))
        self.enterContext(patch.object(GGR.GR, 'script', GGR.rewrite_script))
        self.log = fresh('gemma-vllm').with_suffix('.jsonl')
        self.rewriter = EngineRewriter(self, self.log)
        self.rewriter.pin = GGR.GO.live('selection_contract').rewriter_pin(self.checkpoint)
        self.enterContext(patch.object(GGR.GO.live('query_selection'), 'goal_rewriter', self.rewriter))

    def on_vllm(self):
        """The sixteen sessions' run with goal_rewriter_engine pinned."""
        from contextlib import ExitStack
        GGR = self.GGR
        GR, GO, GE = GGR.GR, GGR.GO, GGR.GE
        live = GO.live('goal_rewriter_vllm')
        records = []
        with ExitStack() as stack:
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            sessions = GR.SESSIONS
            layout = GE.window_layout('goal_query', sessions)
            value = GGR.with_rewriter(self, GR.config('goal_query', output, GGR.long_tasks(sessions), layout, True),
                                      None, 0, sessions)
            record = live.record(self.rewriter.pin['model'])
            value = dict(value, **{live.CONFIG_KEY: record})
            self.assertEqual(live.pinned(value), record)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = GO.OverlapPressure('goal_query', output, GGR.long_tasks(sessions), self.pool(), True, self.TURNS,
                                   sessions)
            h.output, h.turns = output, self.TURNS
            self.assertEqual((h.selection.rewriter_engine, h.selection.rewriter_batch), (record, None))
            self.watch(records)(h)
            h.serve(self.check)
            return self.done(h, records, None), record

    def test_sixteen_sessions_decide_as_with_the_hugging_face_child(self):
        GGR = self.GGR
        alone = self.run_case(self.pool(), None, 0)
        run, record = self.on_vllm()
        self.assertEqual(self.rewriter.engines, [record])
        self.assertEqual(run.result, alone.result)
        self.assertEqual(GGR.exchanges_of(run.rows), GGR.exchanges_of(alone.rows))
        exchanges = [x for row in run.rows for x in row['rewrites']]
        self.assertEqual(len(exchanges), GGR.GR.SESSIONS * self.TURNS)       # every Goal a rewrite
        self.assertTrue(all(set(RB.goal_text.BATCH).isdisjoint(x) for x in exchanges))
        self.assertEqual(len(events(self.log, 'request')), len(exchanges))
        self.assertGreaterEqual(max(e['peak'] for e in events(self.log, 'finish')), 2)    # calls in flight
        for case in (alone, run):
            self.assertGreaterEqual(case.h.parked_peak, 2)
            self.goal_checks(case.records, GGR.GR.SESSIONS)


@unittest.skipUnless(hasattr(os, 'killpg') and sys.platform != 'win32', 'process groups: POSIX only')
class ProcessGroupTests(unittest.TestCase):
    def test_stop_ends_the_whole_group(self):
        # A child in its own session whose own child (an EngineCore stand-in) ignores SIGTERM.
        script = ('import os,signal,subprocess,sys,time;'
                  'g=subprocess.Popen([sys.executable,"-c","import signal,time;'
                  'signal.signal(signal.SIGTERM,signal.SIG_IGN);print(1,flush=True);time.sleep(600)"],'
                  'stdout=subprocess.PIPE);g.stdout.readline();print(g.pid,flush=True);time.sleep(600)')
        process = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, text=True,
                                   start_new_session=True)
        self.addCleanup(process.stdout.close)
        grandchild = int(process.stdout.readline())

        def gone():
            try:
                with open('/proc/%d/stat' % grandchild) as stream:
                    return stream.read().rsplit(')', 1)[1].split()[0] == 'Z'
            except FileNotFoundError:
                return True
        self.assertFalse(gone())
        began = time.monotonic()
        V.stop(process, grace=1.0)
        self.assertIsNotNone(process.poll())
        self.assertTrue(until(gone, 10))
        self.assertLess(time.monotonic() - began, 10)
        with self.assertRaises(ProcessLookupError):
            os.killpg(process.pid, 0)


# ---- launcher, startup config and evidence ------------------------------------------------------------

def engine_kwargs(**kw):
    return dict(dict(admission_overlap=4, goal_rewriter_engine='vllm'), **kw)


class LauncherTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import serve
        import test_installer as INSTALL
        import install
        cls.serve = serve
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with contextlib.redirect_stdout(io.StringIO()):
            package, site, model, cls.release = INSTALL.fixture(cls.root)
            install.install(site, model, cls.release, package_root=package, apply=True)
        cls.rewriter = cls.root / 'qwen3-0.6b'
        cls.rewriter.mkdir()
        (cls.rewriter / 'config.json').write_bytes(b'{"model_type":"qwen3"}')
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.enterContext(patch.object(sys, 'path', [str(OVERLAY)] + sys.path))

    def prepare(self, policy='goal_query', sessions=4, selection=None, **kwargs):
        """serve.prepare of a 4-session server at 16-token pages: (config, command, env, config as written)."""
        type(self).n += 1
        out = self.root / ('out%d' % self.n)
        tasks = [dict(instance_id='repo__m%02d-%d' % (k, k), problem_statement='Fix check %02d.' % k)
                 for k in range(sessions)]
        if selection is None:
            selection = dict(goal_rewriter_model=str(self.rewriter)) if policy == 'goal_query' else {}
        with contextlib.redirect_stdout(io.StringIO()):
            _, config, command, env = self.serve.prepare(self.release, tasks, policy, out, 8000, selection,
                                                         'mswea-xml', 24, 16, None, **kwargs)
        named = json.loads(json.dumps(config).replace(json.dumps(str(out))[1:-1], 'OUT'))
        return named, [a.replace(str(out), 'OUT') for a in command], \
            {k: v.replace(str(out), 'OUT') for k, v in env.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'}, config

    def load(self, config):
        from remote_ra_http import config as HC
        type(self).n += 1
        path = self.root / ('startup%d.json' % self.n)
        path.write_bytes(self.serve.wire(config))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                        REMOTE_RA_HTTP_CONFIG_SHA256=self.serve.pin(config)):
            return HC.load()

    def test_vllm_pins_its_record_and_nothing_else(self):
        plain, plain_command, plain_env, _ = self.prepare(admission_overlap=4)
        model = str(self.rewriter.resolve())
        for kw, record in ((dict(), V.record(model)),
                           (dict(goal_rewriter_guided_backend='xgrammar'), V.record(model, 'xgrammar')),
                           (dict(goal_rewriter_vllm_kv_gib=8, goal_rewriter_batch_invariant=True),
                            V.record(model, kv_gib=8, batch_invariant=True)),
                           (dict(goal_rewriter_guided_backend='lm-format-enforcer', goal_rewriter_vllm_kv_gib=3),
                            V.record(model))):
            with self.subTest(kw=kw):
                config, command, env, written = self.prepare(**engine_kwargs(**kw))
                self.assertEqual(config.pop(V.CONFIG_KEY), record)
                self.assertEqual((config, command, env), (plain, plain_command, plain_env))   # no vLLM argument
                self.assertEqual(self.load(written)[V.CONFIG_KEY], record)
        value = V.record(model)
        self.assertEqual((value['guided_backend'], value['kv_cache_memory_bytes'], value['gpu_memory_utilization'],
                          value['batch_invariant'], value['schema_unique_items']),
                         ('lm-format-enforcer', 3 << 30, 0.05, False, True))
        self.assertEqual({k: value['engine_args'][k] for k in ('max_num_seqs', 'max_model_len', 'max_num_batched_tokens',
                                                               'enable_prefix_caching', 'enforce_eager',
                                                               'generation_config', 'dtype')},
                         dict(max_num_seqs=16, max_model_len=16384, max_num_batched_tokens=8192,
                              enable_prefix_caching=False, enforce_eager=False, generation_config='vllm',
                              dtype='bfloat16'))
        self.assertEqual(value['engine_args']['compilation_config'], dict(cudagraph_capture_sizes=[1, 2, 4, 8, 16]))
        self.assertEqual(value['engine_args']['model'], model)

    def test_hf_is_the_default_and_adds_nothing(self):
        plain = self.prepare(admission_overlap=4)[:3]
        self.assertEqual(self.prepare(admission_overlap=4, goal_rewriter_engine='hf')[:3], plain)
        self.assertNotIn(V.CONFIG_KEY, plain[0])
        for policy in ('full', 'observation_query'):
            self.assertEqual(self.prepare(policy, goal_rewriter_engine='hf')[:3], self.prepare(policy)[:3])
        # The Hugging Face child's command, as before.
        rewriter_pin = selection_contract.rewriter_pin(self.rewriter)
        self.assertEqual(query_selection.rewriter_argv(rewriter_pin, OVERLAY, '/site', '/py'), (
            '/py', '-I', '-S', '-B', '-u', '-c',
            "import sys,runpy;sys.path[:0]=[sys.argv[1],sys.argv[2]];script=sys.argv[3];"
            "sys.argv=[script]+sys.argv[4:];runpy.run_path(script,run_name='__main__')",
            str(OVERLAY), '/site', str(OVERLAY / 'dram_kv_scripts' / 'qwen_goal_rewriter.py'), '--model',
            rewriter_pin['model'], '--device', 'cuda', '--dtype', 'bfloat16', '--max-new-tokens', '384',
            '--local-files-only', '--serve'))

    def test_refusals(self):
        cases = [
            (dict(goal_rewriter_engine='sglang'), 'must be hf or vllm'),
            (dict(admission_overlap=4, goal_rewriter_guided_backend='xgrammar'), 'apply to --goal-rewriter-engine vllm only'),
            (dict(admission_overlap=4, goal_rewriter_vllm_kv_gib=3), 'apply to --goal-rewriter-engine vllm only'),
            (dict(admission_overlap=4, goal_rewriter_batch_invariant=True), 'apply to --goal-rewriter-engine vllm only'),
            (dict(goal_rewriter_engine='vllm'), 'needs --admission-overlap K >= 2'),
            (dict(admission_overlap=1, goal_rewriter_engine='vllm'), 'needs --admission-overlap K >= 2'),
            (engine_kwargs(goal_rewriter_batch=4), 'excludes --goal-rewriter-batch'),
            (engine_kwargs(goal_rewriter_guided_backend='guidance'), 'must be one of lm-format-enforcer, xgrammar'),
            (engine_kwargs(goal_rewriter_guided_backend='auto'), 'must be one of lm-format-enforcer, xgrammar'),
            (engine_kwargs(goal_rewriter_vllm_kv_gib=1), 'integer 2-16'),
            (engine_kwargs(goal_rewriter_vllm_kv_gib=17), 'integer 2-16'),
            (engine_kwargs(goal_rewriter_vllm_kv_gib='3'), 'integer 2-16'),
            (engine_kwargs(goal_rewriter_vllm_kv_gib=True), 'integer 2-16'),
            (engine_kwargs(goal_rewriter_batch_invariant='yes'), 'is a switch'),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                self.prepare(**kwargs)
        with self.assertRaisesRegex(ValueError, 'goal_query with --goal-rewriter-model'):
            self.prepare(selection=dict(no_goal_rewriter=True), **engine_kwargs())
        for policy in ('full', 'observation_query', 'random_fill'):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, 'goal_query with --goal-rewriter-model'):
                self.prepare(policy, **engine_kwargs())

    def test_load_checks_the_pin(self):
        from remote_ra_http import config as HC
        _, _, _, config = self.prepare(**engine_kwargs())
        value = config[V.CONFIG_KEY]
        self.assertEqual(HC.goal_rewriter_engine(config), value)
        self.assertIsNone(HC.goal_rewriter_engine(self.prepare(admission_overlap=4)[3]))

        def changed(**fields):
            return dict(config, **{V.CONFIG_KEY: dict(value, **fields)})
        args = value['engine_args']
        for bad, message in (
                (changed(engine_args=dict(args, max_num_seqs=8)), 'differs from the record'),
                (changed(engine_args=dict(args, model='/elsewhere')), 'differs from the record'),
                (changed(gpu_memory_utilization=0.9), 'differs from the record'),
                (changed(schema_unique_items=False), 'differs from the record'),
                (changed(kv_cache_memory_bytes=(3 << 30) + 1), 'must be the record'),
                (changed(kv_cache_memory_bytes=1 << 30), 'integer 2-16'),
                (changed(guided_backend='guidance'), 'must be one of'),
                (changed(batch_invariant='yes'), 'is a switch'),
                (dict(config, **{V.CONFIG_KEY: 'vllm'}), 'must be the record'),
                (dict(config, admission=None), 'K >= 2'),
                (dict(config, goal_rewriter_batch=2, goal_rewriter_batch_tokens=0), 'excludes goal_rewriter_batch'),
                (dict(config, policy='observation_query'), 'needs goal_query'),
                (dict(config, selection_contract=dict(config['selection_contract'], goal_rewriter=None)),
                 'needs goal_query')):
            with self.subTest(bad=bad.get(V.CONFIG_KEY)), self.assertRaisesRegex(RuntimeError, message):
                HC.goal_rewriter_engine(bad)
        self.assertEqual(self.load(config)[V.CONFIG_KEY], value)
        with self.assertRaisesRegex(RuntimeError, 'differs from the record'):
            self.load(changed(engine_args=dict(args, max_num_seqs=8)))

    def test_the_dry_run_records_it(self):
        task = self.root / 'tasks.json'
        task.write_text(json.dumps([dict(instance_id='repo__m%02d' % k, problem_statement='Fix check %02d.' % k)
                                    for k in range(4)]))
        def dry(*flags):
            """serve.py's dry-run line, its output directory named OUT."""
            type(self).n += 1
            output = self.root / ('dry%d' % self.n)
            argv = ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--task', str(task), '--policy', 'goal_query',
                    '--output', str(output), '--goal-rewriter-model', str(self.rewriter),
                    '--page-tokens', '16', '--kv-cache-gib', '24', '--admission-overlap', '4', *flags]
            stdout = io.StringIO()
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(stdout):
                self.serve.main()
            return json.loads(stdout.getvalue().replace(json.dumps(str(output))[1:-1], 'OUT'))
        plain = dry()
        self.assertEqual(dry('--goal-rewriter-engine', 'hf'), plain)
        value = dry('--goal-rewriter-engine', 'vllm', '--goal-rewriter-guided-backend', 'xgrammar',
                    '--goal-rewriter-vllm-kv-gib', '4', '--goal-rewriter-batch-invariant')
        self.assertEqual(value.pop(V.CONFIG_KEY), V.record(str(self.rewriter.resolve()), 'xgrammar', 4, True))
        self.assertNotIn(V.CONFIG_KEY, plain)
        self.assertEqual(value, plain)                              # the vLLM command unchanged
        with patch.object(sys, 'argv', ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--task', str(task),
                                        '--policy', 'goal_query', '--output', str(self.root / 'refused'),
                                        '--goal-rewriter-vllm-kv-gib', '4', '--no-goal-rewriter']), \
                self.assertRaisesRegex(ValueError, 'apply to --goal-rewriter-engine vllm only'):
            self.serve.main()

    def test_the_child_command(self):
        rewriter_pin = selection_contract.rewriter_pin(self.rewriter)
        value = V.record(rewriter_pin['model'])
        argv = V.child_argv(rewriter_pin, value, OVERLAY, '/site', '/py')
        self.assertEqual(argv, ('/py', '-I', '-S', '-B', '-u', '-c', V.PROGRAM, '/site',
                                str(OVERLAY / 'remote_ra_live_v2' / V.CHILD), '--model', rewriter_pin['model'],
                                '--max-new-tokens', '384', '--local-files-only', '--serve', '--engine',
                                V.canonical_text(value)))
        self.assertIn('sys.path[:0]=[sys.argv[1]]', V.PROGRAM)       # the site only, not the overlay
        with self.assertRaisesRegex(LiveV2Error, 'names another checkpoint'):
            V.child_argv(rewriter_pin, V.record('/elsewhere'), OVERLAY, '/site', '/py')
        (self.rewriter / 'config.json').write_bytes(b'{"model_type":"other"}')
        try:
            with self.assertRaisesRegex(LiveV2Error, 'differs from its startup pin'):
                V.child_argv(rewriter_pin, value, OVERLAY, '/site', '/py')
        finally:
            (self.rewriter / 'config.json').write_bytes(b'{"model_type":"qwen3"}')
        environ = dict(PATH='/bin', REMOTE_RA_HTTP_RUNTIME='1', VLLM_POLICY_RUNTIME_V1='1',
                       SELECTIVE_PREFIX_ARCHIVE_TEST='1', VLLM_BATCH_INVARIANT='1', VLLM_USE_V2_MODEL_RUNNER='0',
                       CUDA_VISIBLE_DEVICES='0', TMPDIR='/t')
        self.assertEqual(V.child_env(value, environ), dict(PATH='/bin', VLLM_USE_V2_MODEL_RUNNER='0',
                                                            CUDA_VISIBLE_DEVICES='0', TMPDIR='/t', **V.OFFLINE))
        self.assertEqual(V.child_env(V.record(rewriter_pin['model'], batch_invariant=True), environ)
                         [V.BATCH_INVARIANT_ENV], '1')


class GemmaLauncherTests(unittest.TestCase):
    """The Gemma 4 profile takes the flags as Qwen does (the same child and record)."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import install
        from test_gemma_install import gemma_fixture
        from remote_ra_live_v2 import model_profile as MP
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with contextlib.redirect_stdout(io.StringIO()):
            package, site, model, gemma, cls.release = gemma_fixture(cls.root)
            install.install(site, gemma, cls.release, package_root=package, apply=True, model_profile=MP.GEMMA4)
        cls.rewriter = cls.root / 'qwen3-0.6b'
        cls.rewriter.mkdir()
        (cls.rewriter / 'config.json').write_bytes(b'{"model_type":"qwen3"}')
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.enterContext(patch.object(sys, 'path', [str(OVERLAY)] + sys.path))

    def prepare(self, policy='goal_query', protocol='mswea-xml', **kw):
        import serve
        import test_gemma_goal_rewriter as GGR
        kw = dict(dict(kv_cache_gib=28, page_tokens=16, max_tokens=4096, max_model_len=131072,
                       language_model_only=True, release_idle_pages=True, dram_arena_gib=79), **kw)
        selection = dict(active_token_budget=8192)
        if policy == 'goal_query':
            selection.update(goal_rewriter_model=str(self.rewriter))
        task = GGR.tasks(16)
        if protocol == 'chat-text':
            task, kw = None, dict(kw, protocol='chat-text', max_running=20, max_model_len=65536, dram_arena_gib=64)
        type(self).n += 1
        output = self.root / ('out%d' % self.n)
        _, config, command, _ = serve.prepare(self.release, task, policy, output, 8000, selection, **kw)
        return GGR.without_output([config, command], output), config

    def test_vllm_pins_its_record_and_nothing_else(self):
        from remote_ra_http import config as HC
        import test_gemma_goal_rewriter as GGR
        model = str(self.rewriter.resolve())
        for protocol in ('mswea-xml', 'chat-text'):
            with self.subTest(protocol=protocol):
                (plain, plain_command), _ = self.prepare(protocol=protocol, admission_overlap=16)
                (config, command), written = self.prepare(protocol=protocol, admission_overlap=16,
                                                          goal_rewriter_engine='vllm',
                                                          goal_rewriter_guided_backend='xgrammar')
                self.assertEqual(config.pop(V.CONFIG_KEY), V.record(model, 'xgrammar'))
                self.assertEqual((config, command), (plain, plain_command))
                self.assertEqual(self.prepare(protocol=protocol, admission_overlap=16, goal_rewriter_engine='hf')[0],
                                 [plain, plain_command])
                loaded = self.load(written)
                self.assertEqual((loaded['model_profile'], loaded[V.CONFIG_KEY]), ('gemma4-31b-it',
                                                                                  V.record(model, 'xgrammar')))
                self.assertEqual(HC.goal_rewriter_engine(written), V.record(model, 'xgrammar'))
        self.assertEqual(GGR.GEMMA_T, 13312)                         # the batch's default is untouched

    def load(self, config):
        import serve
        from remote_ra_http import config as HC
        type(self).n += 1
        path = self.root / ('startup%d.json' % self.n)
        path.write_bytes(serve.wire(config))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                        REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
            return HC.load()

    def test_the_refusals_are_qwens(self):
        cases = [(dict(goal_rewriter_engine='vllm'), 'needs --admission-overlap K >= 2'),
                 (dict(admission_overlap=16, goal_rewriter_engine='vllm', goal_rewriter_batch=4),
                  'excludes --goal-rewriter-batch'),
                 (dict(admission_overlap=16, goal_rewriter_vllm_kv_gib=4), 'apply to --goal-rewriter-engine vllm only'),
                 (dict(admission_overlap=16, goal_rewriter_engine='vllm', goal_rewriter_vllm_kv_gib=32), 'integer 2-16')]
        for kw, message in cases:
            with self.subTest(kw=kw), self.assertRaisesRegex(ValueError, message):
                self.prepare(**kw)
        with self.assertRaisesRegex(ValueError, 'goal_query with --goal-rewriter-model'):
            self.prepare('observation_query', admission_overlap=16, goal_rewriter_engine='vllm')


class EvidenceTests(unittest.TestCase):
    """worker-attached.json and ready.json: the record and the child's start time, only with vllm."""
    maxDiff = None

    def config(self, output, engine):
        import test_decouple_evidence as DE
        from remote_ra_live_v2 import admission_flags
        rewriter_pin = selection_contract.rewriter_pin(STATE.checkpoint)
        value = DE.config(output, 'goal_query', 784, False)
        value.update(selection_contract=selection_contract.default_contract(784, goal_rewriter=rewriter_pin),
                     admission=admission_flags.record(admission_overlap=2))
        if engine:
            value[V.CONFIG_KEY] = V.record(rewriter_pin['model'], 'xgrammar')
        return value

    def files(self, engine, startup_ms=812.25):
        """One startup's worker-attached.json and ready.json (evidence.worker_attached, write_ready)."""
        import pathlib
        from remote_ra_http import evidence as EV
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            value = self.config(output, engine)
            transport = None
            if engine:
                inner = R.PersistentCommandGoalRewriter(('unused',))
                transport = V.VllmRewriter(inner, ('unused',), value[V.CONFIG_KEY])
                transport.startup_ms = startup_ms
            selection = NS(task=value['task'], tasks=lambda: value['tasks'], rewriter=NS(_remote_ra_vllm=transport))
            worker, bridge, life = NS(), NS(), NS()
            core = NS(_remote_ra_live_v2_boundary=NS(proxy=worker), _remote_ra_live_v2_lifecycle=life,
                      scheduler=NS(_remote_ra_live_v2_bridge=bridge), _remote_ra_private_v2_endpoint=NS(selection=selection),
                      _remote_ra_live_session_authority=NS(ids=['s']), _remote_ra_growing_adapter=NS(page_tokens=784),
                      _remote_ra_growing_owner=NS(identity=NS(epoch='e' * 64, manifest='m' * 64)))
            receipt = dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', copies_drained=True,
                           loaded_sources_verified=True, tensor_geometry=dict(page_tokens=784),
                           source_closure=dict(files=1), admission=value['admission'])
            read_text = pathlib.Path.read_text

            def proc(path, *args, **kwargs):
                name = str(path).replace('\\', '/')
                if name == '/proc/sys/kernel/random/boot_id':
                    return 'boot\n'
                if name == '/proc/self/stat':
                    return '1 (python) ' + ' '.join(['0'] * 30)
                return read_text(path, *args, **kwargs)
            with patch.object(EV, 'load', lambda: value), patch.object(EV.os, 'getpid', lambda: 4242), \
                    patch.object(pathlib.Path, 'read_text', proc), patch.object(EV.time, 'monotonic_ns', lambda: 7):
                report = EV.worker_attached(core, worker, bridge, life, receipt)
                authority = NS(identity=NS(epoch='e' * 64, manifest='m' * 64), policy='goal_query', session_id='s',
                               verify_admission=lambda: True, live_v2_readiness=report,
                               loaded=NS(manifest=dict(qualification_parent_manifest='p' * 64)))
                EV.FrontendEvidence(authority, None, dict(max_num_seqs=1)).write_ready(
                    NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})]), NS(started=True))
            return (json.loads((output / 'worker-attached.json').read_bytes()),
                    json.loads((output / 'ready.json').read_bytes()), value)

    def test_only_vllm_records_the_child_and_its_start(self):
        plain_attached, plain_ready, plain_config = self.files(False)
        attached, ready, config = self.files(True)
        self.assertNotIn(V.CONFIG_KEY, plain_attached)
        self.assertNotIn(V.CONFIG_KEY, plain_ready)
        self.assertEqual(attached.pop(V.CONFIG_KEY), {V.STARTUP_KEY: 812.25})
        self.assertEqual(attached, plain_attached)
        self.assertEqual(ready.pop(V.CONFIG_KEY), dict(config[V.CONFIG_KEY], startup_ms=812.25))
        self.assertEqual(ready, plain_ready)

    def test_ready_refuses_a_start_without_the_pin_and_the_pin_without_a_start(self):
        value = V.record('/m')
        self.assertIsNone(V.ready_record({}, {}))
        with self.assertRaisesRegex(LiveV2Error, 'not pinned'):
            V.ready_record({}, {V.CONFIG_KEY: {V.STARTUP_KEY: 1.0}})
        for attached in ({}, {V.CONFIG_KEY: {}}, {V.CONFIG_KEY: {V.STARTUP_KEY: -1.0}},
                         {V.CONFIG_KEY: {V.STARTUP_KEY: '1'}}, {V.CONFIG_KEY: {V.STARTUP_KEY: 1.0, 'x': 1}}):
            with self.subTest(attached=attached), self.assertRaisesRegex(LiveV2Error, 'did not record'):
                V.ready_record({V.CONFIG_KEY: value}, attached)
        inner = R.PersistentCommandGoalRewriter(('unused',))
        with self.assertRaisesRegex(LiveV2Error, 'did not start at startup'):
            V.attached(inner, value)
        transport = V.VllmRewriter(inner, ('unused',), value)
        inner._remote_ra_vllm = transport
        with self.assertRaisesRegex(LiveV2Error, 'did not start at startup'):
            V.attached(inner, value)                             # never installed: no start time


if __name__ == '__main__':
    unittest.main()

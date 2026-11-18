"""Local only (never delivered): controller.CHAT_SAMPLING_PROBE on a stub vLLM, with the sealed runtime's own
chat_text_route / chat_text_protocol (runtime_evidence.py runs it on the sealed clone).

The stub (written to a new directory) stands in for vLLM 0.24.0's serving pieces only as far as the probe calls them:
get_tokenizer (a character-count tokenizer whose chat template renders the thinking-off stream and refuses other
template kwargs), ChatCompletionRequest (the request fields the route reads and sets; to_sampling_params with the
generation defaults), the Qwen parsers (adjust_request unchanged), SamplingParams (clone, the generation-config and
tokenizer updates), and the msgpack round trip (pickle). It shows that the probe drives the release's parse,
effective_max_tokens, prepare_request and check_sampling together and that controller.check_sampling_record accepts
the result; vLLM's own conversions are checked by the gate on DeltaAI. `run_gemma`: controller.GEMMA_CHAT_SAMPLING_PROBE on
the same stub with Gemma's pieces (a character-count tokenizer whose template takes text parts and enable_thinking
false only, eos 1 / pad 0, the generation config's eos ids 1, 106, 50 added to the stops as vLLM's
update_from_generation_config adds them, the gemma4 parsers by name) and the release's model_profile stop rule.
"""
import json
from pathlib import Path
import subprocess
import sys

FILES = {
    'vllm/__init__.py': '',
    'vllm/tokenizers.py': '''
class Tok:
    eos_token_id, pad_token_id = 2, 3

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert kwargs == {'enable_thinking': False, 'preserve_thinking': True}, kwargs
        text = ''.join('<|im_start|>%s\\n%s<|im_end|>\\n' % (m['role'], m['content'].strip()) for m in messages)
        return text + ('<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' if add_generation_prompt else '')

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def get_tokenizer(model, trust_remote_code=False):
    return Tok()
''',
    'vllm/entrypoints/__init__.py': '', 'vllm/entrypoints/openai/__init__.py': '',
    'vllm/entrypoints/openai/chat_completion/__init__.py': '',
    'vllm/entrypoints/openai/chat_completion/protocol.py': '''
FIELDS = dict(model=None, messages=None, stream=False, n=1, temperature=None, top_p=None, top_k=None, min_p=None,
              max_tokens=None, max_completion_tokens=None, min_tokens=0, repetition_penalty=None, presence_penalty=0.0,
              frequency_penalty=0.0, stop=None, stop_token_ids=None, include_stop_str_in_output=False, ignore_eos=False,
              skip_special_tokens=True, logprobs=False, top_logprobs=None, logit_bias=None, bad_words=None,
              allowed_token_ids=None, response_format=None, tools=None, tool_choice=None, return_token_ids=False,
              use_beam_search=False, session_id=None)


class ChatCompletionRequest:
    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(FIELDS)
        assert not unknown, unknown
        for name, value in FIELDS.items():
            setattr(self, name, kwargs.get(name, value))

    def model_dump(self):
        return {name: getattr(self, name) for name in FIELDS}

    def to_sampling_params(self, max_tokens, defaults):
        from vllm.sampling_params import SamplingParams
        pick = lambda name, default: getattr(self, name) if getattr(self, name) is not None else defaults.get(name, default)
        return SamplingParams(n=self.n, temperature=pick('temperature', 1.0), top_p=pick('top_p', 1.0),
                              top_k=pick('top_k', -1), max_tokens=self.max_tokens if self.max_tokens is not None
                              else max_tokens, min_tokens=self.min_tokens, stop=list(self.stop or []),
                              stop_token_ids=list(self.stop_token_ids or []), ignore_eos=self.ignore_eos,
                              repetition_penalty=pick('repetition_penalty', 1.0), presence_penalty=self.presence_penalty,
                              frequency_penalty=self.frequency_penalty, skip_special_tokens=self.skip_special_tokens,
                              logit_bias=self.logit_bias, bad_words=self.bad_words,
                              allowed_token_ids=self.allowed_token_ids)
''',
    'vllm/sampling_params.py': '''
import copy


class SamplingParams:
    def __init__(self, **kwargs):
        self.eos_token_id, self.structured_outputs, self.logits_processors = None, None, None
        for name, value in kwargs.items():
            setattr(self, name, value)

    def clone(self):
        return copy.deepcopy(self)

    def update_from_generation_config(self, generation, eos_token_id):
        pass

    def update_from_tokenizer(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
''',
    'vllm/parser.py': '''
class ParserManager:
    @staticmethod
    def get_parser(**kwargs):
        class Parser:
            def __init__(self, tokenizer, tools, model_config=None, chat_template_kwargs=None):
                pass

            def adjust_request(self, request):
                return request
        return Parser
''',
    'vllm/v1/__init__.py': '',
    'vllm/v1/serial_utils.py': '''
import pickle


class MsgpackEncoder:
    def encode(self, value):
        return pickle.dumps(value)


class MsgpackDecoder:
    def __init__(self, cls):
        self.cls = cls

    def decode(self, raw):
        value = pickle.loads(raw)
        assert isinstance(value, self.cls)
        return value
''',
}


def run(python, source, work, pins_path=None):
    """Write the stub site and model, run the probe with the release source, check the record. Returns the record."""
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import controller as c
    work = Path(work)
    site, model = work / 'site', work / 'model'
    for name, text in FILES.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    model.mkdir(parents=True)
    (model / 'generation_config.json').write_text(json.dumps(dict(temperature=0.6, top_p=0.95, top_k=20)))
    pins = c.load(pins_path or (root / 'pins.json'))
    request = work / 'sampling-request.json'
    task = dict(instance_id='t', problem_statement='Fix the build in /app and run the tests.')
    request.write_text(json.dumps(c.sampling_requests(pins, task=task)))
    s = c.cfg(pins)['server']
    done = subprocess.run([python, '-I', '-B', '-c', c.CHAT_SAMPLING_PROBE, str(source), str(site), str(model),
                           str(request), str(s['max_tokens']), str(s['max_model_len'])],
                          capture_output=True, text=True, timeout=300)
    lines = [line for line in done.stdout.splitlines() if line.startswith('{')]
    value = json.loads(lines[-1]) if lines else dict(status='NO_OUTPUT', stderr=done.stderr[-3000:])
    c.check_sampling_record(value, s['max_tokens'], c.cfg(pins)['agent']['llm_kwargs']['max_tokens'])
    return value


GEMMA_TOKENIZER = '''
class Tok:
    eos_token_id, pad_token_id = 1, 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert kwargs == {'enable_thinking': False}, kwargs
        out = '<bos>'
        for m in messages:
            assert isinstance(m['content'], list) and m['content'][0]['type'] == 'text', m
            text = m['content'][0]['text'].strip()
            out += '<|turn>%s\\n%s%s<turn|>\\n' % ('model' if m['role'] == 'assistant' else m['role'], text,
                                                   ' ' if m['role'] == 'system' else '')
        return out + ('<|turn>model\\n<|channel>thought\\n<channel|>' if add_generation_prompt else '')

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def get_tokenizer(model, trust_remote_code=False):
    return Tok()
'''
GEMMA_SAMPLING = '''
import copy


class SamplingParams:
    def __init__(self, **kwargs):
        self.eos_token_id, self.structured_outputs, self.logits_processors = None, None, None
        for name, value in kwargs.items():
            setattr(self, name, value)

    def clone(self):
        return copy.deepcopy(self)

    def update_from_generation_config(self, generation, eos_token_id):
        ids = generation.get('eos_token_id')
        ids = {ids} if isinstance(ids, int) else set(ids or ())
        ids.discard(eos_token_id)
        if ids and not self.ignore_eos:
            self.stop_token_ids = sorted(ids | set(self.stop_token_ids or ()))

    def update_from_tokenizer(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
'''
GEMMA_PARSERS = '''
class ParserManager:
    @staticmethod
    def get_parser(**kwargs):
        assert kwargs['tool_parser_name'] == 'gemma4' and kwargs['reasoning_parser_name'] == 'gemma4', kwargs

        class Parser:
            def __init__(self, tokenizer, tools, model_config=None, chat_template_kwargs=None):
                pass

            def adjust_request(self, request):
                return request
        return Parser
'''


def run_gemma(python, source, work, pins_path=None):
    """controller.GEMMA_CHAT_SAMPLING_PROBE on the Gemma stub (the release source's chat_text_route,
    chat_text_protocol.GEMMA4_CHAT and model_profile). Returns the record (checked like the gate)."""
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import controller as c
    work = Path(work)
    site, model = work / 'site', work / 'model'
    files = dict(FILES)
    files.update({'vllm/tokenizers.py': GEMMA_TOKENIZER, 'vllm/sampling_params.py': GEMMA_SAMPLING,
                  'vllm/parser.py': GEMMA_PARSERS})
    for name, text in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    model.mkdir(parents=True)
    (model / 'generation_config.json').write_text(json.dumps(dict(bos_token_id=2, eos_token_id=[1, 106, 50],
                                                                  pad_token_id=0, temperature=1.0, top_k=64,
                                                                  top_p=0.95)))
    (model / 'config.json').write_text(json.dumps(dict(text_config=dict(pad_token_id=0, eos_token_id=1))))
    pins = c.load(pins_path or (root / 'pins.json'))
    request = work / 'sampling-request.json'
    task = dict(instance_id='t', problem_statement='Fix the build in /app and run the tests.')
    request.write_text(json.dumps(c.sampling_requests(pins, task=task)))
    s = c.cfg(pins)['server']
    done = subprocess.run([python, '-I', '-B', '-c', c.GEMMA_CHAT_SAMPLING_PROBE, str(source), str(site), str(model),
                           str(request), str(s['max_tokens']), str(s['max_model_len'])],
                          capture_output=True, text=True, timeout=300)
    lines = [line for line in done.stdout.splitlines() if line.startswith('{')]
    value = json.loads(lines[-1]) if lines else dict(status='NO_OUTPUT', stderr=done.stderr[-3000:])
    c.check_sampling_record(value, s['max_tokens'], c.cfg(pins)['agent']['llm_kwargs']['max_tokens'])
    c.need(sorted(value.get('stop') or ()) == list(c.GEMMA_STOP_IDS) and value.get('model_profile') == c.GEMMA_MODEL,
           'Gemma sampling stops %r' % (value.get('stop'),))
    return value

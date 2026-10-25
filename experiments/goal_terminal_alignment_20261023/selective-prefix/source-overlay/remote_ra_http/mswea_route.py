"""API-server half of the XML protocol: fixed generation contract and reply content.

The upstream server ignores client sampling fields: greedy argmax, 1536 new
tokens, stop on the action marker or EOS/pad.  Here the route fixes the same
values before vLLM builds its SamplingParams (EngineCore re-checks them), and
rebuilds the reply from the generated token IDs, as upstream does.  The reply
bound is the server's pinned ``max_tokens``, a server setting the live controller
holds (1536 unless serve.py --max-tokens 4096 pinned the other choice at startup).
"""
from remote_ra_live_v2.context_limits import MAX_TOKENS_CHOICES
from remote_ra_live_v2.mswea_protocol import MAX_TOKENS


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def prepare_request(request, stop_ids, max_tokens=MAX_TOKENS):
    require(not getattr(request, 'tools', None) and getattr(request, 'stream', False) is False
            and getattr(request, 'n', 1) == 1, 'XML protocol requires one nonstreaming reply without tools')
    require(type(max_tokens) is int and max_tokens in MAX_TOKENS_CHOICES, 'pinned reply bound required')
    fixed = dict(max_tokens=max_tokens, max_completion_tokens=None, temperature=0.0, top_p=1.0,
                 min_tokens=0, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                 stop=[], stop_token_ids=sorted(stop_ids), include_stop_str_in_output=False,
                 ignore_eos=False, skip_special_tokens=True, logprobs=False, top_logprobs=None,
                 logit_bias=None, bad_words=None, allowed_token_ids=None, response_format=None,
                 tool_choice='none', return_token_ids=True)
    for name, value in fixed.items():
        if hasattr(request, name):
            setattr(request, name, value)
    require(request.max_tokens == max_tokens and request.temperature == 0.0
            and request.return_token_ids is True
            and set(request.stop_token_ids) == set(stop_ids), 'XML protocol generation contract not applied')
    return request


def finish_response(result, call, transcript):
    """Upstream reply: decode(all generated IDs, skip_special_tokens=True), finish 'stop'."""
    choices = getattr(result, 'choices', None)
    require(type(choices) is list and len(choices) == 1, 'one completion choice required')
    choice = choices[0]
    generated = choice.token_ids
    require(type(generated) is list and generated and all(type(t) is int and t >= 0 for t in generated),
            'generated token IDs missing from the native response')
    committed = transcript.commit(call.capture.tokens, generated)
    message = choice.message
    message.content = committed['text']
    message.tool_calls = []
    for name in ('reasoning', 'reasoning_content'):
        if hasattr(message, name):
            setattr(message, name, None)
    choice.finish_reason, choice.stop_reason, choice.token_ids = 'stop', None, None
    if hasattr(result, 'prompt_token_ids'):
        result.prompt_token_ids = None
    return committed

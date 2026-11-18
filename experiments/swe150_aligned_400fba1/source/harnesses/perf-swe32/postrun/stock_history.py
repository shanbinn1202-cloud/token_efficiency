"""The stock vLLM 0.24.0 control's chat history, rendered offline (2026-11-16: the stock keep-reasoning plans).

What a stock vLLM 0.24.0 server renders for one chat request of Mini-SWE-Agent 2.4.2 (litellm 1.103.1, text-based
model class), without vLLM installed: the request body's messages as vLLM's own parse sees them, then the pinned Qwen3.6
template (fixtures/qwen3.6-27b-chat_template.jinja, e84f32a2...) through the tokenizer's apply_chat_template, then the
prompt's token ids (add_special_tokens false, the chat route's default).

- The request body (captured on tbench, 2026-11-16, D:/Remote-RA-temp/scratch/stockkeep-1116/tb/capture_body.py): an
  assistant message carries content, reasoning_content (the earlier reply's reasoning, from vLLM's `reasoning` through
  litellm), provider_specific_fields ({"reasoning": ...}) and role; tool_calls / function_call (None) are left out.
  With the keep-reasoning model class (tbench/agent_ext/swe32_keep_reasoning.py) it also carries `reasoning`, a copy of
  reasoning_content.
- vLLM 0.24.0's parse (vllm/entrypoints/chat_utils.py _parse_chat_message_content, ~l.1743-1790): the message's
  content as text parts, and for an assistant message `reasoning = message.get("reasoning")`; only when it is not None
  the parsed message gets `reasoning` and `reasoning_content` (both that value). reasoning_content alone is dropped,
  provider_specific_fields is never read. One text part renders as its string in either content format.
- The template kwargs: the request's chat_template_kwargs (the agents' {"enable_thinking": true}) over the server's
  --default-chat-template-kwargs (vllm/renderers/params.py merge_kwargs: the request wins unless its value is None or
  "auto"); the keep plans' server default is {"preserve_thinking": true}. The template (l.100) renders an earlier reply
  as `<|im_start|>assistant\\n<think>\\n` + reasoning|trim + `\\n</think>\\n\\n` + content|trim when preserve_thinking
  is true (or the reply follows the last real user query: never here, every observation is a user message), else as
  `<|im_start|>assistant\\n` + content|trim.

render_ids() is the whole path; history_of() names which of the two histories a recorded prompt length matches.
"""
import hashlib
import json
from pathlib import Path

KEEP_KWARGS = {'preserve_thinking': True}  # the keep plans' --default-chat-template-kwargs
# the pinned Qwen3.6 template (fixtures/qwen3.6-27b-chat_template.jinja = the pinned tokenizer_config.json's own)
TEMPLATE_SHA256 = 'e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259'
AGENT_KWARGS = {'enable_thinking': True}   # config.stock.thinking.stock_model_kwargs.extra_body.chat_template_kwargs


def request_messages(messages, send_reasoning):
    """The request body's messages for one call, from the trajectory's messages before it (Mini-SWE-Agent's
    _prepare_messages_for_api: every key but `extra`; litellm drops the None tool_calls / function_call). With
    send_reasoning (the keep model class) an assistant message with a string reasoning_content also carries
    `reasoning`, its copy."""
    out = []
    for m in messages:
        body = {k: v for k, v in m.items() if k != 'extra' and v is not None}
        if send_reasoning and body.get('role') == 'assistant' and isinstance(body.get('reasoning_content'), str):
            body['reasoning'] = body['reasoning_content']
        out.append(body)
    return out


def vllm_conversation(body_messages):
    """vLLM 0.24.0's parsed conversation of text-only messages (chat_utils._parse_chat_message_content)."""
    conversation = []
    for m in body_messages:
        content = m.get('content')
        if content is None:
            text = ''
        elif isinstance(content, str):
            text = content
        else:
            text = '\n'.join(part['text'] for part in content if part.get('type') == 'text')
        parsed = dict(role=m['role'], content=text)
        if m['role'] == 'assistant':
            reasoning = m.get('reasoning')
            if reasoning is not None:
                parsed['reasoning'] = reasoning
                parsed['reasoning_content'] = reasoning
        conversation.append(parsed)
    return conversation


def template_kwargs(server_default=None, request=None):
    """vllm/renderers/params.py merge_kwargs(defaults, overrides)."""
    return dict(server_default or {}, **{k: v for k, v in (request or {}).items() if v not in (None, 'auto')})


def render_text(tok, template, conversation, kwargs):
    return tok.apply_chat_template(conversation, chat_template=template, tokenize=False, add_generation_prompt=True,
                                   **kwargs)


def render_ids(tok, template, messages, *, send_reasoning, server_default, request_kwargs=None):
    """The prompt token ids a stock vLLM 0.24.0 server renders for one call (messages = the trajectory's messages
    before the call)."""
    conversation = vllm_conversation(request_messages(messages, send_reasoning))
    text = render_text(tok, template, conversation, template_kwargs(server_default, request_kwargs or AGENT_KWARGS))
    return tok.encode(text, add_special_tokens=False)


def history_of(prompt_tokens, kept, dropped):
    """'reasoning_kept' / 'reasoning_dropped' / 'same' (no earlier reasoning to tell them apart) / 'neither'."""
    if kept == dropped:
        return 'same' if prompt_tokens == kept else 'neither'
    return 'reasoning_kept' if prompt_tokens == kept else 'reasoning_dropped' if prompt_tokens == dropped else 'neither'


def agent_calls(trajectory):
    """(index, usage, kind) of every model call of a Mini-SWE-Agent 2.4.2 trajectory in order (join_swe32.agent_calls'
    rule): an assistant message, or a user message whose extra names interrupt_type FormatError. The call's request is
    messages[:index]."""
    calls = []
    for index, m in enumerate(trajectory.get('messages') or []):
        extra = m.get('extra') or {}
        if m.get('role') == 'assistant':
            kind = 'assistant'
        elif m.get('role') == 'user' and extra.get('interrupt_type') == 'FormatError':
            kind = 'format_error'
        else:
            continue
        response = extra.get('response')
        usage = (response.get('usage') if isinstance(response, dict) else None) or {}
        calls.append(dict(index=index, kind=kind, usage=usage,
                          response_id=response.get('id') if isinstance(response, dict) else None))
    return calls


def load_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)



def append_growth(calls):
    """prompt[k+1] - prompt[k] - completion[k] over the calls that follow a reply (kind 'assistant': after a format
    error the failed reply is not in the next request). dict(pairs, negative, min)."""
    values = [b['prompt_tokens'] - a['prompt_tokens'] - a['completion_tokens'] for a, b in zip(calls, calls[1:])
              if a['kind'] == 'assistant' and all(isinstance(x, int) for x in (
                  a['prompt_tokens'], a['completion_tokens'], b['prompt_tokens']))]
    return dict(pairs=len(values), negative=sum(1 for v in values if v < 0), min=min(values) if values else None)

def pinned_template(tokenizer_dir):
    """The chat template of the pinned tokenizer files (tokenizer_config.json), refused unless it is e84f32a2."""
    template = load_json(Path(tokenizer_dir) / 'tokenizer_config.json').get('chat_template')
    if not isinstance(template, str) or hashlib.sha256(template.encode('utf-8')).hexdigest() != TEMPLATE_SHA256:
        raise ValueError('the tokenizer config does not carry the pinned Qwen3.6 template e84f32a2')
    return template


def check_instance(tok, template, messages, calls):
    """stock-history-kept of one instance (the join, 2026-11-16): every call's usage.prompt_tokens against the
    render with the reasoning kept (the keep plans' request + preserve_thinking) and with it dropped (the request
    without `reasoning`, no server default). Returns dict(calls, kept_equal, dropped_equal, distinguishable,
    mismatch: the first call equal to neither or to the dropped render only)."""
    out = dict(calls=len(calls), kept_equal=0, dropped_equal=0, distinguishable=0, mismatch=None)
    for k, call in enumerate(calls):
        history = messages[:call['index']]
        kept = len(render_ids(tok, template, history, send_reasoning=True, server_default=KEEP_KWARGS))
        dropped = len(render_ids(tok, template, history, send_reasoning=False, server_default=None))
        served = call['prompt_tokens']
        out['kept_equal'] += served == kept
        out['dropped_equal'] += served == dropped
        out['distinguishable'] += kept != dropped
        if served != kept and out['mismatch'] is None:
            out['mismatch'] = dict(call=k, kind=call.get('kind'), served=served, kept=kept, dropped=dropped,
                                   history=history_of(served, kept, dropped))
    return out

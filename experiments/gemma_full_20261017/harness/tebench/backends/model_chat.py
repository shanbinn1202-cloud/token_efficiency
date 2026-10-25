"""Native model chat delimiters shared by stateless and KV transports."""


def is_gemma4(tokenizer):
    value = getattr(tokenizer, "_tebench_gemma4", None)
    if value is None:
        value = (hasattr(tokenizer, "get_vocab")
                 and "<|tool_call>" in tokenizer.get_vocab())
        tokenizer._tebench_gemma4 = value
    return value


def eos_ids(tokenizer):
    tokens = ("<eos>", "<turn|>", "<|tool_response>") if is_gemma4(tokenizer) else ("<|im_end|>",)
    return sorted({tokenizer.eos_token_id,
                   *(tokenizer.convert_tokens_to_ids(t) for t in tokens)} - {None})


def template_options(tokenizer, thinking, extra=None):
    options = {"enable_thinking": thinking}
    if is_gemma4(tokenizer):
        options["preserve_thinking"] = True
    options.update(extra or {})
    return options


def parse_gemma4(text, tokenizer, tools, prompt_ids, thinking=True):
    """Use vLLM's native parser, including a reasoning header in the prompt."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.gemma4 import Gemma4Parser

    request = ChatCompletionRequest(model="gemma4", messages=[], tools=tools or None,
                                    tool_choice="auto" if tools else "none")
    parser = Gemma4Parser(tokenizer, tools=request.tools,
                         chat_template_kwargs={"enable_thinking": thinking})
    text = text.rstrip()
    for marker in ("<eos>", "<turn|>", "<|tool_response>"):
        text = text.removesuffix(marker).rstrip()
    # parse() resets its state; reconstruct only the open header for parsing.
    # The model input and generated-token accounting are unchanged.
    vocab = tokenizer.get_vocab()
    start_id = vocab.get("<|channel>")
    boundary_ids = {vocab.get(t) for t in ("<channel|>", "<|tool_call>", "<|turn>", "<|tool_response>")} - {None}
    open_reasoning = False
    for token in reversed(prompt_ids or []):
        if token == start_id:
            open_reasoning = True
            break
        if token in boundary_ids:
            break
    if open_reasoning:
        text = "<|channel>thought\n" + text
    reasoning, content, calls = parser.parse(text, request, enable_auto_tools=True)
    return {"role": "assistant", "content": content or "",
            "reasoning_content": reasoning or "", "tool_calls": [
                {"id": call.id, "type": "function", "function": {
                    "name": call.name, "arguments": call.arguments}}
                for call in calls or []] or None}

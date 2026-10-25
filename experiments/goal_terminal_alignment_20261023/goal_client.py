"""Session transport for the project's vLLM KV-level sliding window."""
import copy
from pathlib import Path
import time
import uuid

import httpx
from tebench.backends.base import Backend, normalize_messages, response
from tebench.backends.model_chat import eos_ids, is_gemma4, template_options


class GoalSessionBackend(Backend):
    method = "dram_backed_kv"

    def __init__(self, config, method="dram_backed_kv"):
        self.method = method
        super().__init__(config)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"],
            local_files_only=config["model"]["local_files_only"])
        self.client = httpx.Client(base_url=config["backend"]["endpoint"].rstrip("/") + "/",
                                  timeout=config["generation"]["request_timeout"])
        info = self.client.get("kv-session/info")
        info.raise_for_status()
        self.execution_info = info.json()
        if self.execution_info.get("method") != method:
            raise RuntimeError("Server KV policy does not match requested method")
        expected = config['goal_runtime']
        if self.execution_info.get('budget') != config['cache']['kv_budget']:
            raise RuntimeError('Goal token budget differs from client')
        options = self.execution_info.get('options', {})
        required = {'policy':'goal_query','thinking':'on','max_model_len':262144,
                    'max_tokens':4096,'max_running_sessions':16}
        if any(options.get(k) != v for k,v in required.items()):
            raise RuntimeError('Goal runtime configuration differs: '+repr(options))
        from hashlib import sha256
        manifest = Path(expected['release'])/'runtime-manifest.json'
        if sha256(manifest.read_bytes()).hexdigest() != expected['manifest_sha256']:
            raise RuntimeError('Goal release manifest changed')

    def complete(self, *, model, messages, tools=None, **options):
        started = time.perf_counter()
        normalized = normalize_messages(messages)
        state = getattr(self._local, "session", None)
        template = template_options(self.tokenizer, self.config["model"]["enable_thinking"],
                    (options.get("extra_body") or {}).get("chat_template_kwargs"))
        def render(msgs, generation):
            return self.tokenizer.apply_chat_template(msgs, tools=tools or None,
                tokenize=False, add_generation_prompt=generation, **template)
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        boundary_tokens = 0
        # Obtain the exact template header, without hard-coding thinking tokens.
        if state is None:
            prefix_text = render(normalized, False)
            text = render(normalized, True)
            if not text.startswith(prefix_text):
                raise ValueError("Chat template does not expose a stable generation header")
            header = encode(text[len(prefix_text):])
            ids = encode(text)
            prefix = len(ids) - len(header)
            state = self._local.session = dict(id=uuid.uuid4().hex, messages=[], last=[], prefix=prefix)
        else:
            previous = state["messages"]
            if normalized[:len(previous)] != previous:
                raise ValueError("KV session requires append-only message history")
            new = normalized[len(previous):]
            if not new or any(m["role"] == "assistant" for m in new[1:]):
                raise ValueError("Expected one assistant turn followed by tool/user feedback")
            # mini-SWE-agent omits the assistant message on FormatError, but
            # that response already exists in the persistent engine session.
            if new[0]["role"] != "assistant":
                rendered_messages = normalized[:len(previous)] + [state["last_message"]] + new
            else:
                rendered_messages = normalized
            # Preserve generated raw tokens, including thinking and tool syntax.
            # Only render the new closing delimiter, feedback, and next header.
            marker = "TE_KV_BOUNDARY_" + uuid.uuid4().hex
            marked = copy.deepcopy(rendered_messages)
            feedback = len(previous) + 1
            if (is_gemma4(self.tokenizer) and feedback < len(marked)
                    and marked[feedback]["role"] == "tool"):
                # Gemma renders tool responses inside the assistant's tool-call
                # branch. Keep those calls and mark the first new response.
                marked[feedback]["content"] = marker
                probe = render(marked, True)
                if probe.count(marker) != 1:
                    raise ValueError("Unsupported Gemma tool continuation")
                start = probe.rfind("<|tool_response>", 0, probe.index(marker))
                with_header = render(rendered_messages, True)
                without_header = render(rendered_messages, False)
                if start < 0 or with_header[:start] != probe[:start]:
                    raise ValueError("Unstable Gemma tool response boundary")
                tail = with_header[start:]
            else:
                marked[len(previous)] = {"role": "assistant", "content": marker}
                with_header, without_header = render(marked, True), render(marked, False)
                if with_header.count(marker) != 1:
                    raise ValueError("Unsupported template continuation")
                tail = with_header.split(marker, 1)[1]
            if not with_header.startswith(without_header):
                raise ValueError("Unsupported template generation header")
            header = encode(with_header[len(without_header):])
            ids = encode(tail)
            if self.method == "dram_backed_kv":
                if is_gemma4(self.tokenizer):
                    if tail.startswith('<|tool_response>'):
                        boundary_tokens = len(encode('<|tool_response>'))
                    elif '<|turn>' in tail:
                        boundary_tokens = len(encode(tail.split('<|turn>',1)[0]))
                    else:
                        raise ValueError('Gemma Goal needs a tool or turn boundary')
                else:
                    if "<|im_start|>" not in tail:
                        raise ValueError("DRAM requires an explicit observation message boundary")
                    boundary_tokens = len(encode(tail.split("<|im_start|>", 1)[0]))
            # Streaming drops its final uncomputed sampled token. The template
            # supplies im_end for EOS; restore a non-EOS final token explicitly.
            last = state["last"][-1:]
            eos = eos_ids(self.tokenizer)
            if last and (not ids or ids[0] != last[0]):
                ids = last + ids
                boundary_tokens += 1
            prefix = state["prefix"]
        g = self.config["generation"]
        body = dict(method=self.method, session_id=state["id"], input_ids=ids, header_tokens=len(header),
                    prefix_tokens=prefix, budget=self.config["cache"]["kv_budget"],
                    max_tokens=options.get("max_tokens", g["max_tokens"]),
                    temperature=options.get("temperature", g["temperature"]),
                    top_p=g["top_p"], top_k=g["top_k"], repetition_penalty=g["repetition_penalty"], seed=g["seed"])
        if self.method == "dram_backed_kv":
            body.update(messages=normalized, boundary_tokens=boundary_tokens)
        result = self.client.post("kv-session/complete", json=body)
        if result.is_error:
            if result.status_code == 400 and result.json().get("error") == "Logical RoPE context limit reached":
                from minisweagent.exceptions import InterruptAgentFlow
                # The server checks this before scheduling any model work.
                # Treat the configured context limit like the agent turn limit,
                # allowing the runner's existing terminal patch recovery.
                raise InterruptAgentFlow({"role": "exit", "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": "",
                              "reason": "Logical RoPE context limit reached",
                              "max_window_size": g["max_window_size"]}})
            raise RuntimeError(f"KV window server: {result.text}")
        data = result.json()
        state.update(messages=copy.deepcopy(normalized), last=data.pop("output_ids"))
        text = data.pop("text")
        self.last_measurement = dict(seconds=time.perf_counter() - started,
                                    admission_wait_seconds=0.0, **data)
        result = response(model, text, data["prompt_tokens"], data["completion_tokens"], tools,
                          cached_tokens=data["cached_tokens"], finish_reason=data["finish_reason"],
                          tokenizer=self.tokenizer, prompt_ids=ids,
                          enable_thinking=template["enable_thinking"])
        state["last_message"] = result.choices[0].message.model_dump(exclude_none=True)
        return result

    def reset(self):
        state = getattr(self._local, "session", None)
        if state:
            result = self.client.post("kv-session/close", json={"session_id": state["id"]})
            result.raise_for_status()
            self._local.session = None
        super().reset()

    def close(self):
        self.reset()
        self.client.close()


KVWindowBackend = GoalSessionBackend  # Existing sliding-window API.

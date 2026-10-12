"""Engine-side normalized-source rendering, with no synthetic runtime fallback.

Server-owned objects and approved source digests are prerequisites. Python object
access is trusted; this is drift detection, not a sandbox against hostile Python.
No import hooks, package installation, HTTP route, or deployment is performed.
"""
import ast
from hashlib import sha256
import json
import marshal
from pathlib import Path
from types import CodeType, FunctionType, MethodType
from dynamic_runtime import RenderedPrompt, digest, require
from qwen_observation_span import (QwenObservationSpanProducer, _load_pinned_artifacts,
    validate_tokenization_options, validate_identity_normalization)
from trusted_renderer_adapter import BoundCapture, bound_observation_geometry
from tokenizer_semantics import TokenizerSemanticGuard


SOURCE_SCHEMA = "remote-ra-normalized-qwen-source-v1"
PLAIN_GOAL_SCHEMA = "remote-ra-plain-goal-probe-v1"
XML_SOURCE_SCHEMA = "remote-ra-mswea-xml-source-v1"
MAX_SOURCE_BYTES = 1024 * 1024
TOKEN_OPTIONS = dict(truncate_prompt_tokens=None, pad_prompt_tokens=None,
                     truncation_side=None, do_lower_case=False, add_special_tokens=False)


def _snapshot(value):
    # Argument insertion order is executable template input; never sort it.
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    require(len(raw.encode("utf-8")) <= MAX_SOURCE_BYTES, "normalized source exceeds limit")
    return json.loads(raw)


def _ordered_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def source_from_bound_capture(bound):
    """Trusted frontend resolver helper; never accepts client-provided spans."""
    require(type(bound) is BoundCapture, "same-call bound capture required")
    bound_observation_geometry(bound)
    capture = bound.capture
    return validate_source(dict(schema=SOURCE_SCHEMA,
        conversation=json.loads(capture.conversation_json),
        template_kwargs=json.loads(capture.template_kwargs_json),
        tokenization_options=json.loads(capture.tokenization_options_json)))


def validate_source(source):
    require(type(source) is dict and set(source) == {
        "schema", "conversation", "template_kwargs", "tokenization_options"}
        and source["schema"] == SOURCE_SCHEMA, "exact normalized source schema required")
    source = _snapshot(source)
    validate_tokenization_options(source["tokenization_options"])
    conversation, kwargs = source["conversation"], source["template_kwargs"]
    require(type(conversation) is list and 0 < len(conversation) <= 4096,
            "bounded normalized conversation required")
    for message in conversation:
        require(type(message) is dict and message.get("role") in {
            "system", "user", "assistant", "tool"} and type(message.get("content")) is str,
            "normalized plain-text messages required")
        require(set(message) <= {"role", "content", "name", "tool_calls", "tool_call_id", "reasoning"},
                "unsupported normalized message field")
        for field in ("name", "tool_call_id", "reasoning"):
            require(field not in message or type(message[field]) is str,
                    "normalized optional fields must be strings")
        if "tool_calls" in message:
            require(message["role"] == "assistant" and type(message["tool_calls"]) is list,
                    "normalized assistant tool_calls required")
            for call in message["tool_calls"]:
                require(type(call) is dict and set(call) == {"id", "type", "function"}
                        and type(call["id"]) is str and call["type"] == "function",
                        "unsupported normalized tool call")
                fn = call["function"]
                require(type(fn) is dict and set(fn) == {"name", "arguments"}
                        and type(fn["name"]) is str and type(fn["arguments"]) is dict,
                        "parsed tool arguments required")
    require(type(kwargs) is dict and set(kwargs) <= {
        "tools", "add_generation_prompt", "enable_thinking", "preserve_thinking"}
        and kwargs.get("add_generation_prompt") is True, "unsupported template options")
    for name in ("enable_thinking", "preserve_thinking"):
        require(name not in kwargs or type(kwargs[name]) is bool, "boolean template option required")
    require("tools" not in kwargs or kwargs["tools"] is None or type(kwargs["tools"]) is list,
            "tools must be a normalized list or None")
    return source


def validate_xml_source(source):
    """XML protocol source: role/content text messages only; tokens come from the transcript."""
    require(type(source) is dict and set(source) == {"schema", "messages"}
            and source["schema"] == XML_SOURCE_SCHEMA, "exact XML protocol source schema required")
    source = _snapshot(source)
    messages = source["messages"]
    require(type(messages) is list and 2 <= len(messages) <= 4096
            and all(type(m) is dict and set(m) == {"role", "content"}
                    and m["role"] in {"system", "user", "assistant"} and type(m["content"]) is str
                    for m in messages), "XML protocol messages must be role/content text")
    return source


def _code_digest(code):
    # Compile the reviewed bytes on this interpreter. Filenames/line tables do
    # not affect execution and differ between copied and installed paths.
    def normalize(item):
        if not isinstance(item, CodeType):
            return item
        return item.replace(co_consts=tuple(normalize(c) for c in item.co_consts),
                            co_filename="<reviewed>", co_firstlineno=1, co_linetable=b"")
    return sha256(marshal.dumps(normalize(code))).hexdigest()


def _implementation_line(source_tree, qualname):
    """Select the runtime definition after typing overload declarations.

    Resolve explicit lexical scopes only. Duplicate executable definitions and
    unsupported conditional/nested bindings stay fail-closed. Overload stubs
    must precede the single implementation and use an imported typing decorator.
    """
    overload_names, typing_names = set(), set()
    for node in source_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in ("typing", "typing_extensions"):
            overload_names.update(alias.asname or alias.name for alias in node.names if alias.name == "overload")
        elif isinstance(node, ast.Import):
            typing_names.update(alias.asname or alias.name for alias in node.names
                                if alias.name in ("typing", "typing_extensions"))
    def is_overload(node):
        return any((isinstance(d, ast.Name) and d.id in overload_names) or
                   (isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name)
                    and d.value.id in typing_names and d.attr == "overload")
                   for d in node.decorator_list)
    body = source_tree.body
    parts = qualname.split(".")
    for part in parts[:-1]:
        owners = [n for n in body if isinstance(n, ast.ClassDef) and n.name == part]
        require(len(owners) == 1, "reviewed callable scope missing or ambiguous")
        body = owners[0].body
    definitions = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == parts[-1]]
    implementations = [n for n in definitions if not is_overload(n)]
    require(len(implementations) == 1, "reviewed callable implementation missing or ambiguous")
    implementation = implementations[0]
    for stub in definitions:
        if stub is not implementation:
            require(stub.lineno < implementation.lineno and len(stub.body) == 1
                    and isinstance(stub.body[0], ast.Expr)
                    and isinstance(stub.body[0].value, ast.Constant)
                    and stub.body[0].value.value is Ellipsis,
                    "unsupported overload declaration")
    return min([implementation.lineno] + [d.lineno for d in implementation.decorator_list])


def _code_at(module_code, qualname, source_tree):
    implementation_line = _implementation_line(source_tree, qualname)
    matches = []
    def visit(code):
        if code.co_qualname == qualname and code.co_firstlineno == implementation_line:
            matches.append(code)
        for child in code.co_consts:
            if isinstance(child, CodeType):
                visit(child)
    visit(module_code)
    require(len(matches) == 1, "reviewed callable qualname missing or ambiguous")
    return matches[0]


class LoadedSourceIdentity:
    """Check explicit loaded callables against approved bytes on every admission.

    Each binding is (owner object, attribute name, reviewed file, SHA256).
    Owners must be actual loaded module/class/instance objects, not snapshots.
    All transitive rendering/parser/tokenizer helpers must be covered by the
    deployment's reviewed binding inventory. This class cannot infer that set.
    """
    def __init__(self, bindings):
        require(type(bindings) in (list, tuple) and bindings, "loaded source bindings required")
        self._bindings = []
        for owner, attribute, location, approved_sha256 in bindings:
            path = Path(location).resolve(strict=True)
            blob = path.read_bytes()
            require(sha256(blob).hexdigest() == approved_sha256, "approved source hash mismatch")
            target = getattr(owner, attribute)
            fn = target.__func__ if isinstance(target, MethodType) else target
            require(type(fn) is FunctionType and fn.__name__ == attribute
                    and Path(fn.__code__.co_filename).resolve() == path,
                    "callable is not loaded from reviewed source path")
            reviewed = _code_at(compile(blob, str(path), "exec", dont_inherit=True),
                                fn.__qualname__, ast.parse(blob, filename=str(path)))
            require(_code_digest(fn.__code__) == _code_digest(reviewed),
                    "loaded callable code differs from reviewed source")
            self._bindings.append((owner, attribute, path, approved_sha256, fn, _code_digest(reviewed)))
        self.identity = digest([dict(path=str(p), attribute=a, sha256=h, code=c)
            for _, a, p, h, _, c in self._bindings])

    def __call__(self):
        for owner, attribute, path, expected, fn, code in self._bindings:
            target = getattr(owner, attribute, None)
            current = target.__func__ if isinstance(target, MethodType) else target
            require(current is fn and _code_digest(fn.__code__) == code,
                    "loaded source callable changed")
            require(sha256(path.read_bytes()).hexdigest() == expected, "loaded source file changed")
        return True


class EngineSourceRenderer:
    """Re-render trusted normalized source with actual HF/vLLM server objects.

    `render_template` must call the loaded server's safe_apply_chat_template and
    `verify_identity` must cover that callback's parser/HF dependencies, loaded
    model config/revision, and topology. A copied-file audit alone is insufficient.
    """
    def __init__(self, *, evidence_dir, tokenizer, render_template, verify_identity, page_tokens):
        require(callable(render_template) and callable(verify_identity), "trusted loaded renderer required")
        require(type(page_tokens) is int and page_tokens > 0, "positive page size required")
        self.tokenizer, self._render_template = tokenizer, render_template
        self.verify_identity, self.page_tokens = verify_identity, page_tokens
        self.template, raw = _load_pinned_artifacts(evidence_dir)
        self._semantic = TokenizerSemanticGuard(evidence_dir, tokenizer)
        self._check_identity()
        self.producer = QwenObservationSpanProducer(evidence_dir, render_template=render_template)
        # Same pinned base plus exact config-derived added tokens.
        self.producer._tokenizer = self._semantic.reference

    def _check_identity(self):
        require(self.verify_identity() is True, "loaded source/model identity unverified")
        require(getattr(self.tokenizer, "is_fast", None) is True,
                "actual fast HF tokenizer required")
        require(getattr(self.tokenizer, "chat_template", None) == self.template,
                "loaded tokenizer template differs from pinned template")
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        require(backend is not None and callable(getattr(backend, "to_str", None)),
                "loaded tokenizer backend unavailable")
        require(self._semantic.verify() is True, "pinned tokenizer semantic contract changed")

    def encode_exact(self, text):
        """Upstream ``_encode``: no added special tokens; special-token text is parsed."""
        require(type(text) is str and len(text.encode("utf-8")) <= MAX_SOURCE_BYTES,
                "bounded protocol text required")
        # The tokenizer's semantics were checked on the pinned corpus at startup. Per message
        # only the two complete encodings must agree: the corpus-style comparison (four
        # tokenizers, encode/decode round trips) took ~120 ms per 6.5K-token observation,
        # once in the frontend and once in the engine (v10d spans).
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        raw = self.producer._tokenizer.encode(text, add_special_tokens=False)
        require(type(tokens) is list and all(type(t) is int and t >= 0 for t in tokens)
                and raw.ids == tokens and not raw.overflowing,
                "loaded HF tokenization differs from pinned complete encoding")
        return tokens

    def render_initial(self, messages):
        """Upstream ``apply_chat_template(messages[:2], enable_thinking=False)``, no tools."""
        require(type(messages) is list and len(messages) == 2
                and [m.get("role") for m in messages] == ["system", "user"],
                "system and task messages required")
        text = self._render_template(self.template, messages,
                                     {"add_generation_prompt": False, "enable_thinking": False})
        require(type(text) is str and text, "real template returned no text")
        return self.encode_exact(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def stop_ids(self):
        """Upstream stop set: tokenizer EOS and pad."""
        ids = frozenset(t for t in (getattr(self.tokenizer, "eos_token_id", None),
                                    getattr(self.tokenizer, "pad_token_id", None)) if t is not None)
        require(ids, "tokenizer EOS/pad ids required")
        return ids

    def transcript(self):
        # Stage B reproduces upstream session rebuilds (a fresh cold session
        # whenever a request is not an append-only extension).
        from remote_ra_live_v2.mswea_protocol import Transcript
        return Transcript(encode=self.encode_exact, render_initial=self.render_initial,
                          decode=self.decode, stop_ids=self.stop_ids(), allow_rebuild=True)

    def __call__(self, source):
        self._check_identity()
        if type(source) is dict and source.get("schema") == PLAIN_GOAL_SCHEMA:
            require(set(source) == {"schema", "text"} and type(source["text"]) is str
                    and source["text"].startswith("<retrieval_query>\n")
                    and source["text"].endswith("\n</retrieval_query>"),
                    "exact upstream retrieval_query Goal probe source required")
            text = source["text"]
            require(0 < len(text.encode("utf-8")) <= MAX_SOURCE_BYTES, "bounded Goal probe required")
            validate_identity_normalization(text=text, normalizer=self.tokenizer.backend_tokenizer.normalizer)
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            raw = self.producer._tokenizer.encode(text, add_special_tokens=False)
            require(type(tokens) is list and tokens and len(tokens) <= 131072
                    and all(type(t) is int and t >= 0 for t in tokens)
                    and raw.ids == tokens and not raw.overflowing,
                    "plain Goal tokenization differs from pinned complete encoding")
            self._check_identity()
            return RenderedPrompt(tuple(tokens))
        source = validate_source(source)
        before = _ordered_json(source)
        text = self._render_template(self.template, source["conversation"], source["template_kwargs"])
        require(type(text) is str and text, "real template returned no text")
        require(_ordered_json(source) == before, "renderer mutated normalized source")
        validate_identity_normalization(text=text, normalizer=self.tokenizer.backend_tokenizer.normalizer)
        self._semantic.check_text(text)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        require(type(tokens) is list and tokens and len(tokens) <= 131072
                and all(type(t) is int and t >= 0 for t in tokens), "invalid loaded tokenizer output")
        raw = self.producer._tokenizer.encode(text, add_special_tokens=False)
        require(raw.ids == tokens and not raw.overflowing,
                "loaded HF tokenization differs from pinned complete encoding")
        if source["conversation"][-1]["role"] == "tool":
            rendered = self.producer.produce(conversation=source["conversation"],
                template_kwargs=source["template_kwargs"], final_rendered_text=text,
                final_token_ids=tokens, page_tokens=self.page_tokens,
                tokenization_options=source["tokenization_options"]).rendered
        else:
            rendered = RenderedPrompt(tuple(tokens))
        self._check_identity()
        return rendered


    def render_growing(self, source, native_request, authority, *, session, turn, local_turn,
                       generation, model, renderer, native_prefix, transcript=None):
        # This method runs on the actual engine-owned renderer with its existing
        # loaded-source, tokenizer and Qwen template identity checks.
        if type(source) is dict and source.get("schema") == XML_SOURCE_SCHEMA:
            # XML protocol: the engine's own transcript (generated token IDs it
            # produced) re-derives the prompt; the frontend tokens must match.
            # turn is the global request ordinal; local_turn is the session's.
            require(transcript is not None and transcript.turn == local_turn,
                    "engine XML transcript missing or at another local turn")
            messages = validate_xml_source(source)["messages"]
            self._check_identity()
            tokens, stable_end, observation = transcript.render(messages)
            try:
                rebuild = transcript.pending["rebuild"]
                self._check_identity()
                return authority.render(native_request, session=session, turn=turn,
                    local_turn=0 if rebuild else local_turn, rebuild=rebuild,
                    generation=generation, model=model, renderer=renderer, tokens=tokens,
                    stable_prefix_tokens=tokens[:stable_end], observation_span=observation,
                    native_prefix=native_prefix)
            except BaseException:
                transcript.abandon()
                raise
        require(transcript is None, "transcript applies to the XML protocol only")
        source = validate_source(source)
        rendered = self(source)
        tokens = tuple(rendered.token_ids)
        if turn == 0:
            require(rendered.observation_start is None,
                    "turn-zero growing render cannot contain terminal tools")
            before = _ordered_json(source)
            kwargs = dict(source["template_kwargs"])
            kwargs["add_generation_prompt"] = False
            stable_text = self._render_template(self.template, source["conversation"], kwargs)
            final_text = self._render_template(self.template, source["conversation"], source["template_kwargs"])
            require(type(stable_text) is str and final_text.startswith(stable_text)
                    and _ordered_json(source) == before, "generation-header render changed history")
            validate_identity_normalization(text=stable_text,
                normalizer=self.tokenizer.backend_tokenizer.normalizer)
            self._semantic.check_text(stable_text)
            stable = tuple(self.tokenizer.encode(stable_text, add_special_tokens=False))
            raw = self.producer._tokenizer.encode(stable_text, add_special_tokens=False)
            require(tuple(raw.ids) == stable and not raw.overflowing and
                    tokens[:len(stable)] == stable,
                    "generation-header token boundary is not a canonical prefix")
            observation = None
        else:
            start, end = rendered.observation_start, rendered.observation_end
            require(type(start) is int and type(end) is int and 0 <= start < end == len(tokens),
                    "growing turn requires actual traced terminal Observation")
            stable = tokens[:start]
            observation = (start, end)
        self._check_identity()
        # Legacy tool-call render (refused at startup in Stage B): no rebuilds.
        return authority.render(native_request, session=session, turn=turn,
            local_turn=local_turn, rebuild=False,
            generation=generation, model=model, renderer=renderer, tokens=tokens,
            stable_prefix_tokens=stable, observation_span=observation,
            native_prefix=native_prefix)



def bind_vllm_template_callback(*, hf_renderer_module, model_config, tokenizer, verify_identity):
    """Bind the actual imported vLLM HF helper, without importing vLLM here.

    Identity inventory must explicitly include safe_apply_chat_template, its
    template resolution helpers, parser methods and HF tokenizer implementation.
    Capture/normalization remains the frontend TrustedRendererAdapter's job.
    """
    require(getattr(hf_renderer_module, "__name__", None) == "vllm.renderers.hf",
            "actual loaded vLLM HF renderer module required")
    fn = getattr(hf_renderer_module, "safe_apply_chat_template", None)
    require(callable(fn) and callable(verify_identity), "loaded template helper required")
    def render(template, conversation, kwargs):
        require(verify_identity() is True
                and hf_renderer_module.safe_apply_chat_template is fn,
                "loaded template callback identity changed")
        return fn(model_config, tokenizer, conversation, chat_template=template,
                  tokenize=False, return_dict=False, **kwargs)
    return render

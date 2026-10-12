from __future__ import annotations

import logging
from typing import List

import torch
from minisgl.message import TokenizeMsg
from minisgl.phase import SemanticPhase
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        debug_rendered_prompt: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.debug_rendered_prompt = debug_rendered_prompt
        self.logger = logger or logging.getLogger(__name__)

    def tokenize(
        self,
        msgs: List[TokenizeMsg],
    ) -> List[tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]]:
        results: List[tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                kwargs = {}
                if msg.tools:
                    kwargs["tools"] = msg.tools
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            if self.debug_rendered_prompt:
                self.logger.info(f"Rendered prompt for uid={msg.uid}:\n{prompt}")
            try:
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )
                input_ids: torch.Tensor = encoded["input_ids"]  # type: ignore
                offsets: torch.Tensor = encoded["offset_mapping"][0]  # type: ignore
                phase_ids = _phase_ids_from_offsets(prompt, offsets)
                semantic_phase_ids = _semantic_phase_ids_from_offsets(prompt, offsets)
            except NotImplementedError:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")  # type: ignore
                phase_ids = torch.zeros(input_ids.numel(), dtype=torch.int32)
                semantic_phase_ids = torch.full(
                    (input_ids.numel(),),
                    int(SemanticPhase.UNKNOWN),
                    dtype=torch.int32,
                )
            results.append(
                (
                    input_ids.view(-1).to(torch.int32),
                    prompt,
                    phase_ids,
                    semantic_phase_ids,
                )
            )
        return results


def _phase_ids_from_offsets(prompt: str, offsets: torch.Tensor) -> torch.Tensor:
    char_phase = _char_phase_ids(prompt)
    return _token_ids_from_offsets(char_phase, offsets, num_stages=4, fallback_stage=0)


def _semantic_phase_ids_from_offsets(
    prompt: str,
    offsets: torch.Tensor,
) -> torch.Tensor:
    char_phase = _char_semantic_phase_ids(prompt)
    return _token_ids_from_offsets(
        char_phase,
        offsets,
        num_stages=max(int(phase) for phase in SemanticPhase) + 1,
        fallback_stage=int(SemanticPhase.UNKNOWN),
    )


def _token_ids_from_offsets(
    char_ids: List[int],
    offsets: torch.Tensor,
    *,
    num_stages: int,
    fallback_stage: int,
) -> torch.Tensor:
    token_phases = torch.empty(offsets.shape[0], dtype=torch.int32)
    last_phase = fallback_stage
    for i, offset in enumerate(offsets.tolist()):
        start, end = offset
        if end <= start:
            token_phases[i] = last_phase
            continue
        span = char_ids[start:end]
        if not span:
            token_phases[i] = last_phase
            continue
        counts = [span.count(phase) for phase in range(num_stages)]
        ignored = span.count(-1)
        if ignored > max(counts, default=0):
            token_phases[i] = -1
            last_phase = -1
            continue
        phase = max(range(num_stages), key=lambda item: counts[item])
        token_phases[i] = phase
        last_phase = phase
    return token_phases


def _char_phase_ids(prompt: str) -> List[int]:
    markers = {
        "<|im_start|>system": 0,
        "<|im_start|>user": 0,
        "<|im_start|>assistant": 1,
        "<think>": 1,
        "</think>": 2,
        "<tool_call>": 2,
        "</tool_call>": 2,
        "<tool_response>": 3,
        "</tool_response>": 3,
        "<|im_end|>": 0,
    }
    phases: List[int] = []
    phase = 0
    i = 0
    while i < len(prompt):
        matched = None
        for marker, marker_phase in markers.items():
            if prompt.startswith(marker, i):
                matched = (marker, marker_phase)
                break
        if matched is None:
            phases.append(phase)
            i += 1
            continue
        marker, marker_phase = matched
        phase = marker_phase
        phases.extend([phase] * len(marker))
        i += len(marker)
    return phases


def _char_semantic_phase_ids(prompt: str) -> List[int]:
    phases: List[int] = []
    phase = int(SemanticPhase.UNKNOWN)
    i = 0
    while i < len(prompt):
        role_marker = _match_role_marker(prompt, i)
        if role_marker is not None:
            marker, role = role_marker
            content_start = i + len(marker)
            content_end = prompt.find("<|im_end|>", content_start)
            content = prompt[content_start:] if content_end < 0 else prompt[content_start:content_end]
            if role == "system":
                phase = int(SemanticPhase.SYSTEM)
            elif role == "assistant" and _looks_like_bfcl_python_tool_call(content):
                phase = int(SemanticPhase.ASSISTANT_TOOL_CALL)
            elif role == "assistant":
                phase = int(SemanticPhase.ASSISTANT_TEXT)
            elif role == "user" and _looks_like_formatted_tool_result(content):
                phase = int(SemanticPhase.TOOL_RESPONSE)
            elif role == "user":
                phase = int(SemanticPhase.USER)
            else:
                phase = int(SemanticPhase.UNKNOWN)
            phases.extend([phase] * len(marker))
            i += len(marker)
            continue

        marker_phase = _match_semantic_phase_marker(prompt, i, phase)
        if marker_phase is not None:
            marker, marker_phase_id = marker_phase
            phase = marker_phase_id
            phases.extend([phase] * len(marker))
            i += len(marker)
            continue

        phases.append(phase)
        i += 1
    return phases


def _match_role_marker(prompt: str, offset: int) -> tuple[str, str] | None:
    for role in ("system", "assistant", "user"):
        marker = f"<|im_start|>{role}\n"
        if prompt.startswith(marker, offset):
            return marker, role
    return None


def _match_semantic_phase_marker(
    prompt: str,
    offset: int,
    current_phase: int,
) -> tuple[str, int] | None:
    if prompt.startswith("<tool_response>", offset):
        return "<tool_response>", int(SemanticPhase.TOOL_RESPONSE)
    if prompt.startswith("</tool_response>", offset):
        return "</tool_response>", int(SemanticPhase.TOOL_RESPONSE)
    if prompt.startswith("<tool_call>", offset):
        return "<tool_call>", int(SemanticPhase.ASSISTANT_TOOL_CALL)
    if prompt.startswith("</tool_call>", offset):
        return "</tool_call>", int(SemanticPhase.ASSISTANT_TOOL_CALL)
    if prompt.startswith("<think>", offset):
        return "<think>", int(SemanticPhase.ASSISTANT_TEXT)
    if prompt.startswith("</think>", offset):
        return "</think>", int(SemanticPhase.ASSISTANT_TEXT)
    if prompt.startswith("<|im_end|>", offset):
        return "<|im_end|>", current_phase
    return None


def _looks_like_formatted_tool_result(content: str) -> bool:
    stripped = content.lstrip()
    prefix = stripped[:256]
    return (
        prefix.startswith("[{'role': 'tool'")
        or prefix.startswith('[{"role": "tool"')
        or "'role': 'tool'" in prefix
        or '"role": "tool"' in prefix
    )


def _looks_like_bfcl_python_tool_call(content: str) -> bool:
    stripped = content.lstrip()
    if not stripped.startswith("["):
        return False
    end = stripped.find("]")
    if end < 0:
        return False
    prefix = stripped[: end + 1]
    return "(" in prefix and ")" in prefix and "=" in prefix

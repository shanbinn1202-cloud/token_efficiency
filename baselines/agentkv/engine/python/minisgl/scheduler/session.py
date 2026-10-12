from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from minisgl.core import Req
from minisgl.message import UserMsg
from minisgl.utils import div_ceil

from .utils import PendingReq


SessionStatus = Literal["RUNNING", "PARKED"]


class SessionError(RuntimeError):
    pass


@dataclass
class SessionState:
    session_id: str
    status: SessionStatus
    req: Req | None
    full_token_ids: torch.Tensor
    retained_token_ids: torch.Tensor
    retained_positions: torch.Tensor
    logical_cached_len: int
    full_phase_ids: torch.Tensor | None = None
    retained_phase_ids: torch.Tensor | None = None
    retained_semantic_phase_ids: torch.Tensor | None = None
    retained_positions_by_layer_head: torch.Tensor | None = None
    full_text: str | None = None
    pending_close: bool = False


class SessionManager:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.sessions: dict[str, SessionState] = {}
        self.closed_sessions: set[str] = set()

    def prepare_request(
        self,
        msg: UserMsg,
        *,
        encode_suffix: Callable[[str], torch.Tensor],
    ) -> PendingReq:
        session_id = msg.session_id
        if not session_id:
            raise SessionError("session_id is required")
        if session_id in self.closed_sessions:
            raise SessionError(f"session {session_id!r} was closed; create a new session_id")

        state = self.sessions.get(session_id)
        if state is None:
            retained = torch.empty(0, dtype=msg.input_ids.dtype, device="cpu")
            retained_phase_ids = torch.empty(0, dtype=torch.int32, device="cpu")
            retained_semantic_phase_ids = torch.empty(0, dtype=torch.int32, device="cpu")
            positions = torch.empty(0, dtype=torch.int32, device="cpu")
            full_phase_ids = _phase_ids_or_default(msg.phase_ids, len(msg.input_ids))
            semantic_phase_ids = _semantic_phase_ids_or_default(
                msg.semantic_phase_ids,
                len(msg.input_ids),
                fallback=full_phase_ids,
            )
            self.sessions[session_id] = SessionState(
                session_id=session_id,
                status="RUNNING",
                req=None,
                full_token_ids=msg.input_ids.clone(),
                full_phase_ids=full_phase_ids.clone(),
                retained_token_ids=retained,
                retained_phase_ids=retained_phase_ids,
                retained_semantic_phase_ids=retained_semantic_phase_ids,
                retained_positions=positions,
                retained_positions_by_layer_head=None,
                logical_cached_len=0,
                full_text=msg.rendered_text,
            )
            return PendingReq(
                uid=msg.uid,
                input_ids=msg.input_ids,
                sampling_params=msg.sampling_params,
                session_id=session_id,
                rendered_text=msg.rendered_text,
                physical_cached_len=0,
                full_input_ids=msg.input_ids,
                phase_ids=full_phase_ids,
                full_phase_ids=full_phase_ids,
                semantic_phase_ids=semantic_phase_ids,
                retained_positions=positions,
                retained_positions_by_layer_head=None,
                logical_cached_len=0,
                session_prompt_len=len(msg.input_ids),
                phase_text_tail=(msg.rendered_text or "")[-128:],
            )

        if state.status == "RUNNING":
            raise SessionError(f"session {session_id!r} already has a running request")
        assert state.req is not None
        if state.full_phase_ids is None:
            state.full_phase_ids = torch.zeros(
                len(state.full_token_ids),
                dtype=torch.int32,
                device="cpu",
            )
        if state.retained_phase_ids is None:
            state.retained_phase_ids = torch.zeros(
                len(state.retained_token_ids),
                dtype=torch.int32,
                device="cpu",
            )
        if state.retained_semantic_phase_ids is None:
            state.retained_semantic_phase_ids = state.retained_phase_ids.clone()

        old_token_ids = state.full_token_ids
        new_token_ids = msg.input_ids
        new_phase_ids = _phase_ids_or_default(msg.phase_ids, len(msg.input_ids))
        new_semantic_phase_ids = _semantic_phase_ids_or_default(
            msg.semantic_phase_ids,
            len(msg.input_ids),
            fallback=new_phase_ids,
        )
        if len(new_token_ids) < len(old_token_ids) or not torch.equal(
            new_token_ids[: len(old_token_ids)], old_token_ids
        ):
            if state.full_text is None or msg.rendered_text is None:
                raise SessionError(f"session {session_id!r} prompt must append to previous prompt")
            if not msg.rendered_text.startswith(state.full_text):
                raise SessionError(f"session {session_id!r} prompt must append to previous prompt")
            # The client may detokenize and re-tokenize identical history differently.
            # Keep parked KV tokens authoritative and only encode the newly appended text.
            suffix_text = msg.rendered_text[len(state.full_text) :]
            suffix_token_ids = encode_suffix(suffix_text)
            new_token_ids = torch.cat([old_token_ids, suffix_token_ids])
            suffix_phase_ids = _suffix_phase_ids_from_rendered_prompt(
                new_phase_ids,
                suffix_len=len(suffix_token_ids),
                fallback_phase=(
                    int(state.full_phase_ids[-1].item()) if len(state.full_phase_ids) > 0 else 0
                ),
            )
            new_phase_ids = torch.cat([state.full_phase_ids, suffix_phase_ids])
            suffix_semantic_phase_ids = _suffix_phase_ids_from_rendered_prompt(
                new_semantic_phase_ids,
                suffix_len=len(suffix_token_ids),
                fallback_phase=(
                    int(state.retained_semantic_phase_ids[-1].item())
                    if len(state.retained_semantic_phase_ids) > 0
                    else -1
                ),
            )
            new_semantic_phase_ids = torch.cat(
                [new_semantic_phase_ids[: len(old_token_ids)], suffix_semantic_phase_ids]
            )

        physical_cached_len = len(state.retained_token_ids)
        if state.logical_cached_len > len(new_token_ids):
            raise SessionError(f"session {session_id!r} prompt is shorter than cached KV")
        if len(new_token_ids) <= state.logical_cached_len:
            raise SessionError(
                f"session {session_id!r} request does not append tokens beyond cached KV"
            )
        suffix_token_ids = new_token_ids[state.logical_cached_len :]
        suffix_phase_ids = new_phase_ids[state.logical_cached_len :]
        suffix_semantic_phase_ids = new_semantic_phase_ids[state.logical_cached_len :]
        physical_input_ids = torch.cat([state.retained_token_ids, suffix_token_ids])
        physical_phase_ids = torch.cat([state.retained_phase_ids, suffix_phase_ids])
        physical_semantic_phase_ids = torch.cat(
            [state.retained_semantic_phase_ids, suffix_semantic_phase_ids]
        )

        state.status = "RUNNING"
        state.pending_close = False
        return PendingReq(
            uid=msg.uid,
            input_ids=physical_input_ids,
            sampling_params=msg.sampling_params,
            session_id=session_id,
            rendered_text=msg.rendered_text,
            session_req=state.req,
            physical_cached_len=physical_cached_len,
            full_input_ids=new_token_ids,
            phase_ids=physical_phase_ids,
            full_phase_ids=new_phase_ids,
            semantic_phase_ids=physical_semantic_phase_ids,
            retained_positions=state.retained_positions,
            retained_positions_by_layer_head=state.retained_positions_by_layer_head,
            logical_cached_len=state.logical_cached_len,
            session_prompt_len=len(new_token_ids),
            phase_text_tail=(msg.rendered_text or state.full_text or "")[-128:],
        )

    def finish_request(
        self,
        req: Req,
        *,
        drop_last_token: bool = False,
        full_text: str | None = None,
    ) -> bool:
        if req.session_id is None:
            return False
        state = self.sessions.get(req.session_id)
        if state is None:
            if req.session_id in self.closed_sessions:
                return True
            state = SessionState(
                session_id=req.session_id,
                status="RUNNING",
                req=None,
                full_token_ids=_full_token_ids(req, drop_last_token=False),
                full_phase_ids=_full_phase_ids(req, drop_last_token=False),
                retained_token_ids=req.input_ids[: req.physical_cached_len].clone(),
                retained_phase_ids=req.phase_ids[: req.physical_cached_len].clone(),
                retained_semantic_phase_ids=req.semantic_phase_ids[
                    : req.physical_cached_len
                ].clone(),
                retained_positions=req.retained_positions.clone(),
                retained_positions_by_layer_head=(
                    req.retained_positions_by_layer_head.clone()
                    if req.retained_positions_by_layer_head is not None
                    else None
                ),
                logical_cached_len=req.logical_cached_len,
                full_text=full_text or req.session_text,
            )
            self.sessions[req.session_id] = state

        if state.pending_close:
            del self.sessions[req.session_id]
            self.closed_sessions.add(req.session_id)
            return True

        full_token_ids = _full_token_ids(req, drop_last_token=False)
        full_phase_ids = _full_phase_ids(req, drop_last_token=False)
        if drop_last_token:
            full_token_ids = full_token_ids[:-1]
            full_phase_ids = full_phase_ids[:-1]
        state.req = req
        state.full_token_ids = full_token_ids.clone()
        state.full_phase_ids = full_phase_ids.clone()
        state.retained_token_ids = req.input_ids[: req.physical_cached_len].clone()
        state.retained_phase_ids = req.phase_ids[: req.physical_cached_len].clone()
        state.retained_semantic_phase_ids = req.semantic_phase_ids[
            : req.physical_cached_len
        ].clone()
        state.retained_positions = req.retained_positions.clone()
        state.retained_positions_by_layer_head = (
            req.retained_positions_by_layer_head.clone()
            if req.retained_positions_by_layer_head is not None
            else None
        )
        state.logical_cached_len = req.logical_cached_len
        if full_text is not None:
            state.full_text = full_text
        state.status = "PARKED"
        return False

    def close(self, session_id: str) -> Req | None:
        if session_id in self.closed_sessions:
            return None
        state = self.sessions.get(session_id)
        if state is None:
            return None
        if state.status == "RUNNING":
            state.pending_close = True
            self.closed_sessions.add(session_id)
            return None
        req = state.req
        del self.sessions[session_id]
        self.closed_sessions.add(session_id)
        return req

    def close_all_parked(self) -> list[Req]:
        reqs = []
        for state in self.sessions.values():
            if state.req is not None:
                reqs.append(state.req)
        self.sessions.clear()
        self.closed_sessions.clear()
        return reqs

    def cancel_prepare(self, pending_req: PendingReq) -> None:
        if pending_req.session_id is None:
            return
        state = self.sessions.get(pending_req.session_id)
        if state is None or state.status != "RUNNING":
            return
        if pending_req.session_req is None:
            del self.sessions[pending_req.session_id]
        else:
            state.status = "PARKED"

    def remove_running(self, req: Req, *, tombstone: bool = True) -> None:
        if req.session_id is None:
            return
        state = self.sessions.get(req.session_id)
        if state is not None and state.status == "RUNNING":
            del self.sessions[req.session_id]
        if tombstone:
            self.closed_sessions.add(req.session_id)

    def owned_pages(self) -> int:
        total = 0
        for state in self.sessions.values():
            if state.req is not None:
                total += div_ceil(state.req.physical_cached_len, self.page_size)
        return total


def _full_token_ids(req: Req, *, drop_last_token: bool) -> torch.Tensor:
    req.materialize_full_host()
    token_ids = req.full_input_ids if req.full_input_ids is not None else req.input_ids
    if drop_last_token:
        token_ids = token_ids[:-1]
    return token_ids


def _full_phase_ids(req: Req, *, drop_last_token: bool) -> torch.Tensor:
    req.materialize_full_host()
    phase_ids = req.full_phase_ids if req.full_phase_ids is not None else req.phase_ids
    assert phase_ids is not None
    if drop_last_token:
        phase_ids = phase_ids[:-1]
    return phase_ids


def _phase_ids_or_default(phase_ids: torch.Tensor | None, length: int) -> torch.Tensor:
    if phase_ids is None:
        return torch.zeros(length, dtype=torch.int32, device="cpu")
    if len(phase_ids) != length:
        raise SessionError("phase_ids length must match input_ids")
    return phase_ids.to(dtype=torch.int32, device="cpu")


def _semantic_phase_ids_or_default(
    stage_ids: torch.Tensor | None,
    length: int,
    *,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if stage_ids is None:
        return fallback.to(dtype=torch.int32, device="cpu")
    if len(stage_ids) != length:
        raise SessionError("semantic_phase_ids length must match input_ids")
    return stage_ids.to(dtype=torch.int32, device="cpu")


def _suffix_phase_ids_from_rendered_prompt(
    rendered_phase_ids: torch.Tensor,
    *,
    suffix_len: int,
    fallback_phase: int,
) -> torch.Tensor:
    if suffix_len <= 0:
        return torch.empty(0, dtype=torch.int32, device="cpu")
    if len(rendered_phase_ids) >= suffix_len:
        return rendered_phase_ids[-suffix_len:].to(dtype=torch.int32, device="cpu")
    return torch.full((suffix_len,), fallback_phase, dtype=torch.int32, device="cpu")

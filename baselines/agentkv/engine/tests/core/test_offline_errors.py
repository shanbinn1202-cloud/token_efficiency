from __future__ import annotations

from minisgl.llm.llm import LLM, RequestStatus
from minisgl.message import DetokenizeMsg, TokenizerErrorMsg


def test_offline_send_result_records_tokenizer_errors() -> None:
    llm = LLM.__new__(LLM)
    llm.eos_token_id = 2
    llm.status_map = {0: RequestStatus(uid=0, input_ids=[1], output_ids=[])}

    llm.offline_send_result([TokenizerErrorMsg(uid=0, error="bad session")])

    assert llm.status_map[0].error == "bad session"
    assert llm.status_map[0].output_ids == []


def test_offline_send_result_still_records_tokens() -> None:
    llm = LLM.__new__(LLM)
    llm.eos_token_id = 2
    llm.status_map = {0: RequestStatus(uid=0, input_ids=[1], output_ids=[])}

    llm.offline_send_result(
        [
            DetokenizeMsg(uid=0, next_token=3, finished=False),
            DetokenizeMsg(uid=0, next_token=2, finished=True),
        ]
    )

    assert llm.status_map[0].error is None
    assert llm.status_map[0].output_ids == [3]

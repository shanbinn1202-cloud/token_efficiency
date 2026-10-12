"""Persistent responses must contribute independent native latency samples."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engines/vllm'))
from vllm.v1.engine import EngineCoreOutput, EngineCoreEvent, EngineCoreEventType, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor, RequestState
from vllm.v1.metrics.stats import IterationStats, RequestStateStats
from vllm.v1.engine.output_processor import StreamingUpdate


def test_streamed_responses_exclude_parked_time_and_reset_token_counts():
    processor = OutputProcessor(None, log_stats=True)
    state = NS(request_id='r', external_req_id='r', stats=RequestStateStats(arrival_time=100),
        is_prefilling=True, lora_name=None, num_cached_tokens=0, prompt_len=2,
        max_tokens_param=4, parent_req=None, streaming_input=True, input_chunk_queue=deque(),
        detokenizer=NS(update=lambda *args: None), logprobs_processor=NS(update_from_output=lambda x: None),
        make_request_output=lambda *args: None, prompt='', prompt_token_ids=[1,2])
    processor.request_states['r'] = state
    samples = []
    for queued, scheduled, first, last in [(10,11,12,14),(110,112,113,116)]:
        iteration = IterationStats()
        processor.process_outputs([EngineCoreOutput(request_id='r',new_token_ids=[3],events=[
            EngineCoreEvent(type=EngineCoreEventType.QUEUED,timestamp=queued),
            EngineCoreEvent(type=EngineCoreEventType.SCHEDULED,timestamp=scheduled)])], first, iteration)
        assert not iteration.finished_requests
        processor.process_outputs([EngineCoreOutput(request_id='r',new_token_ids=[4],
            finish_reason=FinishReason.LENGTH)],last,iteration)
        assert len(iteration.finished_requests)==1
        samples.append(iteration.finished_requests[0])
        assert processor.request_states['r'] is state
        RequestState.apply_streaming_update(state, StreamingUpdate(prompt='next',prompt_token_ids=[5],arrival_time=200))
    assert [s.num_generation_tokens for s in samples]==[2,2]
    assert [s.queued_time for s in samples]==[1,2]
    assert [s.prefill_time for s in samples]==[1,1]
    assert [s.decode_time for s in samples]==[2,3]

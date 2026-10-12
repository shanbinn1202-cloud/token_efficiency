import asyncio
from types import SimpleNamespace as NS

from experiments.benchmarks32_20261004.telemetry import Ledger, TAG


class Processor:
    def __init__(self):
        self.request_states = {}

    def add_request(self, request):
        self.request_states[request.request_id] = NS(
            detokenizer=NS(output_token_ids=[]), external_req_id=request.request_id)

    def process_outputs(self, outputs):
        for output in outputs:
            state = self.request_states.get(output.request_id)
            if state is None:
                continue
            state.detokenizer.output_token_ids.extend(output.tokens)
            if output.done:
                self.request_states.pop(output.request_id)
        return 'unchanged-return'


def test_partial_counts_and_frozen_window(tmp_path):
    async def check():
        processor = Processor()
        ledger = Ledger(processor, tmp_path/'ledger.json')
        for kind in ('agent', 'auxiliary', 'warmup'):
            tag = TAG.set((kind, 'case-1'))
            processor.add_request(NS(request_id=kind))
            TAG.reset(tag)
        processor.process_outputs([NS(request_id='warmup', tokens=[0]*9, done=True)])
        ledger.begin()
        assert processor.process_outputs([
            NS(request_id='agent', tokens=[1, 2, 3], done=False),
            NS(request_id='auxiliary', tokens=[4, 5], done=True),
            NS(request_id='aborted', tokens=[6]*30, done=True),
        ]) == 'unchanged-return'
        frozen = ledger.freeze()
        assert frozen['tokens'] == {'agent': 3, 'auxiliary': 2}
        assert frozen['unfinished_requests'] == 1
        assert frozen['requests']['auxiliary']['finished']
        processor.process_outputs([NS(request_id='agent', tokens=[7]*8, done=True)])
        assert ledger.snapshot()['tokens']['agent'] == 3
        ledger.reset_validation()
        assert not ledger.tokens
        ledger.task.cancel()
    asyncio.run(check())

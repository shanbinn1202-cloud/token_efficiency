"""Observe accepted native output tokens without changing inference decisions."""
import asyncio
from collections import Counter
from contextvars import ContextVar
import json
import os
from pathlib import Path
import threading
import time

TAG = ContextVar('benchmark_request_tag', default=('warmup', ''))
LOCAL = threading.local()


def write(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


class Ledger:
    def __init__(self, processor, output):
        self.processor = processor
        self.output = Path(output)
        self.active = False
        self.started = self.finished = None
        self.tags = {}
        self.requests = {}
        self.tokens = Counter()
        add = processor.add_request
        process = processor.process_outputs

        def add_request(request, *args, **kwargs):
            result = add(request, *args, **kwargs)
            self.tags[request.request_id] = TAG.get()
            return result

        def process_outputs(engine_core_outputs, *args, **kwargs):
            previous = {}
            for output in engine_core_outputs:
                state = processor.request_states.get(output.request_id)
                if state is not None and state.detokenizer is not None:
                    previous[output.request_id] = (state, len(state.detokenizer.output_token_ids))
            result = process(engine_core_outputs, *args, **kwargs)
            for request_id, (state, before) in previous.items():
                count = len(state.detokenizer.output_token_ids)-before
                kind, case = self.tags.get(request_id, ('untagged', ''))
                done = request_id not in processor.request_states
                if self.active and count:
                    row = self.requests.setdefault(request_id, dict(kind=kind, case=case,
                        tokens=0, finished=False, external_id=state.external_req_id))
                    row['tokens'] += count
                    self.tokens[kind] += count
                if self.active and request_id in self.requests:
                    self.requests[request_id]['finished'] = done
                if done:
                    self.tags.pop(request_id, None)
            return result

        processor.add_request = add_request
        processor.process_outputs = process_outputs
        self.task = asyncio.create_task(self.persist())

    def snapshot(self, detailed=False):
        end = self.finished or time.time()
        wall = end-self.started if self.started is not None else 0
        result = dict(time=time.time(), started=self.started, finished=self.finished,
            active=self.active, wall_seconds=wall, tokens=dict(self.tokens),
            agent_tokens_per_second=self.tokens['agent']/wall if wall else None,
            request_count=len(self.requests),
            unfinished_requests=sum(not r['finished'] for r in self.requests.values()))
        if detailed:
            result['requests'] = self.requests
        return result

    async def persist(self):
        while True:
            await asyncio.sleep(5)
            if self.started is not None:
                write(self.output, self.snapshot())

    def begin(self):
        if self.started is not None:
            raise RuntimeError('Measurement interval is already fixed')
        self.started = time.time()
        self.active = True
        return self.snapshot()

    def reset_validation(self):
        if self.active or self.processor.request_states:
            raise RuntimeError('Cannot reset during active inference')
        self.task.cancel()
        self.started = self.finished = None
        self.requests.clear()
        self.tokens.clear()
        self.tags.clear()
        self.task = asyncio.create_task(self.persist())
        return self.snapshot()

    def freeze(self):
        if self.finished is None:
            self.finished = time.time()
            self.active = False
        result = self.snapshot(detailed=True)
        write(self.output, result)
        self.task.cancel()
        return result


async def middleware(request, call_next):
    from starlette.responses import JSONResponse
    engine = request.app.state.engine_client
    ledger = getattr(engine, '_benchmark_ledger', None)
    if ledger is None:
        ledger = engine._benchmark_ledger = Ledger(
            engine.output_processor, os.environ['BENCHMARK_LEDGER_PATH'])
    if request.url.path == '/benchmark-metrics':
        return JSONResponse(ledger.snapshot(detailed=request.query_params.get('detailed') == '1'))
    if request.url.path == '/benchmark-begin' and request.method == 'POST':
        return JSONResponse(ledger.begin())
    if request.url.path == '/benchmark-freeze' and request.method == 'POST':
        return JSONResponse(ledger.freeze())
    if request.url.path == '/benchmark-reset-validation' and request.method == 'POST':
        return JSONResponse(ledger.reset_validation())
    token = TAG.set((request.headers.get('x-benchmark-kind', 'warmup'),
                     request.headers.get('x-benchmark-case', '')))
    try:
        return await call_next(request)
    finally:
        TAG.reset(token)


def instrument_backend(backend, log):
    """Tag the existing HTTP transports; all sampling and policy args pass through."""
    client = getattr(backend.client, '_client', backend.client)
    def headers(request):
        request.headers['x-benchmark-kind'] = getattr(LOCAL, 'kind', 'warmup')
        request.headers['x-benchmark-case'] = getattr(LOCAL, 'case', '')
    client.event_hooks['request'].append(headers)
    complete = backend.complete
    lock = threading.Lock()

    def wrapped(**kwargs):
        LOCAL.kind = ('auxiliary' if kwargs.get('_text_response') else 'agent') if getattr(LOCAL, 'case', '') else 'warmup'
        start = time.time()
        try:
            result = complete(**kwargs)
        except Exception as exc:
            row = dict(case=getattr(LOCAL,'case',''),kind=LOCAL.kind,start=start,
                       finish=time.time(),error=str(exc))
            with lock, Path(log).open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            raise
        row = dict(case=getattr(LOCAL,'case',''),kind=LOCAL.kind,start=start,
                   finish=time.time(),usage=result.usage.model_dump())
        with lock, Path(log).open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        return result
    backend.complete = wrapped
    return backend

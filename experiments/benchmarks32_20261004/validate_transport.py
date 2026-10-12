"""Check per-request telemetry against native usage before the timed experiment."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import time

import httpx

from tebench.backends import create
from experiments.benchmarks32_20261004.telemetry import LOCAL, instrument_backend, write


def warm_policy(config, method, backend, stage):
    """Compile initial selection/packing paths outside the measured interval."""
    if method not in {'random_attention', 'sliding_window', 'agentkv', 'intentkv'}:
        return
    LOCAL.case = ''
    unit = ' Observation value 7.'
    unit_tokens = len(backend.tokenizer.encode(unit, add_special_tokens=False))
    budget = config['cache']['kv_budget']
    observation = unit * ((budget+2048+unit_tokens-1)//unit_tokens)
    assert len(backend.tokenizer.encode(observation, add_special_tokens=False)) > budget
    messages = [{'role':'user', 'content':'Remember that the target value is 7.'}]
    traces = []
    try:
        # Keep the protected initial prompt small; long later observations
        # exercise eviction of history rather than extending a protected prefix.
        for feedback in (observation, unit*256, None):
            result = backend.complete(model=config['model']['name'], messages=messages,
                                      max_tokens=160)
            traces.append(dict(backend.last_measurement))
            if feedback is not None:
                messages += [result.choices[0].message.model_dump(exclude_none=True),
                             {'role':'user','content':feedback+' State the target value.'}]
        write(stage/'policy-warmup.json', dict(passed=True, measured=False,
              budget=budget, traces=traces))
    finally:
        backend.reset()


def validate(config, method, stage):
    root = config['backend']['endpoint'].removesuffix('/v1')
    backend = instrument_backend(create(config, method), stage/'validation-requests.jsonl')
    expected = Counter()
    with httpx.Client(timeout=20) as control:
        def post(path):
            response = control.post(root+path)
            response.raise_for_status()
            return response.json()
        def case(i):
            LOCAL.case = 'validation-'+str(i)
            messages = [{'role':'user','content':f'Reply with the number {i}.'}]
            counts = Counter()
            try:
                for turn in range(2):
                    response = backend.complete(model=config['model']['name'],
                        messages=messages, max_tokens=32)
                    counts['agent'] += response.usage.completion_tokens
                    messages += [response.choices[0].message.model_dump(exclude_none=True),
                                 {'role':'user','content':'Repeat that number.'}]
                if method == 'agentdiet':
                    response = backend.complete(model=config['model']['name'],
                        messages=[{'role':'user','content':'Say OK.'}], max_tokens=8,
                        _text_response=True)
                    counts['auxiliary'] += response.usage.completion_tokens
                return counts
            finally:
                backend.reset()
                LOCAL.case = ''
        try:
            warm_policy(config, method, backend, stage)
            post('/benchmark-begin')
            with ThreadPoolExecutor(max_workers=16) as pool:
                for result in pool.map(case, range(16)):
                    expected.update(result)
            result = post('/benchmark-freeze')
            assert result['tokens'] == dict(expected), (result['tokens'], expected)
            assert result['unfinished_requests'] == 0, result
            write(stage/'transport-validation.json',dict(passed=True, concurrency=16,
                  turns=2,expected=dict(expected),measurement=result))
            post('/benchmark-reset-validation')
            # A non-streaming native request must be visible before its response.
            post('/benchmark-begin')
            def long_request():
                # Stateful policies require their normal session metadata;
                # exercise the same transport used by the measured requests.
                LOCAL.case = 'partial-validation'
                try:
                    response = backend.complete(model=config['model']['name'],
                        messages=[{'role':'user','content':
                            'Write every integer from 1 through 1000, one per line.'}],
                        max_tokens=512, temperature=0)
                    return {'usage':response.usage.model_dump()}
                finally:
                    backend.reset()
                    LOCAL.case = ''
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(long_request)
                deadline = time.time()+120
                while time.time()<deadline:
                    partial = control.get(root+'/benchmark-metrics').json()
                    if partial['tokens'].get('agent',0)>0:
                        assert not future.done(), 'Partial request finished before observation'
                        break
                    time.sleep(.05)
                else:
                    raise TimeoutError('No partial output tokens observed')
                frozen = post('/benchmark-freeze')
                final = future.result()
            assert 0 < frozen['tokens']['agent'] < final['usage']['completion_tokens']
            assert frozen['unfinished_requests'] == 1
            assert control.get(root+'/benchmark-metrics').json()['tokens'] == frozen['tokens']
            write(stage/'partial-validation.json',dict(passed=True,frozen=frozen,
                  completed_usage=final['usage']))
            post('/benchmark-reset-validation')
        finally:
            backend.close()

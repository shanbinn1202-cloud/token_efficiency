"""Exercise native RECOMPUTE after a real RepairKV selection and query probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event
import time
import httpx
from tebench.config import load
from tebench.backends import create


def trajectory(config, backend, ready=None):
    try:
        noise = backend.tokenizer.encode('All unrelated repository tests pass.\n', add_special_tokens=False)
        text = 'Remember the private identifier: repair_preempt_7391.\n'
        text += backend.tokenizer.decode(noise*(config['cache']['kv_budget']//len(noise)+180))
        messages = [{'role': 'user', 'content': text+'\nReturn the private identifier.'}]
        first = backend.complete(model=config['model']['name'], messages=messages, max_tokens=32)
        messages += [first.choices[0].message.model_dump(exclude_none=True),
            {'role': 'user', 'content': 'Explain the private identifier in detail, then enumerate 100 ways to check it.'}]
        if ready is not None:
            ready.set()
        second = backend.complete(model=config['model']['name'], messages=messages, max_tokens=512)
        measurement = dict(backend.last_measurement)
        assert measurement['repairs'] == measurement['recurrent_rollbacks'] == 1
        assert measurement['pool_tokens'] == measurement['logical_computed_tokens']
        assert measurement['native_request_finished']
        assert measurement['last_compression']['resumed_tokens'] == config['cache']['kv_budget']
        return dict(measurement=measurement, text=second.choices[0].message.content)
    finally:
        backend.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = load(args.config)
    baseline = trajectory(config, create(config, 'repairkv'))
    backend = create(config, 'repairkv')
    ready = Event()
    resets = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(trajectory, config, backend, ready)
        assert ready.wait(timeout=1200)
        with httpx.Client(timeout=1200) as client:
            for delay in (2., 1.):
                time.sleep(delay)
                response = client.post(config['backend']['endpoint'].removesuffix('/v1')+
                    '/reset_prefix_cache?reset_running_requests=true')
                response.raise_for_status()
                resets.append(response.json())
        forced = future.result(timeout=1200)
    m = forced['measurement']
    assert m['preemptions'] > 0 and m['recomputed_tokens'] > 0
    same_selection = baseline['measurement']['last_repair']['selected'] == m['last_repair']['selected']
    assert same_selection
    args.output.write_text(json.dumps(dict(passed=True, resets=resets,
        selection_identical=same_selection, text_identical=baseline['text']==forced['text'],
        baseline=baseline, forced=forced), indent=2))
    print('REPAIRKV_NATIVE_PREEMPTION_PASSED', flush=True)


if __name__ == '__main__':
    main()

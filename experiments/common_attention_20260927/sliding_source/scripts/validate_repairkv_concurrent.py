"""Independent long-context sessions sharing one native vLLM batcher."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
from threading import Barrier
from tebench.config import load
from tebench.backends import create


def case(config, gate, number, backend, signal_tokens=0):
    try:
        noise = backend.tokenizer.encode('All unrelated repository tests pass.\n', add_special_tokens=False)
        text = f'Private identifier for this session: case_{number}.\n'
        text += backend.tokenizer.decode(noise*(config['cache']['kv_budget']//len(noise)+180))
        messages = [{'role':'user', 'content':text+'\nReturn the private identifier.'}]
        turns = []
        for turn in range(3):
            gate.wait(timeout=1200)
            result = backend.complete(model=config['model']['name'], messages=messages, max_tokens=32)
            m = dict(backend.last_measurement)
            assert m['repairs'] == turn
            assert m['recurrent_rollbacks'] == turn
            assert m['restored_tokens'] == turn*config['methods']['repairkv']['options']['repair_tokens']
            assert m['logical_computed_tokens'] == m['pool_tokens']
            assert m['pool_device'].startswith('cuda')
            assert m['native_request_finished']
            if turn:
                assert m['last_compression']['resumed_tokens'] == config['cache']['kv_budget']
            turns.append(m)
            signal = backend.tokenizer.decode(noise*(signal_tokens//len(noise))) if signal_tokens else ''
            messages += [result.choices[0].message.model_dump(exclude_none=True),
                         {'role':'user','content':signal+f'Recheck the private identifier for session {number}.'}]
        return dict(case=number, turns=turns)
    except BaseException:
        gate.abort()
        raise
    finally:
        backend.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sessions', type=int, default=2)
    p.add_argument('--signal-tokens', type=int, default=0)
    args = p.parse_args()
    config = load(args.config)
    gate = Barrier(args.sessions)
    # Transformers uses lazy imports. Initialize clients serially so an import
    # failure cannot leave another worker waiting alone at the first barrier.
    backends = [create(config, 'repairkv') for _ in range(args.sessions)]
    with ThreadPoolExecutor(max_workers=args.sessions) as pool:
        futures = [pool.submit(case, config, gate, number, backends[number], args.signal_tokens) for number in range(args.sessions)]
        cases = [future.result() for future in futures]
    args.output.write_text(json.dumps(dict(passed=True, cases=cases), indent=2))
    print('REPAIRKV_CONCURRENT_GPU_SMOKE_PASSED', flush=True)


if __name__ == '__main__':
    main()

"""Exercise real eviction/repair, chunked prefills and repeated probe rollback.

Not an accuracy/throughput benchmark. Server configuration is unchanged;
each synthetic response is limited to 32 tokens to keep this test small.
"""
import argparse
import json
from pathlib import Path
import time

from tebench.config import load
from tebench.backends import create


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = load(args.config)
    backend = create(config, 'repairkv')
    report = dict(configuration=config, cases=[])
    try:
        for kind in ('short', 'long', 'growing', 'long_signal'):
            backend.reset()
            content = 'Remember the identifier repair_target = 7391.\n'
            if kind in ('long', 'long_signal'):
                noise = backend.tokenizer.encode('Unrelated build log: checks completed successfully.\n', add_special_tokens=False)
                content += backend.tokenizer.decode((noise * (config['cache']['kv_budget']//len(noise)+300)))
            messages = [{'role':'system', 'content':'Follow the user instruction.'},
                        {'role':'user', 'content':content+'\nWhat is repair_target?'}]
            case = dict(kind=kind, turns=[])
            for turn in range(3):
                began = time.perf_counter()
                result = backend.complete(model=config['model']['name'], messages=messages, max_tokens=32)
                measurement = dict(backend.last_measurement)
                case['turns'].append(dict(seconds=time.perf_counter()-began, measurement=measurement,
                    response=result.choices[0].message.content))
                assert measurement['pool_tokens'] == measurement['logical_computed_tokens']
                assert measurement['pool_device'].startswith('cuda')
                assert measurement['native_request_finished']
                assert measurement['completion_tokens'] > 0
                assert measurement['recurrent_rollbacks'] == measurement['repairs']
                expected_repairs = turn if kind in ('long', 'long_signal') else max(0, turn-1) if kind == 'growing' else 0
                assert measurement['repairs'] == expected_repairs
                if expected_repairs:
                    assert measurement['restored_tokens'] == expected_repairs*config['methods']['repairkv']['options']['repair_tokens']
                    assert measurement['last_repair']['score_layers'] > 0
                    assert measurement['last_repair']['recurrent_layers'] > 0
                if turn:
                    compression = measurement['last_compression']
                    assert compression['resumed_tokens'] <= config['cache']['kv_budget']
                    assert compression['base_tokens'] <= config['cache']['kv_budget']-config['methods']['repairkv']['options']['repair_tokens']
                followup = 'Check the earlier identifier again. What is repair_target?'
                if kind == 'growing' and turn == 0:
                    noise = backend.tokenizer.encode('New tool output: repository checks passed.\n', add_special_tokens=False)
                    followup = backend.tokenizer.decode(noise*(config['cache']['kv_budget']//len(noise)+180))+'\n'+followup
                if kind == 'long_signal' and turn == 0:
                    noise = backend.tokenizer.encode('Tool output: all repository checks completed.\n', add_special_tokens=False)
                    followup = backend.tokenizer.decode(noise*(5000//len(noise)+1))+'\n'+followup
                messages.extend([result.choices[0].message.model_dump(exclude_none=True),
                                 {'role':'user','content':followup}])
            report['cases'].append(case)
        report['passed'] = True
        print('REPAIRKV_GPU_SMOKE_PASSED', flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        backend.close()


if __name__ == '__main__':
    main()

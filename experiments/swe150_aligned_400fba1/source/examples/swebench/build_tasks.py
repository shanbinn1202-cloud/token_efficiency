"""Build serve.py task files for SWE-bench Verified instances, byte-for-byte as the pilot harness did.

For each instance it writes `task-<instance_id>.json` = {instance_id, problem_statement}, the statement verbatim
from the dataset row, encoded as json.dumps(task, indent=1) + newline. It then splits the instances, in the order
given, into the fewest server groups of at most --max-sessions (sizes differ by at most one) and writes one
`tasks-g<k>.json` list per group, encoded as json.dumps(tasks, indent=1) + newline. Start one server per group
and policy with `serve.py --task tasks-g<k>.json`, and run exactly that group's instances against it.

The multi-session SWE-bench pilots (Slurm jobs 3299922 and 3300434) used these encodings and this split; for their
ten Astropy instances, and for the other 22 of the 32 benchmark instances (verified-32.txt), the statement SHA-256
values are pinned below and checked here. Needs the `datasets` package and the dataset at its pinned revision
(REVISION; online, or cached with HF_DATASETS_OFFLINE=1).

With --pool (a session-pool server, `serve.py --max-running-sessions K`) it writes one `tasks-pool.json` with every
instance, in the order given (1 to 256; the order in which run_agents.sh refills its K agents), instead of groups.

Usage: build_tasks.py --output-dir DIR [--max-sessions 5 | --pool] INSTANCE_ID...
"""
import argparse
import hashlib
import json
from pathlib import Path

DATASET = 'princeton-nlp/SWE-Bench_Verified'   # the pilots found SWE-bench/SWE-bench_Verified identical for all ten
REVISION = 'c104f840cc67f8b6eec6f759ebc8b2693d585d4a'   # the dataset's commit (its HEAD since 2025-02-18)
MAX_SESSIONS = 32                              # serve.py: 1..32 tasks per server
MAX_TASKS = 256                                # serve.py --max-running-sessions: 1..256 registered tasks
# Problem-statement SHA-256 of the pilot instances (pilot pins; jobs 3299922 and 3300434).
PILOT_STATEMENT_SHA256 = {
    'astropy__astropy-12907': 'c01334ec1b21a089c650cf2e7b96ab974469076bf1260d23885799e1f0a7551f',
    'astropy__astropy-13236': '1f01b57c4e17057a2427b8ed897618f91782d22e98804b706826ba936a3d5ec4',
    'astropy__astropy-14096': '938971021e89cd882f6ea33d61202fe7aa0091d7be4748b100ddc7e164db90cd',
    'astropy__astropy-14309': '27b74ed7081579a5f310188d6ed6ac4da1d785c492b00052d605b7ba04d036b9',
    'astropy__astropy-14369': '005d3d61c2ffbfa601f53125a9fb5271a1a87979977b857204d9f3cf5190fba9',
    'astropy__astropy-14508': 'd08eb8c70acd4f003f7f477092385046092d36ff4e4155594b286795e7f9cd9f',
    'astropy__astropy-14539': 'deeb8ddb329b19eb6de7602db7ffbfb94a0887bedebd68ce49594609b6f4cf1a',
    'astropy__astropy-14995': '2eb1c507dd3ad67acc74872966e4bb5e14d571e20e67f246fc454ed9461f5745',
    'astropy__astropy-7336': '1c51cc8817f7b2f1feb043bdd2a54fcc4f89150dad82c60dbd43f68d1dcdf90d',
    'astropy__astropy-7671': '90a901a7b3d510c5fe9317b76fc9f83e883b577de167936d9e76ccbd69677f70',
}
# The statements of all 32 benchmark instances (verified-32.txt): the pilot pins and the other 22, as the
# benchmark runs' preflight checked them.
STATEMENT_SHA256 = {**PILOT_STATEMENT_SHA256,
    'astropy__astropy-13033': 'a295c83e68ff62290947b0796b75659b94621b7f2340d6f0055512d04be3f1d1',
    'astropy__astropy-13398': 'c26c8dc258dc5cb9dd34f30bdbaaed3f19a8ada137e0f19d81d7f57dc5f11bba',
    'astropy__astropy-13453': 'be929904c3062b54da2fa3ad7de338bd7f1f41a5aa48e64bec41f244d6e17c5a',
    'astropy__astropy-13579': '70a48b6e4c77b5abc12b1e750bbb70110b7215d19c9ea3550f2b7261be84021d',
    'astropy__astropy-13977': '53b72f8a5aed23f5d11650d7d4ae0e36c820cfade3c8ebcc7b718055a80d1cb2',
    'astropy__astropy-14182': '2f743a72613ab0e8e063ff7b7f0a233015ad12ba14992625d9f84b315fa885fd',
    'astropy__astropy-14365': '615252d5c44819451977588671ce903d3a21c41a8bf011738e2aee39218d56c6',
    'astropy__astropy-14598': '5edf1315e9ce360f72c2d3b23f08cdb974a758b64b13a6724236ca5d54beb9a9',
    'astropy__astropy-7166': '70c79d6da0284ce81f9e951dbab5c77d94bf8a27cd156f54169b424d75d11e43',
    'astropy__astropy-7606': 'd61965d0866c6a856d68dc4229f873e783399415a60d2a906d7847b777623ddd',
    'astropy__astropy-8707': 'f7420a19ba5b0c807396a84da376d8303e69ff66d0216c332adf099da589a714',
    'astropy__astropy-8872': '495ed57c2183a49ff489deab1cf1f0a4fafe7e6fad6d6af79a407426de354746',
    'django__django-10097': '2288319e7e72693551b8454b5c63f165f23dfb32bc6579ad03602d3cd311bdaa',
    'django__django-10554': '2f72fa6137f1d53983cff629a05e06c42a2ad477ea4557746c4688cbd6aeb724',
    'django__django-10880': 'd554ba8ef84e2e4dc22c291d413e4cb58070d5d730dae3aca383a4cdb77c20b2',
    'django__django-10914': 'ba101c6d80094fe56bc64d74dcb7efe033a6e0d43dd77e8f00e65806fdbcdb8b',
    'django__django-10973': 'bde1c4529fef5d030d1bd73a02e45bdbf9535a3c351aa9a43dba5c8249800271',
    'django__django-10999': 'efbac10cda6c1ef780205a8d9fc176b5489d1da8f76d9cbd366ef303052fd8d3',
    'django__django-11066': '2c4cbca71895396f6bac73ad9454105f56828f542278e2e927483f6cac599db8',
    'django__django-11087': '9bd93972a91db1ad0e44e65720dd0f36f000de6264e770ffdaaeb06996a338f3',
    'django__django-11095': '0564572e65282f6920d43eaaa77ae7b103eec95b435aa1ba183d7daa32b9e83d',
    'django__django-11099': 'b0526bc6d9268fe373bf624621c8d5f560fc6b0ffae7e64b460d84f6726f3ed5',
}


def task_bytes(task):
    return (json.dumps(task, indent=1) + '\n').encode()


def tasks_bytes(tasks):
    return (json.dumps(tasks, indent=1) + '\n').encode()


def session_groups(instances, cap):
    """The instances in order, split into the fewest groups of at most `cap`, sizes differing by at most one."""
    if not instances or not 1 <= cap <= MAX_SESSIONS:
        raise SystemExit('need at least one instance and 1 <= --max-sessions <= %d' % MAX_SESSIONS)
    count = -(-len(instances) // cap)
    size, extra = divmod(len(instances), count)
    groups, start = [], 0
    for k in range(count):
        end = start + size + (k < extra)
        groups.append(list(instances[start:end]))
        start = end
    return groups


def check_tasks(tasks):
    """serve.py's own rule: distinct ids, and no statement may contain another (routing would be ambiguous)."""
    ids = [t['instance_id'] for t in tasks]
    if len(set(ids)) != len(ids):
        raise SystemExit('instance ids repeat')
    statements = [t['problem_statement'] for t in tasks]
    for i, a in enumerate(statements):
        for j, b in enumerate(statements):
            if i != j and a in b:
                raise SystemExit('the statement of %s is contained in that of %s' % (ids[i], ids[j]))


def load_tasks(instances, dataset=DATASET, split='test', revision=REVISION):
    from datasets import load_dataset
    wanted = set(instances)
    rows = {r['instance_id']: r for r in load_dataset(dataset, split=split, revision=revision) if r['instance_id'] in wanted}
    missing = [i for i in instances if i not in rows]
    if missing:
        raise SystemExit('not in %s/%s: %s' % (dataset, split, ', '.join(missing)))
    return [dict(instance_id=i, problem_statement=rows[i]['problem_statement']) for i in instances]


def write(path, raw):
    with open(path, 'xb') as stream:   # never overwrite
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-sessions', type=int, default=5,
                        help='sessions per server (default 5: the validated cap at --kv-cache-gib 24)')
    parser.add_argument('--pool', action='store_true',
                        help='one tasks-pool.json with every instance, for serve.py --max-running-sessions')
    parser.add_argument('--dataset', default=DATASET)
    parser.add_argument('--split', default='test')
    parser.add_argument('--revision', default=REVISION, help='the dataset revision (default: the pinned commit)')
    parser.add_argument('instances', nargs='+')
    args = parser.parse_args()
    if args.pool:
        if not 1 <= len(args.instances) <= MAX_TASKS:
            raise SystemExit('a session pool registers 1 to %d instances' % MAX_TASKS)
        groups = []
    else:
        groups = session_groups(args.instances, args.max_sessions)
    tasks = load_tasks(args.instances, args.dataset, args.split, args.revision)
    check_tasks(tasks)
    for task in tasks:
        pin = STATEMENT_SHA256.get(task['instance_id'])
        if pin is not None and hashlib.sha256(task['problem_statement'].encode()).hexdigest() != pin:
            raise SystemExit('the problem statement of %s differs from its pin' % task['instance_id'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {t['instance_id']: t for t in tasks}
    report = dict(dataset=args.dataset, split=args.split, revision=args.revision, tasks={}, groups=[])
    for task in tasks:
        statement = task['problem_statement']
        report['tasks'][task['instance_id']] = dict(
            file_sha256=write(args.output_dir / ('task-%s.json' % task['instance_id']), task_bytes(task)),
            statement_chars=len(statement), statement_sha256=hashlib.sha256(statement.encode()).hexdigest(),
            statement_pin_checked=task['instance_id'] in STATEMENT_SHA256)
    for k, group in enumerate(groups):
        name = 'tasks-g%d.json' % k
        report['groups'].append(dict(file=name, instances=group,
                                     sha256=write(args.output_dir / name, tasks_bytes([by_id[i] for i in group]))))
    if args.pool:
        report['pool'] = dict(file='tasks-pool.json', instances=list(args.instances),
                              sha256=write(args.output_dir / 'tasks-pool.json', tasks_bytes(tasks)))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()

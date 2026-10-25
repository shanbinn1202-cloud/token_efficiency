"""Build serve.py task files for SWE-bench Verified instances, byte-for-byte as the pilot harness did.

For each instance it writes `task-<instance_id>.json` = {instance_id, problem_statement}, the statement verbatim
from the dataset row, encoded as json.dumps(task, indent=1) + newline. It then splits the instances, in the order
given, into the fewest server groups of at most --max-sessions (sizes differ by at most one) and writes one
`tasks-g<k>.json` list per group, encoded as json.dumps(tasks, indent=1) + newline. Start one server per group
and policy with `serve.py --task tasks-g<k>.json`, and run exactly that group's instances against it.

The multi-session SWE-bench pilots (Slurm jobs 3299922 and 3300434) used these encodings and this split; for their
ten Astropy instances the statement SHA-256 values are pinned below and checked here. Needs the `datasets` package
and the dataset (online, or cached with HF_DATASETS_OFFLINE=1).

With --pool (a session-pool server, `serve.py --max-running-sessions K`) it writes one `tasks-pool.json` with every
instance, in the order given (1 to 64; the order in which run_agents.sh refills its K agents), instead of groups.

Usage: build_tasks.py --output-dir DIR [--max-sessions 5 | --pool] INSTANCE_ID...
"""
import argparse
import hashlib
import json
from pathlib import Path

DATASET = 'princeton-nlp/SWE-Bench_Verified'   # the pilots found SWE-bench/SWE-bench_Verified identical for all ten
MAX_SESSIONS = 16                              # serve.py: 1..16 tasks per server
MAX_TASKS = 64                                 # serve.py --max-running-sessions: 1..64 registered tasks
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


def load_tasks(instances, dataset=DATASET, split='test'):
    from datasets import load_dataset
    wanted = set(instances)
    rows = {r['instance_id']: r for r in load_dataset(dataset, split=split) if r['instance_id'] in wanted}
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
    parser.add_argument('instances', nargs='+')
    args = parser.parse_args()
    if args.pool:
        if not 1 <= len(args.instances) <= MAX_TASKS:
            raise SystemExit('a session pool registers 1 to %d instances' % MAX_TASKS)
        groups = []
    else:
        groups = session_groups(args.instances, args.max_sessions)
    tasks = load_tasks(args.instances, args.dataset, args.split)
    check_tasks(tasks)
    for task in tasks:
        pin = PILOT_STATEMENT_SHA256.get(task['instance_id'])
        if pin is not None and hashlib.sha256(task['problem_statement'].encode()).hexdigest() != pin:
            raise SystemExit('the problem statement of %s differs from the pilot pin' % task['instance_id'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {t['instance_id']: t for t in tasks}
    report = dict(dataset=args.dataset, split=args.split, tasks={}, groups=[])
    for task in tasks:
        statement = task['problem_statement']
        report['tasks'][task['instance_id']] = dict(
            file_sha256=write(args.output_dir / ('task-%s.json' % task['instance_id']), task_bytes(task)),
            statement_chars=len(statement), statement_sha256=hashlib.sha256(statement.encode()).hexdigest(),
            pilot_pin_checked=task['instance_id'] in PILOT_STATEMENT_SHA256)
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

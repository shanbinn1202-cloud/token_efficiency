"""The task images of a Terminal-Bench 2.0 task set against the digests recorded in tasks.json.

Harbor 0.21.0 names each task's image by its tag (tasks.json "docker_image", from the task package) and starts it
with `docker compose up`, which pulls an image only when it is absent and otherwise uses the local one as it is.
tasks.json records, for the tasks of trial16 and smoke32, the digest that the agent host of the trial runs had
for that tag ("docker_image_digest": docker image inspect, RepoDigests); the other tasks have none.

- check: for every task of the set with a recorded digest whose image is present locally, the image's RepoDigests
  must contain <repository>@<digest>. A mismatch, or a docker error other than an absent image, refuses (exit 1,
  "status": "IMAGES_REFUSED"); otherwise "status": "IMAGES_OK" (exit 0). Absent images (Harbor pulls them by tag
  during the run, unchecked) and tasks without a recorded digest are listed, not refused.
- list: one line "<repository>@<digest> <repository>:<tag>" per task of the set with a recorded digest, for
  `docker pull` of the recorded digest and `docker tag` to the name Harbor uses.

Usage:
  python3 -I -B examples/terminalbench/check_images.py check --tasks trial16
  python3 -I -B examples/terminalbench/check_images.py list --tasks smoke32
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_job as M  # noqa: E402

DIGEST = re.compile(r'sha256:[0-9a-f]{64}\Z')
IMAGE = re.compile(r'([a-z0-9][a-z0-9._/-]{0,199}):([A-Za-z0-9_][A-Za-z0-9_.-]{0,127})\Z')
ABSENT = re.compile(r'(?i)no such image')


def pins(tasks, names):
    """{task: (repository, tag, digest or None)} of the set's tasks."""
    out = {}
    for name in names:
        t = tasks['tasks'][name]
        m = IMAGE.match(str(t.get('docker_image')))
        M.need(m, 'tasks.json: task %s: docker_image must be <repository>:<tag>' % name)
        digest = t.get('docker_image_digest')
        M.need(digest is None or DIGEST.match(str(digest)), 'tasks.json: task %s: docker_image_digest must be '
               'sha256:<64 hex>' % name)
        out[name] = (m.group(1), m.group(2), digest)
    return out


def docker_inspect(image, docker='docker'):
    """(returncode, stdout, stderr) of `docker image inspect --format '{{json .RepoDigests}}' <image>`."""
    try:
        done = subprocess.run([docker, 'image', 'inspect', '--format', '{{json .RepoDigests}}', image],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, '', '%s: %s' % (type(error).__name__, error)
    return done.returncode, done.stdout, done.stderr


def check(tasks, names, inspect=docker_inspect):
    """The check of every image of `names` (see the module docstring); returns the JSON record."""
    value = dict(tasks=len(names), checked=[], absent=[], unpinned=[], mismatched={}, errors={})
    for name, (repository, tag, digest) in pins(tasks, names).items():
        if digest is None:
            value['unpinned'].append(name)
            continue
        code, out, err = inspect('%s:%s' % (repository, tag))
        if code != 0:
            if ABSENT.search(err or ''):
                value['absent'].append(name)
            else:
                value['errors'][name] = (err or out or 'docker exit %d' % code).strip()[-300:]
            continue
        try:
            found = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            value['errors'][name] = 'unreadable RepoDigests: %r' % out[-200:]
            continue
        if not isinstance(found, list) or '%s@%s' % (repository, digest) not in found:
            value['mismatched'][name] = dict(expected='%s@%s' % (repository, digest), found=found)
        else:
            value['checked'].append(name)
    refused = bool(value['mismatched'] or value['errors'])
    value['status'] = 'IMAGES_REFUSED' if refused else 'IMAGES_OK'
    return value


def main(argv=None, inspect=docker_inspect):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('check', 'list'))
    p.add_argument('--tasks', required=True, choices=M.TASK_SETS)
    a = p.parse_args(argv)
    try:
        tasks = M.load(HERE / M.TASKS_FILE)
        M.check_tasks(tasks)
        names = M.task_set(tasks, a.tasks)
        if a.mode == 'list':
            for repository, tag, digest in pins(tasks, names).values():
                if digest is not None:
                    print('%s@%s %s:%s' % (repository, digest, repository, tag))
            return 0
        value = check(tasks, names, inspect)
    except M.JobError as error:
        print(json.dumps(dict(status='IMAGES_REFUSED', error=str(error))), file=sys.stderr)
        return 2
    value['task_set'] = a.tasks
    print(json.dumps(value, sort_keys=True))
    return 0 if value['status'] == 'IMAGES_OK' else 1


if __name__ == '__main__':
    sys.exit(main())

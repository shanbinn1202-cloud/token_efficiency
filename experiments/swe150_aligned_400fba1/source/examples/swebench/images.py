"""The pinned SWE-bench Verified instance images: check that the local images are the pinned digests.

The agents (Mini-SWE-Agent 2.4.2 through the prototype's runner) and the evaluator (SWE-bench at commit 02e7a74, version 5.0.2) run each instance in
docker.io/swebench/sweb.eval.x86_64.<instance id with __ as _1776_, lowercase>:latest. That tag can move on Docker Hub,
so verified-32.txt pins the image of each of the 32 benchmark instances by digest: one line
"<instance_id> <image>@sha256:<digest>" per instance, in run order, the digests of the images of the validated runs
(their RepoDigests). Before anything starts, run_agents.sh and evaluate.sh run

  python3 -I -B images.py <pins file> <out file> <instance_id>...

For every instance it needs a pin, and the local tag <image>:latest must be the pinned image (docker image inspect:
the same image id as <image>@sha256:<digest>). It then writes the instances' pin lines to <out file>, which must not
exist; otherwise it prints, for every missing or different image, the two commands that place the pinned one:

  docker pull <image>@sha256:<digest>
  docker tag <image>@sha256:<digest> <image>:latest

Exit 0: every image is the pinned one; 2: usage, the pins file, or an instance without a pin; 3: an image is missing
or is not the pinned one. Only the standard library and the docker CLI are used; nothing is pulled or changed.
"""
import re
import subprocess
import sys
from pathlib import Path

PIN = re.compile(r'([A-Za-z0-9_-]{1,120}) (\S+)@(sha256:[0-9a-f]{64})')


def image(instance):
    """The image name the agents and the evaluator use for an instance (without its tag)."""
    return ('docker.io/swebench/sweb.eval.x86_64.' + instance.replace('__', '_1776_')).lower()


def read_pins(path):
    """{instance_id: '<image>@sha256:<digest>'} of a pins file, in file order."""
    pins = {}
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        m = PIN.fullmatch(line)
        if not m or m.group(2) != image(m.group(1)) or m.group(1) in pins:
            raise ValueError('%s line %d: not "<instance_id> <its image>@sha256:<digest>" (or a repeated instance)'
                             % (path, number))
        pins[m.group(1)] = '%s@%s' % (m.group(2), m.group(3))
    return pins


def image_id(reference):
    """The local image id of a reference, or None when docker has no such image."""
    run = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', reference], capture_output=True,
                         text=True, check=False)
    return run.stdout.strip() or None if run.returncode == 0 else None


def check(pins, instances, inspect=image_id):
    """(pin lines of the instances, problems): a problem for every image that is missing or not the pinned one."""
    missing = [i for i in instances if i not in pins]
    if missing:
        raise ValueError('no pinned image for %s (IMAGE_PINS names another pins file)' % ', '.join(missing))
    lines, problems = [], []
    for instance in instances:
        reference = pins[instance]
        name = reference.split('@')[0]
        pinned, latest = inspect(reference), inspect(name + ':latest')
        if pinned is None or latest != pinned:
            problems.append('%s: %s:latest is %s, not the pinned image; place it with\n  docker pull %s\n'
                            '  docker tag %s %s:latest' % (instance, name, 'missing' if latest is None else latest,
                                                           reference, reference, name))
        lines.append('%s %s' % (instance, reference))
    return lines, problems


def main(argv):
    if len(argv) < 4:
        print('usage: images.py <pins file> <out file> <instance_id>...', file=sys.stderr)
        return 2
    try:
        lines, problems = check(read_pins(argv[1]), argv[3:])
    except (OSError, ValueError) as error:
        print('images.py: %s' % error, file=sys.stderr)
        return 2
    if problems:
        print('\n'.join(problems), file=sys.stderr)
        return 3
    with open(argv[2], 'x', encoding='utf-8', newline='\n') as out:
        out.write(''.join(line + '\n' for line in lines))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

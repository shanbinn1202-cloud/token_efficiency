#!/usr/bin/env python3
"""Scan the harness snapshot before a commit: secrets, private text in fixtures, account and host data, sizes.

  python -I -B harnesses/scan_snapshot.py                 the snapshot directories under harnesses/
  python -I -B harnesses/scan_snapshot.py --live NAME...  the live folders' selection (sync_from_live.select)
  python -I -B harnesses/scan_snapshot.py --text FILE...  only the blocked-terms check, on these files (commit messages)

Read-only; standard library only. It prints a report and exits 1 when it finds secret material (a private key
block, a known token format, a credential file, a credential assignment with a literal value) or a blocked
term (sync_from_live.blocked_terms: the terms are held as SHA-256 hashes, not in the repository), 0 otherwise. The other kinds are listed for a decision, not refused:
  private text: JSON / JSONL strings and text-fixture lines of LONG_TEXT characters or more that read as natural
    language or code (the project rule: fixtures hold counts, not trajectories or prompts); each case is listed
    with its key and the first characters; strings under keys named like our own notes (about, why, ...) are
    counted separately;
  account and host data: e-mail addresses, cluster account and user names in paths, Windows user folders, IPv4
    addresses, host names, SSH host keys; every kind with a count;
  sizes: the total and the largest files.
Secret values are masked in the report.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sync_from_live  # noqa: E402

LONG_TEXT = 300
TEXT_SUFFIXES = {'.py', '.ps1', '.in', '.sh', '.md', '.txt', '.json', '.jsonl', '.yaml', '.yml', '.jinja', '.env',
                 '.log', '.cfg', '.toml', ''}
NOTE_KEYS = re.compile(r'(about|why|reason|note|notes|description|decision|comment|summary|doc|help|rule|_about)$',
                       re.I)
SECRET_PATTERNS = (
    ('private key block', re.compile(r'-----BEGIN [A-Z0-9 ]*PRIVATE KEY( BLOCK)?-----')),
    ('GitHub token', re.compile(r'\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b')),
    ('Hugging Face token', re.compile(r'\bhf_[A-Za-z0-9]{30,}\b')),
    ('sk- API key', re.compile(r'\bsk-(ant-)?[A-Za-z0-9_-]{20,}\b')),
    ('AWS access key id', re.compile(r'\b(AKIA|ASIA)[0-9A-Z]{16}\b')),
    ('AWS secret name', re.compile(r'aws_secret_access_key|aws_session_token', re.I)),
    ('Slack token', re.compile(r'\bxox[abprs]-[A-Za-z0-9-]{10,}')),
    ('Bearer token', re.compile(r'Bearer\s+[A-Za-z0-9._~+/-]{20,}=*')),
    ('credential assignment', re.compile(
        r'''(?i)\b(password|passwd|pwd|api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)'''
        r'''\b\s*["']?\s*[:=]\s*["']([^"'\s$<>{}()]{8,})["']''')),
)
# Secret-pattern hits reviewed as harmless: (file label, kind) -> why. Listed in the report, not counted as secrets.
ALLOWED = {
    ('perf-tb16-strip/tasks/tasks-pool.json', 'AWS secret name'):
        'a public Terminal-Bench 2.0 task statement names the variable (replace its value with a placeholder); no value',
}
SECRET_FILES = re.compile(r'(?i)^(auth\.json|credentials(\.json)?|\.netrc|_netrc|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|'
                          r'.*\.(pem|key|p12|pfx|ppk|kdbx))$')
SSH_PATH = re.compile(r'[~\w/\\.-]*[\\/]\.ssh[\\/][\w.-]+')
EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
CLUSTER_PATH = re.compile(r'/(?:u|home|projects|work/nvme|work/hdd|scratch)/([A-Za-z0-9_.-]+)(?:/([A-Za-z0-9_.-]+))?')
WIN_USER = re.compile(r'(?i)[A-Z]:\\{1,2}Users\\{1,2}([A-Za-z0-9_.-]+)')
IPV4 = re.compile(r'(?<![\d.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\d.])')
HOST_KEY = re.compile(r'\b(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp\d+)\s+(AAAA[A-Za-z0-9+/=]{40,})')
SLURM_ACCOUNT = re.compile(r'(?:--account[= ]|#SBATCH\s+-A\s+|"account"\s*:\s*")([A-Za-z0-9_-]+)')
HOST_NAME = re.compile(r'\b(ec2-\d+-\d+-\d+-\d+\.[a-z0-9.-]*amazonaws\.com|[a-z0-9-]+\.(?:delta\.)?ncsa\.illinois\.edu)\b')
# Blocked terms: sync_from_live.blocked_terms (hashed, so the terms themselves are not in the repository).
blocked_terms = sync_from_live.blocked_terms
# Text of a model conversation: a fixture with these holds a trajectory or a prompt, not counts.
TRAJECTORY = re.compile(r'"role"\s*:\s*"(?:assistant|user|system)"|THOUGHT:|<pr_description>|"reply"\s*:\s*"')
TOKEN_ARRAY = 200           # a JSON list of this many integers or more: token ids (a prompt or a reply)


def mask(value):
    return '%s...(%d chars)' % (value[:4], len(value))


def files_of(roots):
    """[(label, path)] of every file to scan."""
    out = []
    for label, root, selected in roots:
        if selected is not None:
            out += [('%s/%s' % (label, rel), path) for rel, (path, _) in sorted(selected.items())]
            continue
        for directory, dirs, files in os.walk(root):
            dirs.sort()
            for name in sorted(files):
                path = Path(directory) / name
                out.append(('%s/%s' % (label, path.relative_to(root).as_posix()), path))
    return out


def looks_like_prose_or_code(text):
    words = re.findall(r'[A-Za-z]{2,}', text)
    spaces = text.count(' ')
    return len(words) >= 25 and spaces >= 20


def token_arrays(value, key=''):
    """(key, length) of every list of TOKEN_ARRAY integers or more."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from token_arrays(v, str(k))
    elif isinstance(value, list):
        if len(value) >= TOKEN_ARRAY and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            yield key, len(value)
        else:
            for v in value:
                yield from token_arrays(v, key)


def json_strings(value, key=''):
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and len(k) >= LONG_TEXT:
                yield '<key>', k
            yield from json_strings(v, str(k))
    elif isinstance(value, list):
        for v in value:
            yield from json_strings(v, key)
    elif isinstance(value, str):
        yield key, value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--live', nargs='+', metavar='NAME', help='scan the live selection of these harnesses')
    parser.add_argument('--root', type=Path, default=HERE, help='the snapshot root (default: this directory)')
    parser.add_argument('--top', type=int, default=12, help='largest files to list')
    parser.add_argument('--text', nargs='+', type=Path, metavar='FILE', help='only the blocked-terms check, on these files')
    args = parser.parse_args()
    if args.text:
        bad = 0
        for path in args.text:
            hits = blocked_terms(path.name, path.read_text('utf-8', 'replace'))
            bad += sum(hits.values())
            print('%s: %s' % (path, dict(hits) if hits else 'no blocked term'))
        print('BLOCKED_TERMS_%s' % ('FAIL' if bad else 'PASS'))
        return 1 if bad else 0
    if args.live:
        roots = [(n, sync_from_live.SOURCE_ROOT / n, sync_from_live.select(n, sync_from_live.SOURCE_ROOT / n)[0])
                 for n in args.live]
    else:
        roots = [(n, args.root / n, None) for n in sync_from_live.HARNESSES if (args.root / n).is_dir()]
        roots += [('harnesses', args.root, {p.name: (p, p.stat().st_size) for p in sorted(args.root.iterdir())
                                             if p.is_file()})]
    files = files_of(roots)
    secrets, ssh_paths = [], collections.Counter()
    long_cases, note_long = [], collections.Counter()
    account = {k: collections.Counter() for k in ('e-mail address', 'Slurm account', 'cluster user/project path name',
                                                     'Windows user folder', 'IPv4 address', 'host name',
                                                     'SSH host key')}
    first = {}                  # (kind, value): the first file it appears in
    trajectory, tokens = collections.Counter(), []
    blocked = {}                # label: {tag: count}
    sizes = []
    for label, path in files:
        raw = path.read_bytes()
        sizes.append((len(raw), label))
        name = path.name
        if SECRET_FILES.match(name):
            secrets.append((label, 'credential file name', name))
        if path.suffix.lower() not in TEXT_SUFFIXES and b'\0' in raw[:4096]:
            continue
        text = raw.decode('utf-8', 'replace')
        hits = blocked_terms(label, text)
        if hits:
            blocked[label] = hits
        if path.resolve() == Path(__file__).resolve():
            continue                # its own patterns
        for kind, pattern in SECRET_PATTERNS:
            for m in pattern.finditer(text):
                value = m.group(m.lastindex) if m.lastindex else m.group(0)
                if kind == 'credential assignment' and re.fullmatch(r'[a-z_]+(\.[a-z_]+)*', value):
                    continue    # a variable or attribute name, not a literal secret
                secrets.append((label, kind, mask(value)))
        for m in SSH_PATH.finditer(text):
            ssh_paths[m.group(0)] += 1
        for m in EMAIL.finditer(text):
            if not m.group(0).lower().endswith(('example.com', 'example.invalid', 'example.org', 'noreply.github.com')):
                account['e-mail address'][m.group(0)] += 1
        for m in SLURM_ACCOUNT.finditer(text):
            account['Slurm account'][m.group(1)] += 1
        for m in CLUSTER_PATH.finditer(text):
            for part in m.groups():
                if part and part not in ('bin', 'lib', 'usr', 'local', 'share', 'tmp', 'models', 'hf_cache', 'ubuntu'):
                    account['cluster user/project path name'][part] += 1
            if m.group(0).startswith('/home/ubuntu'):
                account['cluster user/project path name']['ubuntu (agent host)'] += 1
        for m in WIN_USER.finditer(text):
            account['Windows user folder'][m.group(1)] += 1
        for m in IPV4.finditer(text):
            octets = [int(x) for x in m.groups()]
            if all(o <= 255 for o in octets) and m.group(0) not in ('127.0.0.1', '0.0.0.0') and octets[0] != 0:
                account['IPv4 address'][m.group(0)] += 1
        for m in HOST_KEY.finditer(text):
            account['SSH host key'][m.group(1) + ' ' + mask(m.group(2))] += 1
        for m in HOST_NAME.finditer(text):
            account['host name'][m.group(1)] += 1
        for kind, counter in account.items():
            for value in counter:
                first.setdefault((kind, value), label)
        n = len(TRAJECTORY.findall(text))
        if n:
            trajectory[label] += n
        suffix = path.suffix.lower()
        if suffix in ('.json', '.jsonl'):
            docs = []
            try:
                docs = [json.loads(text)]
            except ValueError:
                for line in text.splitlines():
                    try:
                        docs.append(json.loads(line))
                    except ValueError:
                        pass
            for doc in docs:
                tokens += [(label, key, length) for key, length in token_arrays(doc)]
                for key, value in json_strings(doc):
                    if len(value) >= LONG_TEXT and looks_like_prose_or_code(value):
                        if NOTE_KEYS.search(key):
                            note_long[label] += 1
                        else:
                            long_cases.append((label, key, len(value), value[:110].replace('\n', '\\n')))
        elif suffix in ('.txt', '.log') or '/fixtures/' in label:
            for number, line in enumerate(text.splitlines(), 1):
                if len(line) >= LONG_TEXT and looks_like_prose_or_code(line):
                    long_cases.append((label, 'line %d' % number, len(line), line[:110]))
    total = sum(s for s, _ in sizes)
    print('SCAN of %d files, %d bytes (%s)' % (len(files), total, 'live selection' if args.live else str(args.root)))
    allowed = [s for s in secrets if (s[0], s[1]) in ALLOWED]
    secrets = [s for s in secrets if (s[0], s[1]) not in ALLOWED]
    print('\n== 1. secrets: %d finding(s) (+ %d reviewed and allowed)' % (len(secrets), len(allowed)))
    for label, kind, value in secrets:
        print('  %-28s %s  %s' % (kind, label, value))
    for label, kind, value in allowed:
        print('  allowed: %-19s %s  %s: %s' % (kind, label, value, ALLOWED[(label, kind)]))
    print('  .ssh path references (paths to key files, not key material): %d distinct, %d mentions'
          % (len(ssh_paths), sum(ssh_paths.values())))
    for value, count in ssh_paths.most_common():
        print('    %4d  %s' % (count, value))
    print('\n== 2. long natural-language / code strings in JSON and text fixtures: %d case(s) '
          '(+ %d long note strings under about/why-like keys in %d files)'
          % (len(long_cases), sum(note_long.values()), len(note_long)))
    by_file = collections.defaultdict(list)
    for label, key, length, head in long_cases:
        by_file[label].append((key, length, head))
    for label in sorted(by_file):
        cases = by_file[label]
        print('  %s: %d case(s), longest %d chars' % (label, len(cases), max(c[1] for c in cases)))
        for key, length, head in cases[:4]:
            print('    [%s] %d chars: %s' % (key, length, head))
        if len(cases) > 4:
            print('    ... %d more' % (len(cases) - 4))
    print('  conversation markers (role / THOUGHT: / <pr_description> / "reply"): %d files' % len(trajectory))
    for label, count in sorted(trajectory.items()):
        print('    %5d  %s' % (count, label))
    print('  token-id arrays (JSON lists of %d+ integers): %d' % (TOKEN_ARRAY, len(tokens)))
    for label, key, length in tokens[:20]:
        print('    %s [%s] %d ids' % (label, key, length))
    print('\n== 3. account and host data (kinds and counts)')
    for kind, counter in account.items():
        print('  %-32s %d distinct, %d mentions' % (kind, len(counter), sum(counter.values())))
        for value, count in counter.most_common(12):
            print('    %5d  %s   (first in %s)' % (count, value, first[(kind, value)]))
    print('\n== 4. sizes: %d files, %d bytes; largest:' % (len(sizes), total))
    for size, label in sorted(sizes, reverse=True)[:args.top]:
        print('  %10d  %s' % (size, label))
    print('\n== 5. blocked terms (hashed list; must be 0): %d in %d files'
          % (sum(sum(h.values()) for h in blocked.values()), len(blocked)))
    for label, hits in sorted(blocked.items()):
        print('  %s: %s' % (label, ', '.join('%s x%d' % kv for kv in sorted(hits.items()))))
    print('\nSCAN_%s' % ('SECRETS_FOUND' if secrets else 'NO_SECRETS'))
    print('BLOCKED_TERMS_%s' % ('FAIL' if blocked else 'PASS'))
    return 1 if secrets or blocked else 0


if __name__ == '__main__':
    sys.exit(main())

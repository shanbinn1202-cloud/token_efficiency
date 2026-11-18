#!/usr/bin/env python3
"""Refresh harnesses/<name>/ from the live harness folders: a filtered snapshot.

  python -I -B harnesses/sync_from_live.py                    every harness of HARNESSES
  python -I -B harnesses/sync_from_live.py perf-mab           only the harnesses named
  python -I -B harnesses/sync_from_live.py --dry-run          report what would change; write nothing
  python -I -B harnesses/sync_from_live.py --list perf-mab    also print every selected file and its size
  python -I -B harnesses/sync_from_live.py --note "..." NAME  store a note on the snapshot's state

The live folders (SOURCE_ROOT/<name>/) stay the canonical working copies (their scripts, READY files and the
agents use their absolute paths); this script only reads them. For each harness named it mirrors the selected
files into harnesses/<name>/ (copies new and changed files byte for byte, removes files the selection no longer
has) and rewrites that harness's entry of harnesses/SNAPSHOT.json. It never writes outside harnesses/<name>/ and
SNAPSHOT.json. Standard library only; run it with `python -I -B`.

Edits: every selected UTF-8 text file passes through the exact-phrase edits of EDIT_RULES, a local edit list
kept outside the repository. An edited file no longer equals its live bytes; SNAPSHOT.json lists each one
(`edited`: path -> edits). The run stops (exit 1, nothing written) without the edit list, when an edit breaks a
JSON file, or when blocked_terms() finds a blocked term in any file.

Selection (the filters; see the README):
  directories left out at any depth: EXCLUDE_DIRS (validation/ and dist/ hold run outputs and sealed payloads);
  files left out by name: EXCLUDE_FILES (Git bundles, bytecode, *-PROGRESS.md working notes, scratch copies);
  *.log only under LOG_KEPT_DIRS (fixtures/: recorded log excerpts the harness tests read);
  HELD_BACK: single files left out after the scan (scan_snapshot.py);
  every other file is copied. A selected file over MAX_BYTES that is not in ALLOW_LARGE stops the run before
  anything is written (exit 1), as does a missing harness folder.
"""
import argparse
import datetime
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path(r'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages')
EDIT_RULES = Path(r'D:\Remote-RA-temp\scratch\harnesses-1110\sync-edits.json')
# The release smoke matrix (the maintainer's decision of 2026-11-10: only these four).
HARNESSES = ('perf-swe32', 'perf-gemma-swe32', 'perf-tb16-strip', 'perf-mab')
MAX_BYTES = 2 * 1024 * 1024
ALLOW_LARGE = {}            # 'harness/relative/path': why it may exceed MAX_BYTES
EXCLUDE_DIRS = {
    'validation': 'local validation logs (run outputs)',
    'dist': 'build output (sealed payloads; build.py rebuilds them)',
    '__pycache__': 'bytecode',
    '.pytest_cache': 'test cache',
    '.git': 'Git metadata',
}
EXCLUDE_FILES = (
    ('*.bundle', 'sealed Git bundle of the pinned runtime (build.py rebuilds it from the commit)'),
    ('*.pyc', 'bytecode'),
    ('*.pyo', 'bytecode'),
    ('*-PROGRESS.md', "agents' resumable working notes (in-flight state; STATUS.md and the READY files hold the result)"),
    ('*.bak', 'scratch copy'),
    ('*.orig', 'scratch copy'),
    ('*.rej', 'scratch copy'),
    ('*.tmp', 'scratch copy'),
    ('*.swp', 'scratch copy'),
    ('*~', 'scratch copy'),
    ('* - Copy*', 'scratch copy'),
)
LOG_KEPT_DIRS = ('fixtures',)
LOG_REASON = 'run log outside fixtures/'
# 'harness/relative/path': why it stays out (scan_snapshot.py; the live copy keeps it).
HELD_BACK = {
    'perf-gemma-swe32/fixtures/gemma-render-job3429830.json':
        "job 3429830's first three calls with the model's replies and the prompt / reply token ids (a trajectory "
        "sample, against the counts-only rule for fixtures; test_real_tokenizer.GemmaJoinRender reads it)",
}
SNAPSHOT = 'SNAPSHOT.json'


def utc(seconds=None):
    moment = datetime.datetime.now(datetime.timezone.utc) if seconds is None else \
        datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc)
    return moment.strftime('%Y-%m-%dT%H:%M:%SZ')


def excluded_file(name, top):
    """The reason a file of this name (in top-level directory `top`, '' at the harness root) is left out, or None."""
    for pattern, reason in EXCLUDE_FILES:
        if fnmatch.fnmatchcase(name, pattern):
            return reason
    if name.lower().endswith('.log') and top not in LOG_KEPT_DIRS:
        return LOG_REASON
    return None


def select(harness, source):
    """(selected {relative posix path: (path, bytes)}, excluded {reason: [files, bytes]}, too large [(rel, bytes)])."""
    selected, excluded, large = {}, {}, []

    def leave(reason, size):
        entry = excluded.setdefault(reason, [0, 0])
        entry[0] += 1
        entry[1] += size

    for directory, dirs, files in os.walk(source):
        here = Path(directory)
        for name in list(dirs):
            if name in EXCLUDE_DIRS or (here / name).is_symlink():
                dirs.remove(name)
                for sub, _, inner in os.walk(here / name):
                    for f in inner:
                        leave(EXCLUDE_DIRS.get(name, 'symlinked directory'), (Path(sub) / f).stat().st_size)
        dirs.sort()
        for name in sorted(files):
            path = here / name
            rel = path.relative_to(source).as_posix()
            size = path.stat().st_size
            top = rel.split('/')[0] if '/' in rel else ''
            reason = 'symlink' if path.is_symlink() else excluded_file(name, top)
            if reason is None and harness + '/' + rel in HELD_BACK:
                reason = 'held back: ' + HELD_BACK[harness + '/' + rel]
            if reason:
                leave(reason, size)
                continue
            if size > MAX_BYTES and harness + '/' + rel not in ALLOW_LARGE:
                large.append((rel, size))
            selected[rel] = (path, size)
    return selected, excluded, large


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# Terms that must not appear in the snapshot, as SHA-256 of the lowercase word or word pair (the terms themselves
# are not in the repository). BLOCKED_TERMS fail anywhere; BLOCKED_OUTSIDE_PUBLIC and BLOCKED_WORDS (checked on
# words written in lower case or capitalized, not after a hyphen) fail outside PUBLIC_TEXT_FILES.
BLOCKED_TERMS = {
    'd394b4ccfaad0ed15468ba78084ff5f5cac82be6b69f6be131ced3504b5d7e8e': 'term #1',
    '01380046d1d8e2162c4da19af3749b5078881de44f2bf7fc354b12649274bed8': 'term #2',
    '35dde764870012d02d51ad4d34f350ded715c6c0872be44ce1c3035c3b86cbbb': 'term #3',
    '2ec79f89b431914d516564d688bdbecb0f350943e19b0fcd05a96db2f6b7dd70': 'term #4',
    '893da6004c5057faad6a7c96a0ec0e5ee27845ecea7e284a72967e1e8a434e60': 'term #5',
    'f0aefa48d73cd598dcaef8cc68380b8380e62ba97f9d9ac9864b429c9b01cc3f': 'term #6',
    '7f629881370524f0a18d581f532ce03ca260e22029b56436f3bb3904fa5351d1': 'term #7',
    'c106b62292f7648be9516ca73cbb747ea0997d5f92f0cfe4f44e02aafaf4ad7e': 'term #8',
    '2acdfeaf00c9564353fb6a66bef454c176ae391be28ec1807d49666f86e9047c': 'term #9',
    '937a9a792fb29d1b468800773293b68f95c7b1201c6cafa2f638b7e30eddc60d': 'term #10',
}
BLOCKED_OUTSIDE_PUBLIC = {'17c1532ca6cff8f6a3a8200028af6c2580bf37f39e10cb0966e8a573e3b24a1f': 'term #11'}
BLOCKED_WORDS = {
    '73c9c98668a34c54d131ff609d0bf129068d1b5ed3efd7cdfe753f909596456c': 'word #1',
    '354cbfc814262a7a81d343f7d6ebc4adfb4266e5081a78f6a19ff16802dce8b2': 'word #2',
    '5a7cf4ac138054a82f60f73e4d078b4bcb964184c9a97c19375b4d78770b99ba': 'word #3',
    '372f7e2fd2d01ce2a1d71dc072acbba4c6fd25a1087cd7f153f4ec0ce37e1ede': 'word #4',
    '2cda85e504e155f30a40f1f1cc4e83f3f8ec178033f6b6170cc9381f5e4f9d3d': 'word #5',
}
PUBLIC_TEXT_FILES = ('*/tasks/tasks-pool.json', '*/tasks/tasks-pool-64.json',   # public benchmark task statements
                     '*/tasks/tasks-pool-h2.json',   # the 64 pool's last 32 entries, byte for byte
                     '*/tasks/tasks-pool-150.json')  # the first 150 SWE-bench Verified statements (first 64 = the 64 pool)
WORD = re.compile(r'[a-z]+')
PLAIN_WORD = re.compile(r'(?<![-\w])([A-Za-z][a-z]*)\b')
_DIGESTS = {}


def _digest(token):
    if token not in _DIGESTS:
        _DIGESTS[token] = hashlib.sha256(token.encode('utf-8')).hexdigest()
    return _DIGESTS[token]


def blocked_terms(label, text):
    """{tag: count} of the blocked terms in `text` (file `label`, harness/relative path)."""
    words = WORD.findall(text.lower())
    hits = {}
    public = any(fnmatch.fnmatchcase(label, glob) for glob in PUBLIC_TEXT_FILES)
    for i, word in enumerate(words):
        tokens = [word] + ([word[:-1]] if word.endswith('s') else [])
        if i + 1 < len(words):
            tokens.append(word + ' ' + words[i + 1])
        for token in tokens:
            tag = BLOCKED_TERMS.get(_digest(token)) or (None if public else BLOCKED_OUTSIDE_PUBLIC.get(_digest(token)))
            if tag:
                hits[tag] = hits.get(tag, 0) + 1
    if not public:
        for match in PLAIN_WORD.finditer(text):
            tag = BLOCKED_WORDS.get(_digest(match.group(1).lower()))
            if tag:
                hits[tag] = hits.get(tag, 0) + 1
    return hits


def load_rules(path):
    """The ordered exact-phrase edits: [(old, new, skip globs)]."""
    data = json.loads(Path(path).read_text('utf-8'))
    edits = [(e['old'], e['new'], tuple(e.get('skip', ()))) for e in data['edits']]
    if any(not old for old, _, _ in edits):
        raise ValueError('an edit with an empty old text')
    return edits


def apply_edits(label, raw, edits, used=None):
    """(bytes to write, edits made) for the file `label` (harness/relative path); `used` counts each edit's hits."""
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        return raw, 0
    made = 0
    for index, (old, new, skip) in enumerate(edits):
        if any(fnmatch.fnmatchcase(label, glob) for glob in skip):
            continue
        n = text.count(old)
        if n:
            text = text.replace(old, new)
            made += n
            if used is not None:
                used[index] = used.get(index, 0) + n
    if not made:
        return raw, 0
    if label.endswith('.json'):
        try:
            json.loads(raw.decode('utf-8'))
        except ValueError:
            pass                    # not JSON before: nothing to keep valid
        else:
            json.loads(text)        # ValueError: the edits broke the JSON
    return text.encode('utf-8'), made


def plan(content, target):
    """(added, changed, removed) relative paths for mirroring `content` {rel: bytes} into `target`."""
    present = set()
    if target.is_dir():
        for directory, _, files in os.walk(target):
            for name in files:
                present.add((Path(directory) / name).relative_to(target).as_posix())
    added = sorted(set(content) - present)
    removed = sorted(present - set(content))
    changed = sorted(rel for rel in set(content) & present
                     if (target / rel).stat().st_size != len(content[rel])
                     or (target / rel).read_bytes() != content[rel])
    return added, changed, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='*', help='harnesses to refresh (default: all of HARNESSES)')
    parser.add_argument('--source', type=Path, default=SOURCE_ROOT, help='the live root (default SOURCE_ROOT)')
    parser.add_argument('--dest', type=Path, default=HERE, help='the snapshot root (default: this directory)')
    parser.add_argument('--dry-run', action='store_true', help='report the changes; write nothing')
    parser.add_argument('--list', action='store_true', help='print every selected file and its size')
    parser.add_argument('--note', help="a note on the snapshot's state, stored in SNAPSHOT.json (default: keep the "
                        "previous note)")
    parser.add_argument('--rules', type=Path, default=EDIT_RULES, help='the edit list (default EDIT_RULES)')
    args = parser.parse_args()
    names = args.names or list(HARNESSES)
    unknown = [n for n in names if n not in HARNESSES]
    if unknown:
        parser.error('not in HARNESSES: %s' % ', '.join(unknown))
    source_root, dest_root = args.source.resolve(), args.dest.resolve()
    if dest_root.is_relative_to(source_root) or source_root.is_relative_to(dest_root):
        parser.error('the snapshot root and the live root must not contain each other')
    work, problems, used = [], [], {}
    try:
        rules = load_rules(args.rules)
    except (OSError, ValueError, KeyError, TypeError) as error:
        rules = None
        problems.append('no usable edit list %s: %s' % (args.rules, error))
    for name in names:
        source = source_root / name
        if not source.is_dir():
            problems.append('%s: no live folder %s' % (name, source))
            continue
        selected, excluded, large = select(name, source)
        problems += ['%s/%s: %d bytes > MAX_BYTES %d (list it in ALLOW_LARGE to keep it)' % (name, rel, size, MAX_BYTES)
                     for rel, size in large]
        content, edited = {}, {}
        if rules is not None:
            for rel, (path, _) in selected.items():
                try:
                    content[rel], made = apply_edits(name + '/' + rel, path.read_bytes(), rules, used)
                except ValueError as error:
                    problems.append('%s/%s: the edits broke the JSON: %s' % (name, rel, error))
                    continue
                if made:
                    edited[rel] = made
                hits = blocked_terms(name + '/' + rel, content[rel].decode('utf-8', 'replace'))
                if hits:
                    problems.append('%s/%s: blocked term after the edits: %s' % (name, rel, hits))
        work.append((name, source, selected, excluded, content, edited))
    if rules is not None and set(args.names or HARNESSES) == set(HARNESSES):
        for index, (old, _, _) in enumerate(rules):
            if index not in used:
                print('note: edit %d matches nothing (%d characters; stale?)' % (index, len(old)))
    if problems:
        print('SYNC_REFUSED\n' + '\n'.join(problems))
        return 1
    snapshot_path = dest_root / SNAPSHOT
    snapshot = json.loads(snapshot_path.read_text('utf-8')) if snapshot_path.is_file() else {}
    snapshot['about'] = ('The state of each snapshot directory: written by sync_from_live.py, one entry per '
                         'harness, rewritten when that harness is refreshed. listing_sha256 is the SHA-256 of the '
                         'sha256sum-style listing ("<sha256>  <path>" per file, sorted by path). edited: the files '
                         'changed by the local edit list (path -> edits); they no longer equal the live bytes, every '
                         'other file does.')
    snapshot['source_root'] = str(source_root)
    entries = snapshot.setdefault('harnesses', {})
    totals = dict(files=0, bytes=0, excluded_files=0, excluded_bytes=0)
    for name, source, selected, excluded, content, edited in work:
        target = dest_root / name
        added, changed, removed = plan(content, target)
        files, size = len(content), sum(len(b) for b in content.values())
        ex_files, ex_bytes = sum(v[0] for v in excluded.values()), sum(v[1] for v in excluded.values())
        for key, value in zip(totals, (files, size, ex_files, ex_bytes)):
            totals[key] += value
        print('%-18s %4d files %10d bytes | excluded %4d files %10d bytes | +%d ~%d -%d'
              % (name, files, size, ex_files, ex_bytes, len(added), len(changed), len(removed)))
        for reason, (count, nbytes) in sorted(excluded.items()):
            print('    excluded %4d files %10d bytes  %s' % (count, nbytes, reason))
        print('    edited   %4d files, %d edits' % (len(edited), sum(edited.values())))
        if args.list:
            for rel in sorted(content):
                print('    %10d  %s%s' % (len(content[rel]), rel, '  (edited %d)' % edited[rel] if rel in edited else ''))
        for label, paths in (('add', added), ('change', changed), ('remove', removed)):
            for rel in paths:
                print('    %-6s %s' % (label, rel))
        if args.dry_run:
            continue
        for rel in removed:
            (target / rel).unlink()
        for rel in added + changed:
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            (target / rel).write_bytes(content[rel])
        for directory, dirs, files_here in sorted(os.walk(target), key=lambda t: -len(t[0])):
            if not os.listdir(directory) and Path(directory) != target:
                os.rmdir(directory)
        listing = ''.join('%s  %s\n' % (sha256(target / rel), rel) for rel in sorted(content))
        pins = source / 'pins.json'
        commit = None
        if pins.is_file():
            try:
                commit = json.loads(pins.read_text('utf-8')).get('commit')
            except ValueError:
                commit = 'unreadable pins.json'
        entry = dict(synced_utc=utc(), files=files, bytes=size,
                     listing_sha256=hashlib.sha256(listing.encode('utf-8')).hexdigest(),
                     newest_source_utc=utc(max(p.stat().st_mtime for p, _ in selected.values())) if selected else None,
                     pinned_runtime_commit=commit,
                     excluded={reason: dict(files=c, bytes=b) for reason, (c, b) in sorted(excluded.items())},
                     edited=dict(sorted(edited.items())))
        tests = sorted(rel for rel in edited if rel.rsplit('/', 1)[-1].startswith('test_'))
        if tests:
            entry['edited_tests'] = {rel: 'snapshot-only edit (an assertion, docstring or path follows the edited '
                                            'text); the live test is unchanged' for rel in tests}
        note = args.note or entries.get(name, {}).get('note')
        if note:
            entry['note'] = note
        entries[name] = entry
    print('TOTAL %d files %d bytes | excluded %d files %d bytes%s'
          % (totals['files'], totals['bytes'], totals['excluded_files'], totals['excluded_bytes'],
             ' (dry run: nothing written)' if args.dry_run else ''))
    if not args.dry_run:
        snapshot['harnesses'] = dict(sorted(entries.items()))
        snapshot_path.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False) + '\n', 'utf-8', newline='\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

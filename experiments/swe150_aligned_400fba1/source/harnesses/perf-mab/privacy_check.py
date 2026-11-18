"""Local only (never delivered): the real MemAgent32 data never enters the harness directory.

usage: python -B privacy_check.py <harness dir> <data dir with manifest.json and contexts/>
Reads the local copy (read only) for probes - every case's first 80 characters of its first chunk and of its question,
every context file name - and fails when any harness file (validation logs, the sealed bundle and archives aside)
holds one. Exit 0 with PRIVATE_DATA_CHECK ... files_with_private_text=0, or 0 with 'skipped' when the copy is absent.
"""
import json
from pathlib import Path
import sys


def main(here, data):
    here, data = Path(here), Path(data)
    if not (data / 'manifest.json').is_file():
        print('PRIVATE_DATA_CHECK skipped (no local copy at %s)' % data)
        return 0
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    probes, names = set(), set()
    for case in manifest['cases']:
        context = json.loads((data / case['context_file']).read_text(encoding='utf-8'))
        probes.add(context['chunks'][0][:80])
        probes.add(case['question'][:80])
        names.add(Path(case['context_file']).name)
    probes = {p for p in probes if len(p.strip()) >= 15}      # empty questions (summaries) probe nothing
    bad = []
    for path in here.rglob('*'):
        if not path.is_file() or 'validation' in path.parts or path.suffix in ('.bundle', '.xz'):
            continue
        if path.name in names:
            bad.append(str(path))
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        if any(p in text for p in probes):
            bad.append(str(path))
    print('PRIVATE_DATA_CHECK cases=%d files_with_private_text=%d %s' % (len(manifest['cases']), len(bad), bad[:5]))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))

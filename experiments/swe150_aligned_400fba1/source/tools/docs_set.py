"""The documentation set of this repository, per language, for the tests that read the documentation.

Until integrate-1107 the README held every section; now the README keeps the status, the quickstart and the
documentation map, and its other sections are docs pages (its documentation map lists them). Tests that read "the
README" read README_SET: the README of the language and the pages that came out of it, in this order, so
every check finds what it found in the one file. text(root, 'README.md') is the English set,
text(root, 'README.zh-CN.md') the Chinese one; any other name is that one file. scope 'all' adds every other
page of the language under docs/ (English: the pages without a .zh-CN twin as well).
"""
from pathlib import Path

README_SET = {
    'README.md': ('README.md', 'docs/status.md', 'docs/install.md', 'docs/running.md', 'docs/reference.md',
                  'docs/design.md', 'docs/benchmarks/swe-bench-mini-swe-agent.md', 'docs/history.md'),
    'README.zh-CN.md': ('README.zh-CN.md', 'docs/status.zh-CN.md', 'docs/install.zh-CN.md', 'docs/running.zh-CN.md',
                        'docs/reference.zh-CN.md', 'docs/design.zh-CN.md',
                        'docs/benchmarks/swe-bench-mini-swe-agent.zh-CN.md', 'docs/history.zh-CN.md'),
}


def files(root, name, scope='readme'):
    """The paths of the set: README_SET[name] (every one must exist), or [root / name] for another name."""
    root = Path(root)
    if name not in README_SET:
        return [root / name]
    paths = [root / n for n in README_SET[name]]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('documentation set incomplete: %s' % missing)
    if scope == 'all':
        zh = name.endswith('.zh-CN.md')
        rest = sorted(p for p in (root / 'docs').rglob('*.md') if p.name.endswith('.zh-CN.md') == zh)
        paths += [p for p in rest if p not in paths]
    elif scope != 'readme':
        raise ValueError('scope must be readme or all')
    return paths


def text(root, name, scope='readme'):
    """The set's text: its files in order, joined by a newline."""
    return '\n'.join(p.read_text(encoding='utf-8') for p in files(root, name, scope))

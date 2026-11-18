"""The documentation's links and its English / Chinese parity (tools/docs_check.py; docs-reorg-1116). CPU only.

Tested here: every relative link of the repository's own pages (README, harnesses/README.md, docs/) names a file
that exists and, with an anchor, a heading of that page; every page has its twin in the other language with the
same headings per level, code blocks, commit ids, numbers of four or more digits and linked pages. The harness
snapshots (harnesses/perf-*/) are not checked (English only, copies of files kept elsewhere).
"""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import docs_check  # noqa: E402


def page(root, name, text):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


class RepositoryDocs(unittest.TestCase):
    def test_links_resolve(self):
        self.assertEqual(docs_check.links(ROOT), [])

    def test_english_and_chinese_pages_agree(self):
        self.assertEqual(docs_check.parity(ROOT), [])

    def test_snapshots_are_not_pages(self):
        names = [p.relative_to(ROOT).as_posix() for p in docs_check.pages(ROOT)]
        self.assertIn('docs/status.md', names)
        self.assertIn('docs/known-limits.zh-CN.md', names)
        self.assertFalse([n for n in names if n.startswith('harnesses/perf-')])


class Anchors(unittest.TestCase):
    def test_github_anchors(self):
        self.assertEqual(docs_check.anchor('4.10 Admission optimizations (opt-in)'), '410-admission-optimizations-opt-in')
        self.assertEqual(docs_check.anchor('What `main` contains (runtime `4ebff97`)'),
                         'what-main-contains-runtime-4ebff97')
        self.assertEqual(docs_check.anchor('各模型 × benchmark 的状态'), '各模型--benchmark-的状态')
        self.assertEqual(docs_check.anchor('integrate-1107（2026-11-07 起为 `main`）'), 'integrate-11072026-11-07-起为-main')
        self.assertEqual(docs_check.anchors('# A\n## A\n```\n# not a heading\n```\n'), {'a', 'a-1'})
        # An explicit anchor kept for a renamed heading counts; one inside code does not.
        self.assertEqual(docs_check.anchors('<a id="old-name"></a>\n\n# New name\n```\n<a id="x"></a>\n```\n'
                                            '`<a id="y"></a>`\n'), {'old-name', 'new-name'})


class Findings(unittest.TestCase):
    def test_broken_link_and_anchor_found(self):
        with TemporaryDirectory() as d:
            page(d, 'README.md', '[x](docs/a.md#b) [y](docs/none.md) `[z](docs/none.md)` [w](https://example.org)\n')
            page(d, 'README.zh-CN.md', '[x](docs/a.zh-CN.md)\n')
            page(d, 'docs/a.md', '# A\n')
            page(d, 'docs/a.zh-CN.md', '# A\n')
            found = {(f['link'], f['problem']) for f in docs_check.links(d)}
            self.assertEqual(found, {('docs/a.md#b', 'missing anchor'), ('docs/none.md', 'missing target')})

    def test_parity_differences_found(self):
        with TemporaryDirectory() as d:
            page(d, 'README.md', '# T\n\nJob 3,611,917 at `4ebff97`; [s](docs/s.md).\n\n```sh\nrun --a\n```\n')
            page(d, 'README.zh-CN.md', '# T\n\n作业 3611917，`4ebff97`；[s](docs/s.zh-CN.md)。\n\n```sh\nrun --a\n```\n')
            page(d, 'docs/s.md', '# S\n\n## One\n\nJob 3612333, commit 51b4085.\n')
            page(d, 'docs/s.zh-CN.md', '# S\n\n作业 3612334，提交 51b4085。\n')
            page(d, 'docs/only.md', '# O\n')
            problems = {(f['page'], f['problem']) for f in docs_check.parity(d)}
            self.assertEqual(problems, {('docs/s.md', 'headings'), ('docs/s.md', 'numbers'),
                                        ('docs/only.md', 'no Chinese page')})


if __name__ == '__main__':
    unittest.main()

"""Documentation checks: relative links and anchors, and English / Chinese parity (docs-reorg-1116).

  python -B tools/docs_check.py            prints every finding as JSON lines; exit 1 when there is one

The repository's own pages are README.md, harnesses/README.md and every .md under docs/, each with its
.zh-CN.md twin. The harness snapshots (harnesses/perf-*/) are copies of files kept elsewhere and are English only
(harnesses/README.md); they are not checked.

links(root): every relative link of a page, outside code, names a file or directory that exists, and an anchor
(#...) names a heading of its target page, as GitHub writes heading anchors (lower case, punctuation other than
'-' and '_' dropped, each space a '-', -1, -2 ... on repeats) or an explicit `<a id="..."></a>` outside code (kept
where a heading was renamed, so that its old anchor still resolves). Links to other sites are not followed.

parity(root): every English page has its Chinese twin and the other way round; the two have the same number of
headings at each level, the same fenced code blocks (comment lines dropped), the same set of 7-hex commit ids and
of numbers with at least four digits (job ids, counts; thousands separators dropped), and link to the same pages
(.zh-CN.md read as .md; anchors differ by language and are checked by links()). The wording and the line breaks
may differ. ALLOWED lists the known differences, each with its reason.
"""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ZH = '.zh-CN.md'
FENCE = re.compile(r'^(```|~~~)[^\n]*\n.*?^\1[ \t]*$', re.S | re.M)
INLINE = re.compile(r'`[^`\n]*`')
LINK = re.compile(r'(?<!!)\[(?:[^\[\]]|\[[^\]]*\])*\]\(([^()\s]+(?:\([^()\s]*\))?[^()\s]*)\)')
HEAD = re.compile(r'^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$', re.M)
ANCHOR_ID = re.compile(r'<a[ \t]+id="([^"]+)"[ \t]*>[ \t]*</a>')  # an old anchor kept for a renamed heading
COMMIT = re.compile(r'(?<![0-9A-Za-z])(?=[0-9]*[a-f])[0-9a-f]{7}(?![0-9A-Za-z])')  # not a 7-digit job id
NUMBER = re.compile(r'(?<![\w.])(?<!\d,)(?:\d{1,3}(?:,\d{3})+|\d{4,})(?!\w|[.,]\d)')

# Known differences between the English and the Chinese text of a page: {page: {kind: {value: reason}}}.
ALLOWED = {}


def pages(root=ROOT):
    """The repository's own documentation pages (both languages), sorted."""
    root = Path(root)
    found = [root / 'README.md', root / 'README.zh-CN.md', root / 'harnesses' / 'README.md',
             root / 'harnesses' / 'README.zh-CN.md']
    found += (root / 'docs').rglob('*.md')
    return sorted(p for p in found if p.is_file())


def prose(text):
    """The text without fenced code blocks and inline code (both may show link syntax)."""
    return INLINE.sub('', FENCE.sub('', text))


def anchor(title):
    s = INLINE.sub(lambda m: m.group(0)[1:-1], title).strip().lower()
    s = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', s)
    s = ''.join(c for c in s if c.isalnum() or c in ' -_' or ord(c) > 0x2e7f and c.isalnum())
    return s.replace(' ', '-')


def anchors(text):
    seen, out = {}, set()
    for m in HEAD.finditer(FENCE.sub('', text)):
        a = anchor(m.group(2))
        n = seen.get(a, 0)
        seen[a] = n + 1
        out.add(a if n == 0 else '%s-%d' % (a, n))
    out.update(ANCHOR_ID.findall(prose(text)))
    return out


def links(root=ROOT):
    """Broken relative links: a list of {page, link, problem}."""
    root = Path(root).resolve()
    cache, out = {}, []
    for page in pages(root):
        text = page.read_text(encoding='utf-8')
        for target in LINK.findall(prose(text)):
            if re.match(r'^[a-z][a-z0-9+.-]*:', target, re.I):
                continue
            path, _, frag = target.partition('#')
            dest = page if not path else (page.parent / path).resolve()
            rel = page.relative_to(root).as_posix()
            if not dest.exists():
                out.append(dict(page=rel, link=target, problem='missing target'))
                continue
            if frag and dest.is_file() and dest.suffix == '.md':
                if dest not in cache:
                    cache[dest] = anchors(dest.read_text(encoding='utf-8'))
                if frag not in cache[dest]:
                    out.append(dict(page=rel, link=target, problem='missing anchor'))
    return out


def facts(text):
    body = FENCE.sub('', text)
    heads = {}
    for m in HEAD.finditer(body):
        heads[len(m.group(1))] = heads.get(len(m.group(1)), 0) + 1
    fences = ['\n'.join(l for l in m.group(0).splitlines() if not l.lstrip().startswith('#'))
              for m in FENCE.finditer(text)]
    linked = set()
    for target in LINK.findall(prose(text)):
        if re.match(r'^[a-z][a-z0-9+.-]*:', target, re.I):
            continue
        path = target.partition('#')[0].replace(ZH, '.md')
        if path:
            linked.add(path)
    words = re.sub(r'\]\([^()\s]*\)', ']', text)  # a link's target is not text (anchors hold dates)
    return dict(headings=heads, fences=fences, commits=set(COMMIT.findall(words)),
                numbers={n.replace(',', '') for n in NUMBER.findall(words)}, links=linked)


def parity(root=ROOT):
    """English / Chinese differences: a list of {page, problem, ...}."""
    root = Path(root).resolve()
    out = []
    names = {p.relative_to(root).as_posix() for p in pages(root)}
    for name in sorted(names):
        if name.endswith(ZH):
            if name[:-len(ZH)] + '.md' not in names:
                out.append(dict(page=name, problem='no English page'))
            continue
        twin = name[:-3] + ZH
        if twin not in names:
            out.append(dict(page=name, problem='no Chinese page'))
            continue
        en = facts((root / name).read_text(encoding='utf-8'))
        zh = facts((root / twin).read_text(encoding='utf-8'))
        allowed = ALLOWED.get(name, {})
        if en['headings'] != zh['headings']:
            out.append(dict(page=name, problem='headings', en=en['headings'], zh=zh['headings']))
        if en['fences'] != zh['fences']:
            bad = [i for i, (a, b) in enumerate(zip(en['fences'], zh['fences'])) if a != b]
            out.append(dict(page=name, problem='code blocks', en=len(en['fences']), zh=len(zh['fences']),
                            first_differing=bad[:5]))
        for kind in ('commits', 'numbers', 'links'):
            ok = set(allowed.get(kind, {}))
            only_en = sorted(en[kind] - zh[kind] - ok)
            only_zh = sorted(zh[kind] - en[kind] - ok)
            if only_en or only_zh:
                out.append(dict(page=name, problem=kind, only_en=only_en[:20], only_zh=only_zh[:20]))
    return out


def main():
    found = [dict(check='links', **x) for x in links()] + [dict(check='parity', **x) for x in parity()]
    for x in found:
        print(json.dumps(x, ensure_ascii=False, sort_keys=True))
    print(json.dumps(dict(status='DOCS_CHECK_PASS' if not found else 'DOCS_CHECK_FAIL', findings=len(found),
                          pages=len(pages()))))
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())

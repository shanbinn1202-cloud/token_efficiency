"""MemoryAgentBench cases for run_cases.py: the MemAgent32 selection, the chunking rule, builder, loader, synthetics.

A cases directory holds `manifest.json` (one record per case: id, split, source, metric, system, query, question,
answers, context_file, context_sha256, chunks, context_tokens, and the case's source row: file, row,
question_index) and `contexts/<sha256>.json` (dict(chunks, context_tokens, sha256)). Cases that share a context
share its file. The loader checks every case before use (file names, hashes, chunk counts, types, sizes) and
treats the data as untrusted text: it is only ever sent as message content.

- **Chunking** (`chunk_context`): the case's context is encoded with the Qwen3.6-27B tokenizer (the release's
  pinned copy, source-overlay/*/model-evidence; `encode(text, add_special_tokens=False)`) and cut into consecutive
  windows of 4,096 tokens, each decoded back to text (the last window is shorter). The chunks concatenate to the
  context; `context_tokens` is the token count. The same chunks serve every model (also Gemma 4). The loop adds the
  `[History chunk k/N]` header when it delivers a chunk.
- **Prompts** (`official_prompts`): `system` is MemoryAgentBench's SYSTEM_MESSAGE and `query` its
  `get_template(source, 'query', 'Long_context_agent')` filled with the question (utils/templates.py of the
  user's MemoryAgentBench checkout at commit 538026089d, SHA-256 pinned in upstream/SHA256SUMS; not copied here).
- **Metrics** (`METRICS`): the official metric each source is scored with (score.py): substring_exact_match for
  RULER QA and FactConsolidation, exact_match for the in-context-learning sets, llm_judge_f1 for the InfBench
  summaries (the official GPT judge; score.py leaves them judge_pending).
- **MemAgent32** (`MEMAGENT32`): the 32 cases of the benchmark runs, 8 per ability, as (split file, row, question
  index) of the Hugging Face dataset `SOURCE_DATASET` at `SOURCE_REVISION` (not ai-hyz/MemoryAgentBench), with
  each case's chunk count and context tokens under the rule above. No data of that dataset is distributed with
  this repository, and the dataset declares no license: `build` reads the split files of your own download at that
  revision (Parquet, or JSON Lines; each row with `context`, `questions`, `answers` and `metadata.source`) and
  `verify` compares any cases directory with the pinned counts.
- **Synthetic cases** (`synthetic`): deterministic ledger texts with one code word to recall, for smoke runs
  without the dataset (about 4,000 Qwen3.6 tokens per chunk; the code word is checked by substring).

Usage:
  python -I -B examples/memoryagentbench/cases.py synthetic --out <dir> [--cases 4 --chunks 6]
  python -I -B examples/memoryagentbench/cases.py build --rows <download dir> --mab <checkout> --out <dir>
  python -I -B examples/memoryagentbench/cases.py verify --cases <dir>
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHUNK_TOKENS = 4096
MAB_REPOSITORY = 'https://github.com/HUST-AI-HYZ/MemoryAgentBench'
MAB_COMMIT = '538026089d1a8a8eff05121d0db89b388f360eba'
SUMS = HERE / 'upstream' / 'SHA256SUMS'
# The source of the MemAgent32 rows (a public Hugging Face dataset; no license declared; nothing of it is here).
SOURCE_DATASET = 'cxkjntmngm/MemAgent_Bench'
SOURCE_REVISION = 'a785de0f8d2511a43b9e524b5326e1105a4d218a'
# MemoryAgentBench's SYSTEM_MESSAGE and its RULER-QA long-context query template (utils/templates.py, MIT,
# Copyright (c) 2026 Yuanzhe Hu); used for the synthetic cases, and compared with the checkout's in build.
SYSTEM = 'You are a helpful assistant that can read the context and memorize it for future retrieval.'
QUERY = ('Answer the question based on the memorized documents. Only give me the answer and do not output any other '
         'words. \n\nQuestion: %s \n\n Answer:')
METRICS = {'ruler_qa': 'substring_exact_match', 'factconsolidation': 'substring_exact_match',
           'in_context_learning': 'exact_match', 'infbench_sum': 'llm_judge_f1'}
CONTEXT_FILE = re.compile(r'contexts/([0-9a-f]{64})\.json\Z')
CASE_ID = re.compile(r'[A-Za-z0-9_.+-]{1,160}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
MAX_MANIFEST_BYTES = 8 << 20
MAX_CONTEXT_BYTES = 8 << 20
MAX_CHUNKS = 64
MAX_TEXT = 1 << 20


def _memagent32():
    """(id, split, file, row, question_index, chunks, context_tokens) of the 32 cases."""
    out = []
    files = {'Accurate_Retrieval': 'Accurate_Retrieval-00000-of-00002.parquet',
             'Conflict_Resolution': 'Conflict_Resolution-00000-of-00001.parquet',
             'Long_Range_Understanding': 'Long_Range_Understanding-00000-of-00001.parquet',
             'Test_Time_Learning': 'Test_Time_Learning-00000-of-00001.parquet'}
    groups = (
        ('Accurate_Retrieval', 'ruler_qa2_validation_65536',
         [(342, 0, 14, 54852), (341, 0, 14, 55215), (391, 0, 14, 56571), (309, 0, 14, 53296),
          (365, 0, 15, 58405), (350, 0, 15, 57636), (301, 0, 13, 53051), (370, 0, 14, 55876)]),
        ('Conflict_Resolution', 'factconsolidation_sh_32k',
         [(5, q, 10, 39409) for q in (53, 37, 60, 22, 5, 4, 70, 93)]),
        ('Long_Range_Understanding', 'infbench_sum_eng_shots2',
         [(58, 0, 17, 67635), (91, 0, 18, 70981), (84, 0, 19, 77050), (77, 0, 18, 70375),
          (90, 0, 18, 73699), (52, 0, 20, 78885), (56, 0, 19, 75767), (29, 0, 18, 73658)]),
        ('Test_Time_Learning', 'icl_trec_coarse_6600shot_balance',
         [(r, 0, 28, 113458) for r in (337, 324, 375, 350, 396, 393, 311, 373)]),
    )
    for split, source, rows in groups:
        for row, question, chunks, tokens in rows:
            out.append(dict(id='%s-r%d-q%d' % (source, row, question), split=split, source=source,
                            file=files[split], row=row, question_index=question, chunks=chunks,
                            context_tokens=tokens))
    return out


MEMAGENT32 = _memagent32()


class DataError(ValueError):
    """Case data refused (a hash, shape or size check failed)."""


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _need(condition, message):
    if not condition:
        raise DataError(message)


# ---------------------------------------------------------------- the chunking rule

def chunk_context(text, encode, decode, size=CHUNK_TOKENS):
    """(chunks, token count): consecutive `size`-token windows of encode(text), each decoded."""
    ids = list(encode(text))
    chunks = [decode(ids[i:i + size]) for i in range(0, len(ids), size)]
    _need(''.join(chunks) == text, 'the decoded windows do not concatenate to the context')
    return chunks, len(ids)


def qwen_tokenizer(path=None):
    """(encode, decode) of the Qwen3.6-27B tokenizer: a model directory, or by default the release's pinned copy."""
    import shutil
    import tempfile
    from transformers import AutoTokenizer
    if path is None:
        found = sorted(ROOT.glob('source-overlay/*/model-evidence/tokenizer.json'))
        _need(len(found) == 1, 'the release\'s Qwen3.6 tokenizer files were not found under source-overlay')
        evidence = found[0].parent
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for target, name in (('config.json', 'model-config.json'), ('tokenizer.json', 'tokenizer.json'),
                                 ('tokenizer_config.json', 'tokenizer-config.json')):
                shutil.copyfile(evidence / name, work / target)
            tok = AutoTokenizer.from_pretrained(str(work), local_files_only=True)
    else:
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return (lambda text: tok.encode(text, add_special_tokens=False),
            lambda ids: tok.decode(ids, skip_special_tokens=False))


# ---------------------------------------------------------------- the official prompts

def pinned_sums(path=SUMS):
    """{upstream path: SHA-256} of upstream/SHA256SUMS."""
    out = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.strip()] = digest
    return out


def pinned_file(mab_root, name):
    """A file of the user's MemoryAgentBench checkout, checked against its pin."""
    path = Path(mab_root) / name
    _need(path.is_file(), 'MemoryAgentBench %s not found under %s (a checkout at %s)' % (name, mab_root, MAB_COMMIT))
    expected = pinned_sums()[name]
    got = sha256_file(path)
    _need(got == expected, '%s has SHA-256 %s, not the pinned %s (commit %s)' % (path, got, expected, MAB_COMMIT))
    return path


def official_templates(mab_root):
    """utils/templates.py of the checkout (data only: dictionaries and two lookup functions), SHA-256 pinned."""
    path = pinned_file(mab_root, 'utils/templates.py')
    spec = importlib.util.spec_from_file_location('memoryagentbench_templates', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _need(module.SYSTEM_MESSAGE == SYSTEM, 'the checkout\'s SYSTEM_MESSAGE differs')
    return module


def official_prompts(templates, source, question):
    """(system, query, metric) of one case."""
    key = templates.normalize_dataset_name(source)
    _need(key in METRICS, 'no metric for the source %r (template %r)' % (source, key))
    system = templates.get_template(source, 'system', 'Long_context_agent')
    query = templates.get_template(source, 'query', 'Long_context_agent').format(question=question)
    return system, query, METRICS[key]


# ---------------------------------------------------------------- building from source rows

def read_rows(path):
    """The rows of one split file: Parquet (by its magic bytes; needs pyarrow) or else JSON Lines; each a dict with
    context, questions, answers and metadata.source."""
    path = Path(path)
    with path.open('rb') as stream:
        magic = stream.read(4)
    if magic == b'PAR1':
        import pyarrow.parquet as pq
        rows = pq.read_table(str(path)).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    for k, row in enumerate(rows):
        _need(type(row) is dict and type(row.get('context')) is str and type(row.get('questions')) is list
              and type(row.get('answers')) is list and type(row.get('metadata')) is dict,
              '%s row %d: context, questions, answers and metadata expected' % (path.name, k))
    return rows


def find_split_file(rows_dir, name):
    """A split file of the download: <dir>/<name> or <dir>/data/<name> (hf download --local-dir keeps data/)."""
    for candidate in (Path(rows_dir) / name, Path(rows_dir) / 'data' / name):
        if candidate.is_file():
            return candidate
    raise DataError('%s not found in %s (or its data/)' % (name, rows_dir))


def build(rows_dir, selection, encode, decode, templates):
    """The cases of `selection` (records with id, split, source, file, row, question_index) from the split files
    in `rows_dir`."""
    cache, out = {}, []
    for item in selection:
        if item['file'] not in cache:
            cache[item['file']] = read_rows(find_split_file(rows_dir, item['file']))
        rows = cache[item['file']]
        _need(0 <= item['row'] < len(rows), '%s: row %d not in %s (%d rows)' % (item['id'], item['row'], item['file'],
                                                                                len(rows)))
        row = rows[item['row']]
        source = (row.get('metadata') or {}).get('source')
        _need(source == item['source'], '%s: the row\'s source is %r' % (item['id'], source))
        q = item['question_index']
        _need(0 <= q < len(row['questions']), '%s: no question %d' % (item['id'], q))
        question, answers = row['questions'][q], row['answers'][q]
        chunks, tokens = chunk_context(row['context'], encode, decode)
        _need(len(chunks) <= MAX_CHUNKS, '%s: %d chunks' % (item['id'], len(chunks)))
        system, query, metric = official_prompts(templates, source, question)
        out.append(dict(id=item['id'], split=item['split'], source=source, metric=metric, system=system,
                        query=query, question=question, answers=answers, chunks=chunks,
                        context_sha256=sha256_text(''.join(chunks)), context_tokens=tokens,
                        file=item['file'], row=item['row'], question_index=q))
    return out


def write_cases(out, cases, about=None):
    """A cases directory: manifest.json and contexts/<sha256>.json."""
    out = Path(out)
    _need(not out.exists() or not any(out.iterdir()), 'the cases directory is not empty: %s' % out)
    (out / 'contexts').mkdir(parents=True, exist_ok=True)
    records = []
    for c in cases:
        name = 'contexts/%s.json' % c['context_sha256']
        path = out / name
        if not path.exists():
            path.write_text(json.dumps(dict(chunks=c['chunks'], context_tokens=c['context_tokens'],
                                            sha256=c['context_sha256']), ensure_ascii=False) + '\n',
                            encoding='utf-8', newline='\n')
        records.append(dict({k: v for k, v in c.items() if k not in ('chunks', 'fact_chunk')},
                            context_file=name, chunks=len(c['chunks'])))
    manifest = dict(about or {}, count=len(records), chunk_tokens=CHUNK_TOKENS, cases=records)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n', encoding='utf-8',
                                       newline='\n')
    return out


# ---------------------------------------------------------------- loading (untrusted data)

def _read_json(path, limit):
    path = Path(path)
    _need(path.is_file() and not path.is_symlink(), 'not a regular file: %s' % path.name)
    raw = path.read_bytes()
    _need(len(raw) <= limit, '%s exceeds %d bytes' % (path.name, limit))
    try:
        return json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DataError('%s is not UTF-8 JSON: %s' % (path.name, exc)) from None


def _text(value, name, limit=MAX_TEXT):
    _need(type(value) is str and len(value) <= limit, '%s must be text of at most %d characters' % (name, limit))
    return value


def check_record(case):
    """One manifest record's fields (types, sizes, the context file name)."""
    _need(type(case) is dict, 'manifest case must be an object')
    cid = case.get('id')
    _need(type(cid) is str and CASE_ID.match(cid), 'bad case id %r' % (cid,))
    found = CONTEXT_FILE.match(case.get('context_file') or '') if type(case.get('context_file')) is str else None
    _need(found is not None, '%s: context_file must be contexts/<64 hex>.json' % cid)
    digest = case.get('context_sha256')
    _need(type(digest) is str and HEX64.match(digest) and found.group(1) == digest,
          '%s: context_sha256 must be 64 hex and name the context file' % cid)
    _need(type(case.get('chunks')) is int and 1 <= case['chunks'] <= MAX_CHUNKS, '%s: chunks' % cid)
    _need(type(case.get('context_tokens')) is int and case['context_tokens'] > 0, '%s: context_tokens' % cid)
    answers = case.get('answers')
    plain = lambda a: type(a) is str or (type(a) is list and all(type(x) is str for x in a))  # noqa: E731
    _need(type(answers) is str or (type(answers) is list and all(plain(a) for a in answers)), '%s: answers' % cid)
    return dict(id=cid, system=_text(case.get('system'), cid + '.system'),
                query=_text(case.get('query'), cid + '.query'), question=_text(case.get('question'), cid + '.question'),
                answers=answers if type(answers) is list else [answers],
                source=_text(case.get('source'), cid + '.source', 200),
                split=_text(case.get('split'), cid + '.split', 200),
                metric=_text(case.get('metric'), cid + '.metric', 200), context_file=case['context_file'],
                context_sha256=digest, chunk_count=case['chunks'], context_tokens=case['context_tokens'])


def read_manifest(cases_dir):
    """(manifest SHA-256, checked records) of <cases_dir>/manifest.json."""
    path = Path(cases_dir) / 'manifest.json'
    value = _read_json(path, MAX_MANIFEST_BYTES)
    _need(type(value) is dict and type(value.get('cases')) is list and value['cases'], 'manifest has no cases')
    records = [check_record(c) for c in value['cases']]
    ids = [c['id'] for c in records]
    _need(len(set(ids)) == len(ids), 'duplicate case ids in the manifest')
    return sha256_file(path), records


def load_context(cases_dir, record):
    """The chunks of one checked record, verified against its context_sha256."""
    value = _read_json(Path(cases_dir) / record['context_file'], MAX_CONTEXT_BYTES)
    cid = record['id']
    _need(type(value) is dict and type(value.get('chunks')) is list, '%s: context file has no chunks' % cid)
    chunks = value['chunks']
    _need(len(chunks) == record['chunk_count'] and all(type(c) is str and 0 < len(c) <= MAX_TEXT for c in chunks),
          '%s: %d chunks, the manifest says %d' % (cid, len(chunks), record['chunk_count']))
    _need(value.get('sha256') == record['context_sha256'], '%s: context file sha256 field differs' % cid)
    actual = sha256_text(''.join(chunks))
    _need(actual == record['context_sha256'], '%s: chunks hash to %s, the manifest says %s'
          % (cid, actual, record['context_sha256']))
    return chunks


def load_cases(cases_dir, ids=None):
    """The checked cases of a cases directory (with their chunks), optionally only `ids` in that order."""
    _, records = read_manifest(cases_dir)
    if ids is not None:
        order = {cid: k for k, cid in enumerate(ids)}
        _need(len(order) == len(ids), 'duplicate case ids')
        unknown = sorted(set(order) - {c['id'] for c in records})
        _need(not unknown, 'unknown case ids %s' % unknown)
        records = sorted((c for c in records if c['id'] in order), key=lambda c: order[c['id']])
    out = []
    for c in records:
        chunks = load_context(cases_dir, c)
        out.append(dict(id=c['id'], system=c['system'], query=c['query'], question=c['question'],
                        answers=c['answers'], chunks=chunks, source=c['source'], split=c['split'],
                        metric=c['metric'], context_sha256=c['context_sha256'], context_tokens=c['context_tokens']))
    return out


def verify(cases_dir, expected=None):
    """Every case of a cases directory checked, and compared with the pinned counts (MEMAGENT32 by default)."""
    expected = {c['id']: c for c in (MEMAGENT32 if expected is None else expected)}
    cases = load_cases(cases_dir)
    rows = []
    for c in cases:
        pin = expected.get(c['id'])
        pinned = pin is not None and 'chunks' in pin and 'context_tokens' in pin
        rows.append(dict(id=c['id'], chunks=len(c['chunks']), context_tokens=c['context_tokens'], pinned=pinned,
                         counts_match=None if not pinned else (len(c['chunks']), c['context_tokens'])
                         == (pin['chunks'], pin['context_tokens'])))
    found = {r['id'] for r in rows}
    return dict(status='VERIFIED', cases=len(rows), pinned=sum(r['pinned'] for r in rows),
                counts_match=sum(r['counts_match'] is True for r in rows),
                counts_differ=[r['id'] for r in rows if r['counts_match'] is False],
                missing=sorted(set(expected) - found), rows=rows)


# ---------------------------------------------------------------- synthetic cases

NAMES = ('Halvard Ostrem', 'Maren Vidal', 'Tobias Kerr', 'Ilse Brandt', 'Oskar Lund', 'Petra Novak', 'Ansel Moreau',
         'Greta Holm', 'Lukas Aalto', 'Nadia Petrov', 'Emil Sauer', 'Freya Lind', 'Jonas Weber', 'Carla Mendes',
         'Viktor Hale', 'Sofie Dahl', 'Rafael Ortiz', 'Hanna Berg', 'Milo Fischer', 'Tessa Quinn')
PLACES = ('Velmora', 'Ashgrove', 'Kestrel Point', 'Duncliff', 'Marrowgate', 'Sunhallow', 'Brindle Ford', 'Oakmere',
          'Saltmarsh', 'Thornwick', 'Greyhaven', 'Lowmoor', 'Ravensby', 'Coldwater', 'Elmstead', 'Highbarrow')
GOODS = ('crates of amber', 'barrels of salt', 'bolts of linen', 'sacks of barley', 'casks of cider',
         'bundles of hides', 'chests of glassware', 'coils of rope', 'jars of honey', 'reams of paper',
         'baskets of apples', 'kegs of nails')
GATES = ('north gate', 'river quay', 'east market', 'old toll bridge', 'harbour office', 'west granary',
         'chapel yard', 'customs house')
WEATHER = ('under a clear sky', 'in steady rain', 'through morning fog', 'against a cold wind', 'in light snow',
           'on a humid afternoon')
CODEWORDS = ('MARIGOLD', 'TAMARIND', 'BLACKTHORN', 'LANTERN', 'SAFFRON', 'CORMORANT', 'JUNIPER', 'OBSIDIAN',
             'PERIWINKLE', 'HAWTHORN', 'CINNABAR', 'TOURMALINE', 'WILLOWHERB', 'KINGFISHER', 'SANDALWOOD', 'ANEMONE')


def _record(rng, number):
    name, place, goods = rng.choice(NAMES), rng.choice(PLACES), rng.choice(GOODS)
    return ('Record %05d. On day %d the courier %s carried %d %s from %s to the %s %s; the ledger notes a toll of %d '
            'silver marks and a seal numbered %d.' % (number, rng.randrange(1, 366), name, rng.randrange(2, 400), goods,
                                                      place, rng.choice(GATES), rng.choice(WEATHER),
                                                      rng.randrange(1, 90), rng.randrange(1000, 99999)))


def _fact(place, code, number):
    return ('Record %05d. The archivist sealed the vault of %s with the code word %s, and wrote that this code word '
            'would be asked for later.' % (number, place, code))


def synthetic_case(name, seed, index, chunks, chunk_chars):
    """One deterministic case: `chunks` chunks of about `chunk_chars` characters, one fact record away from both
    ends."""
    rng = random.Random('%s:%s:%d' % (seed, name, index))
    place, code = rng.choice(PLACES), rng.choice(CODEWORDS)
    fact_chunk = rng.randrange(1, chunks - 1) if chunks > 2 else 0
    number, texts = 1, []
    for k in range(chunks):
        lines = ['Ledger section %d of %d.' % (k + 1, chunks)]
        fact_at = rng.randrange(3, 12) if k == fact_chunk else None
        size = len(lines[0])
        while size < chunk_chars:
            if fact_at is not None and len(lines) == fact_at:
                line = _fact(place, code, number)
                fact_at = None
            else:
                line = _record(rng, number)
            number += 1
            lines.append(line)
            size += len(line) + 1
        if fact_at is not None:
            lines.append(_fact(place, code, number))
            number += 1
        texts.append('\n'.join(lines) + '\n')
    question = 'What code word did the archivist use to seal the vault of %s?' % place
    joined = ''.join(texts)
    return dict(id='%s-%02d' % (name, index), system=SYSTEM, question=question, query=QUERY % question,
                answers=[code], chunks=texts, source='synthetic', split='synthetic', metric='substring_exact_match',
                context_sha256=sha256_text(joined), context_tokens=max(1, len(joined) // 4),
                file=None, row=index, question_index=0)


def synthetic(name='synthetic', seed=20261014, cases=4, chunks=6, chunk_chars=13000):
    """`cases` synthetic cases; `chunks` an int or a list cycled over the cases. context_tokens is an estimate
    (characters / 4), used only for ordering."""
    counts = [chunks] * cases if isinstance(chunks, int) else [chunks[k % len(chunks)] for k in range(cases)]
    return [synthetic_case(name, seed, k, n, chunk_chars) for k, n in enumerate(counts)]


# ---------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('synthetic', help='write synthetic cases')
    s.add_argument('--out', required=True, type=Path)
    s.add_argument('--cases', type=int, default=4)
    s.add_argument('--chunks', type=int, default=6)
    s.add_argument('--chunk-chars', type=int, default=13000)
    s.add_argument('--seed', type=int, default=20261014)
    b = sub.add_parser('build', help='build cases from source rows (Parquet or JSON Lines split files)')
    b.add_argument('--rows', required=True, type=Path,
                   help='the split files the selection names: for MemAgent32 your download of %s at %s'
                        % (SOURCE_DATASET, SOURCE_REVISION[:12]))
    b.add_argument('--mab', required=True, type=Path, help='a MemoryAgentBench checkout at %s' % MAB_COMMIT[:10])
    b.add_argument('--out', required=True, type=Path)
    b.add_argument('--select', default='memagent32',
                   help='memagent32 (the default), or a JSON file: a list of {id, split, source, file, row, '
                        'question_index}')
    b.add_argument('--tokenizer', type=Path, help='a Qwen3.6-27B model directory (default: the release\'s copy)')
    v = sub.add_parser('verify', help='check a cases directory and compare it with the MemAgent32 counts')
    v.add_argument('--cases', required=True, type=Path)
    args = p.parse_args(argv)
    if args.command == 'synthetic':
        out = write_cases(args.out, synthetic(seed=args.seed, cases=args.cases, chunks=args.chunks,
                                              chunk_chars=args.chunk_chars),
                          dict(kind='synthetic', seed=args.seed, context_tokens='characters / 4 (an estimate)'))
        print(json.dumps(dict(status='CASES_WRITTEN', out=str(out), cases=args.cases)))
        return 0
    if args.command == 'build':
        selection = MEMAGENT32 if args.select == 'memagent32' else json.loads(Path(args.select).read_text(
            encoding='utf-8'))
        encode, decode = qwen_tokenizer(args.tokenizer)
        cases = build(args.rows, selection, encode, decode, official_templates(args.mab))
        about = dict(kind='built', selection=args.select, mab_commit=MAB_COMMIT)
        if args.select == 'memagent32':
            about.update(dataset=SOURCE_DATASET, revision=SOURCE_REVISION)
        write_cases(args.out, cases, about)
        report = verify(args.out, selection)
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
        return 0
    report = verify(args.cases)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""perf-mab workloads (stdlib only; delivered): synthetic read_next cases and the real MemAgent32 cases.

A case is dict(id, system, query, question, answers, chunks, source, split, metric, context_sha256, context_tokens,
fact_chunk): the read_next loop (mab_runner.run_case) sends `system` + the read instruction and the first user
message, delivers `chunks` one per read_next call as tool messages, and appends `query` as a user message in the same
request as the last chunk.

Synthetic cases (`synthetic_cases`): deterministic text from a fixed seed (no tokenizer needed; build.py seals the
plan, test_real_tokenizer measures the token counts): ledger records of couriers, goods, places and numbers, about
`chunk_chars` characters per chunk (~4,000 Qwen3.6 tokens at the configured value), and one fact record (a vault code
word) in a chunk away from both ends; the question asks for that code word. The text is the same on every server,
so token-level comparisons (G1c replay, G2a vs G2b) see identical inputs.

Real cases (`load_real`): MemoryAgentBench MemAgent32 data, read as untrusted JSON only (never executed): a manifest
(`manifest.json`, 32 cases) and `contexts/<sha256>.json` files. Every case is checked before use: the manifest's
SHA-256 against the pin, the context file name against `contexts/<64 hex>.json` (no other path), the file's own
`sha256` field and the SHA-256 of its concatenated chunks against the case's `context_sha256`, the chunk count, the
field types and size limits. A mismatch rejects the case set (DataError); nothing is repaired. The text is only ever
used as message content. `real_pin` derives the hash-only pin (manifest SHA-256 and per-case ids, context SHA-256,
chunk counts and token counts) that build.py stores in pins.json: no case text leaves the data directory.
"""
import hashlib
import json
from pathlib import Path
import random
import re

SCHEMA = 'pmab-workload-plan-v1'
SYSTEM = 'You are a helpful assistant that can read the context and memorize it for future retrieval.'
QUERY = ('Answer the question based on the memorized documents. Only give me the answer and do not output any other '
         'words. \n\nQuestion: %s \n\n Answer:')
CONTEXT_FILE = re.compile(r'contexts/([0-9a-f]{64})\.json\Z')
CASE_ID = re.compile(r'[A-Za-z0-9_.+-]{1,160}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
MAX_MANIFEST_BYTES = 8 << 20
MAX_CONTEXT_BYTES = 8 << 20
MAX_CHUNKS = 64
MAX_TEXT = 1 << 20

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


class DataError(ValueError):
    """Real data refused (a hash, shape or size check failed)."""


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


# ---------------------------------------------------------------- synthetic cases

def record_text(rng, number):
    name, place, goods = rng.choice(NAMES), rng.choice(PLACES), rng.choice(GOODS)
    return ('Record %05d. On day %d the courier %s carried %d %s from %s to the %s %s; the ledger notes a toll of %d '
            'silver marks and a seal numbered %d.' % (number, rng.randrange(1, 366), name, rng.randrange(2, 400), goods,
                                                      place, rng.choice(GATES), rng.choice(WEATHER),
                                                      rng.randrange(1, 90), rng.randrange(1000, 99999)))


def fact_text(place, code, number):
    return ('Record %05d. The archivist sealed the vault of %s with the code word %s, and wrote that this code word '
            'would be asked for later.' % (number, place, code))


def synthetic_case(name, seed, index, chunks, chunk_chars):
    """One deterministic case: `chunks` chunks of about `chunk_chars` characters, one fact record."""
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
                line = fact_text(place, code, number)
                fact_at = None
            else:
                line = record_text(rng, number)
            number += 1
            lines.append(line)
            size += len(line) + 1
        if fact_at is not None:                      # a short chunk: the fact record closes it
            lines.append(fact_text(place, code, number))
            number += 1
        texts.append('\n'.join(lines) + '\n')
    question = 'What code word did the archivist use to seal the vault of %s?' % place
    joined = ''.join(texts)
    return dict(id='%s-%02d' % (name, index), system=SYSTEM, question=question, query=QUERY % question,
                answers=[code], chunks=texts, source='synthetic', split='synthetic', metric='exact_code_word',
                context_sha256=sha256_text(joined), context_tokens=None, fact_chunk=fact_chunk)


def chunk_counts(spec):
    """Chunks per case: spec['chunks'] is an int (every case) or a list cycled over the cases."""
    value = spec['chunks']
    if isinstance(value, int):
        return [value] * spec['cases']
    return [value[k % len(value)] for k in range(spec['cases'])]


def synthetic_cases(spec):
    """The synthetic cases of a workload spec {kind: synthetic, name, seed, cases, chunks, chunk_chars}."""
    return [synthetic_case(spec['name'], spec['seed'], k, n, spec['chunk_chars'])
            for k, n in enumerate(chunk_counts(spec))]


# ---------------------------------------------------------------- real cases (untrusted data)

def _need(condition, message):
    if not condition:
        raise DataError(message)


def _read_json(path, limit):
    path = Path(path)
    _need(path.is_file() and not path.is_symlink(), 'not a regular file: %s' % path.name)
    raw = path.read_bytes()
    _need(len(raw) <= limit, '%s exceeds %d bytes' % (path.name, limit))
    try:
        return raw, json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DataError('%s is not UTF-8 JSON: %s' % (path.name, exc)) from None


def _text(value, name, limit=MAX_TEXT):
    _need(type(value) is str and len(value) <= limit, '%s must be text of at most %d characters' % (name, limit))
    return value


def check_case(case):
    """One manifest case's fields (types, sizes, the context file name); returns the cleaned record."""
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
    # A text, or a list of texts (or of lists of texts): the dataset's own shapes, kept as given.
    plain = lambda a: type(a) is str or (type(a) is list and all(type(x) is str for x in a))
    _need(type(answers) is str or (type(answers) is list and all(plain(a) for a in answers)), '%s: answers' % cid)
    return dict(id=cid, system=_text(case.get('system'), cid + '.system'),
                query=_text(case.get('query'), cid + '.query'), question=_text(case.get('question'), cid + '.question'),
                answers=answers, source=_text(case.get('source'), cid + '.source', 200),
                split=_text(case.get('split'), cid + '.split', 200), metric=_text(case.get('metric'), cid + '.metric', 200),
                context_file=case['context_file'], context_sha256=digest, chunk_count=case['chunks'],
                context_tokens=case['context_tokens'])


def read_manifest(data_dir, manifest_sha256=None):
    """(manifest SHA-256, checked cases) of <data_dir>/manifest.json."""
    raw, value = _read_json(Path(data_dir) / 'manifest.json', MAX_MANIFEST_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    _need(manifest_sha256 is None or digest == manifest_sha256,
          'manifest.json SHA-256 %s differs from the pinned %s' % (digest, manifest_sha256))
    _need(type(value) is dict and type(value.get('cases')) is list and value['cases'], 'manifest has no cases')
    cases = [check_case(c) for c in value['cases']]
    ids = [c['id'] for c in cases]
    _need(len(set(ids)) == len(ids), 'duplicate case ids in the manifest')
    return digest, cases


def load_context(data_dir, case):
    """The chunks of one checked case, verified against its context_sha256 (file field and concatenation)."""
    raw, value = _read_json(Path(data_dir) / case['context_file'], MAX_CONTEXT_BYTES)
    cid = case['id']
    _need(type(value) is dict and type(value.get('chunks')) is list, '%s: context file has no chunks' % cid)
    chunks = value['chunks']
    _need(len(chunks) == case['chunk_count'] and all(type(c) is str and 0 < len(c) <= MAX_TEXT for c in chunks),
          '%s: %d chunks, the manifest says %d' % (cid, len(chunks), case['chunk_count']))
    _need(value.get('sha256') == case['context_sha256'], '%s: context file sha256 field differs' % cid)
    actual = sha256_text(''.join(chunks))
    _need(actual == case['context_sha256'], '%s: chunks hash to %s, the manifest says %s'
          % (cid, actual, case['context_sha256']))
    return chunks


def load_real(data_dir, *, manifest_sha256=None, case_ids=None, pin=None):
    """Checked real cases (with chunks) of `data_dir`, optionally only `case_ids` (in their order); with `pin`
    (real_pin's record) every case's id, context SHA-256 and chunk count must equal the pin too."""
    digest, cases = read_manifest(data_dir, manifest_sha256 or (pin or {}).get('manifest_sha256'))
    if pin is not None:
        pinned = {c['id']: c for c in pin['cases']}
        _need([c['id'] for c in cases] == [c['id'] for c in pin['cases']], 'manifest case list differs from the pin')
        for c in cases:
            p = pinned[c['id']]
            _need((c['context_sha256'], c['chunk_count'], c['context_tokens']) ==
                  (p['context_sha256'], p['chunks'], p['context_tokens']), '%s differs from the pin' % c['id'])
    if case_ids is not None:
        # The workload's order (the runner's pool takes the cases in it: G4 lists the largest first).
        order = {cid: k for k, cid in enumerate(case_ids)}
        _need(len(order) == len(case_ids), 'duplicate case ids')
        _need(set(order) <= {c['id'] for c in cases}, 'unknown case ids %s' % sorted(set(order) - {c['id'] for c in cases}))
        cases = sorted((c for c in cases if c['id'] in order), key=lambda c: order[c['id']])
    out = []
    for c in cases:
        chunks = load_context(data_dir, c)
        out.append(dict(id=c['id'], system=c['system'], query=c['query'], question=c['question'],
                        answers=c['answers'] if type(c['answers']) is list else [c['answers']], chunks=chunks, source=c['source'], split=c['split'], metric=c['metric'],
                        context_sha256=c['context_sha256'], context_tokens=c['context_tokens'], fact_chunk=None))
    return digest, out


def real_pin(data_dir):
    """The hash-only pin of a real data directory (every case verified first): no text."""
    digest, cases = load_real(data_dir)
    return dict(manifest_sha256=digest, count=len(cases),
                cases=[dict(id=c['id'], context_sha256=c['context_sha256'], chunks=len(c['chunks']),
                            context_tokens=c['context_tokens'], split=c['split']) for c in cases])


def verify_real(data_dir, pin):
    """Job start / login gate: every pinned case verified (manifest SHA-256, per-case context SHA-256). Returns a
    record (never raises): status VERIFIED, or REFUSED / UNAVAILABLE with the reason."""
    path = Path(data_dir)
    if not (path / 'manifest.json').is_file():
        return dict(status='UNAVAILABLE', data_dir=str(path), reason='manifest.json not found')
    try:
        digest, cases = load_real(path, pin=pin)
    except (DataError, OSError) as exc:
        return dict(status='REFUSED', data_dir=str(path), reason='%s: %s' % (type(exc).__name__, exc))
    return dict(status='VERIFIED', data_dir=str(path), manifest_sha256=digest, cases=len(cases),
                chunks=sum(len(c['chunks']) for c in cases),
                context_sha256={c['id']: c['context_sha256'] for c in cases})


# ---------------------------------------------------------------- workloads and the sealed plan

def cases_for(spec, *, data_dir=None, pin=None):
    """The cases of one workload spec: synthetic (generated) or real (verified from `data_dir` against `pin`)."""
    if spec['kind'] == 'synthetic':
        return synthetic_cases(spec)
    if spec['kind'] == 'real':
        _need(data_dir is not None and pin is not None, 'real workload needs its data directory and pin')
        _, cases = load_real(data_dir, pin=pin, case_ids=spec.get('cases'))
        return cases
    raise ValueError('unknown workload kind %r' % spec['kind'])


def describe(spec, *, data_pin=None):
    """The sealed description of one workload (no text): case ids, chunk counts and digests, planning estimates."""
    est = spec.get('estimate') or {}
    if spec['kind'] == 'synthetic':
        cases = synthetic_cases(spec)
        rows = [dict(id=c['id'], chunks=len(c['chunks']), context_sha256=c['context_sha256'],
                     chars=sum(len(t) for t in c['chunks']), fact_chunk=c['fact_chunk'],
                     answer_sha256=sha256_text(c['answers'][0])) for c in cases]
        chunk_tokens = est.get('chunk_tokens')
    else:
        _need(data_pin is not None, 'a real workload needs pins.real_data')
        pinned = {c['id']: c for c in data_pin['cases']}
        ids = list(spec['cases']) if spec.get('cases') is not None else [c['id'] for c in data_pin['cases']]
        _need(len(set(ids)) == len(ids) and set(ids) <= set(pinned), 'real workload cases outside pins.real_data')
        rows = [dict(id=i, chunks=pinned[i]['chunks'], context_sha256=pinned[i]['context_sha256'],
                     context_tokens=pinned[i]['context_tokens']) for i in ids]
        chunk_tokens = None
    tokens = [(r['context_tokens'] if r.get('context_tokens') else r['chunks'] * (chunk_tokens or 4000)) for r in rows]
    return dict(kind=spec['kind'], name=spec.get('name'), cases=rows, count=len(rows),
                chunks=sum(r['chunks'] for r in rows), max_chunks=max(r['chunks'] for r in rows),
                context_tokens_est=sum(tokens), max_context_tokens_est=max(tokens))


def build_plan(config, data_pin=None):
    """pins.json `workload_plan`: every workload's description and the plan's SHA-256 (regenerated by the job)."""
    workloads = {name: describe(spec, data_pin=data_pin if spec['kind'] == 'real' else None)
                 for name, spec in sorted(config['workloads'].items())}
    value = dict(schema=SCHEMA, workloads=workloads)
    value['sha256'] = hashlib.sha256(canonical(value).encode()).hexdigest()
    return value


if __name__ == '__main__':
    # Local helper: python -B mab_workload.py check <data dir>  (verifies every case; prints the hash-only pin)
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == 'check':
        pin = real_pin(sys.argv[2])
        print(json.dumps(dict(status='VERIFIED', manifest_sha256=pin['manifest_sha256'], cases=pin['count'],
                              chunks=sum(c['chunks'] for c in pin['cases']),
                              context_tokens=[min(c['context_tokens'] for c in pin['cases']),
                                              max(c['context_tokens'] for c in pin['cases'])])))
    else:
        print('usage: python -B mab_workload.py check <data dir with manifest.json and contexts/>')
        sys.exit(2)

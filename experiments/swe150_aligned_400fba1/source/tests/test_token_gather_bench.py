"""tools/token_gather_bench.py (gather-bench-1115): prototype B of per-token scoring, the token gather throughput probe
(today's gather path, a device-direct read of the pinned arena, the Page restore of the same bytes).

CPU only (CPU torch for the tensor tests). Covered:

- the arguments, the geometry limits and the constants (5,051 = 315.7 x 16 tokens, 316 blocks, 5,056 slots);
- the token sets: scattered (even over the Pages, never a whole Page), clustered (runs of 12 on consecutive Pages),
  their refusals; the restore Pages of the same bytes;
- the KV layout: the bench's replica of vLLM 0.24.0's FlashAttention layout against vLLM's own code compiled from the
  pinned source closure (test_token_gather.vllm_code), and the bench's caches against vLLM's own
  _reshape_attention_kv_cache (test_token_gather.FlashRunner), in both stride orders;
- the world: TensorIO.capture_pages fills the arena exactly, the catalog's slots and captures;
- today_parts: bit-identical to the runtime's TensorIO.gather_tokens (the same caches, the same H2D bytes), and its
  pin GATHER_TOKENS_SHA256 equal to the loaded source (a change of gather_tokens fails here: update the replica);
- DirectGather (with the host buffers themselves as the "mapping"): bit-identical to gather_tokens on device rows,
  arena rows across chunks and captured Pages, both stride orders; its refusals before any write; the
  __cuda_array_interface__ it would map, map_pinned's refusal of unpinned memory;
- the exactness check finds a flipped bit and a write outside the targets;
- the timing summary (nearest-rank p90), the G1 estimates (140.7 ms at 5,051 tokens), the decision rule and its line,
  the record schema (validate_record) and the verdicts;
- main() end to end in --cpu-test (every row, copy set and attention check; the record and the stdout lines) and its
  refusals (an existing record, no CUDA without --cpu-test), a load failure, a corrupted device-direct copy (FAIL) and
  a mapping that cannot be made (ERROR).

Not covered (GPU only, the bench's GPU stage): map_pinned's CUDA alias of page-locked memory (zero-copy reads over the
interconnect), vLLM's backend and flash_attn_varlen_func, CUDA events, pinned allocations, and every timing.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
OVERLAY = HERE / 'source-overlay'
sys.path.insert(0, str(OVERLAY))

try:
    import torch
except ImportError:
    torch = None


def tool():
    sys.path.insert(0, str(HERE / 'tools'))
    try:
        import token_gather_bench as bench
    finally:
        sys.path.remove(str(HERE / 'tools'))
    return bench


B = tool()
P = B.P


def runtime():
    from remote_ra_growing_worker import tensors
    from remote_ra_live_v2 import token_gather as TG
    return tensors, TG


def world(order='HND', seed=5, geometry=None):
    tensors, TG = runtime()
    return B.World(torch, tensors, TG, torch.device('cpu'), dict(geometry or B.CPU), order, True, seed)


def refused():
    """The bench's own refusals (ValueError) and the runtime's (the worker's validate, the live engine's require)."""
    from remote_ra_growing_worker.contract import WorkerAbort
    from remote_ra_live_v2.contract import LiveV2Error
    return (ValueError, WorkerAbort, LiveV2Error)


REFUSED = refused()


class ArgumentTests(unittest.TestCase):
    def test_defaults_and_constants(self):
        args = B.parse(['--source', 's', '--out', 'o.json'])
        self.assertEqual(args.sizes, [144, 512, 1024, 2048, 3072, 4096, 5051, 6144, 8192])
        self.assertEqual((args.spreads, args.repeat, args.warmup, args.target, args.one_thread_sizes),
                         (['scattered', 'clustered'], 20, 3, 5051, [144, 1024, 5051]))
        self.assertEqual(B.TARGET_TOKENS, round(315.7 * 16))
        self.assertEqual((B.blocks_of(5051), B.slots_of(5051), B.slots_of(144), B.slots_of(1024)), (316, 5056, 144,
                                                                                                    1024))
        self.assertEqual(B.limit(B.GPU), 12 * 1024)                 # clustered over the 1,024 resident Pages
        self.assertGreaterEqual(B.limit(dict(B.GPU, **B.GPU['other'])), B.slots_of(5051))  # the NHD world's targets
        cpu = B.parse(['--source', 's', '--out', 'o.json', '--cpu-test'])
        self.assertEqual((cpu.sizes, cpu.target, cpu.repeat, cpu.warmup, cpu.one_thread_sizes),
                         ([16, 48, 144], 144, 2, 1, [48]))
        self.assertEqual(B.parse(['--source', 's', '--out', 'o', '--sizes', '1024,144,144', '--target', '144',
                                  '--one-thread-sizes', '']).sizes, [144, 1024])

    def test_refusals(self):
        for bad in (['--sizes', '0'], ['--sizes', '12289'], ['--sizes', 'a,b'], ['--spreads', 'random'],
                    ['--spreads', 'scattered,scattered'], ['--sizes', '144,1024', '--target', '5051'],
                    ['--one-thread-sizes', '7'], ['--repeat', '2'], ['--warmup', '-1'], ['--spreads', '']):
            with self.subTest(bad=bad), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                B.parse(['--source', 's', '--out', 'o.json'] + bad)
        B.parse(['--source', 's', '--out', 'o.json', '--cpu-test', '--repeat', '1'])   # CPU: one repetition is fine


class TokenSetTests(unittest.TestCase):
    def test_scattered_even_never_a_whole_page(self):
        for count, pages in ((144, 2048), (5056, 2048), (8192, 2048), (8192, 1024), (48, 48), (15 * 10, 10)):
            with self.subTest(count=count, pages=pages):
                got = B.scattered(count, range(100, 100 + pages), random.Random(count))
                self.assertEqual(len(got), count)
                self.assertEqual(got, sorted(set(got)))
                per = {}
                for t in got:
                    per[t // P] = per.get(t // P, 0) + 1
                self.assertTrue(set(per) <= set(range(100, 100 + pages)))
                self.assertLessEqual(max(per.values()), P - 1)
                self.assertEqual(len(per), min(count, pages))
                self.assertLessEqual(max(per.values()) - min(per.values()), 1)
                self.assertEqual(got, B.scattered(count, range(100, 100 + pages), random.Random(count)))
        with self.assertRaises(ValueError):
            B.scattered(15 * 10 + 1, range(10), random.Random(0))
        with self.assertRaises(ValueError):
            B.scattered(0, range(10), random.Random(0))

    def test_clustered_runs_on_consecutive_pages(self):
        for count in (16, 144, 5056, 8192):
            with self.subTest(count=count):
                got = B.clustered(count, range(1024, 3072), random.Random(count))
                self.assertEqual(len(got), count)
                pages = sorted({t // P for t in got})
                self.assertEqual(pages, list(range(pages[0], pages[0] + len(pages))))
                self.assertEqual(len(pages), -(-count // B.CLUSTER_RUN))
                for page in pages:
                    offsets = sorted(t % P for t in got if t // P == page)
                    self.assertEqual(offsets, list(range(offsets[0], offsets[0] + len(offsets))))
                    self.assertLessEqual(len(offsets), B.CLUSTER_RUN)
        with self.assertRaises(ValueError):
            B.clustered(12 * 10 + 1, range(10), random.Random(0))
        with self.assertRaises(ValueError):
            B.clustered(24, [1, 2, 4], random.Random(0))                 # not consecutive


@unittest.skipUnless(torch is not None, 'CPU torch required')
class LayoutTests(unittest.TestCase):
    def test_replica_is_vllm_0_24_0s_flash_attention_layout(self):
        import test_token_gather as TT
        flash, _, state = TT.vllm_code()
        for layout in ('NHD', 'HND'):
            with self.subTest(layout=layout):
                state['layout'] = layout
                stand_in = B.replica(layout)
                for n, b, h, d in ((17, 16, 4, 256), (3072, 16, 4, 256), (80, 32, 2, 8)):
                    self.assertEqual(stand_in.get_kv_cache_shape(n, b, h, d), flash.get_kv_cache_shape(n, b, h, d))
                self.assertEqual(stand_in.get_kv_cache_stride_order(), flash.get_kv_cache_stride_order())
                self.assertEqual(stand_in.get_kv_cache_stride_order(include_num_layers_dimension=True),
                                 flash.get_kv_cache_stride_order(include_num_layers_dimension=True))
                with self.assertRaises(ValueError):
                    stand_in.get_kv_cache_shape(17, 8, 4, 256)

    def test_caches_as_vllms_runner_reshapes_them(self):
        import test_token_gather as TT
        g = B.CPU
        for order in ('NHD', 'HND'):
            with self.subTest(order=order):
                w = world(order)
                flash = TT.FlashRunner(order, g['blocks'], P, g['heads'], g['size'], layers=g['layers'])
                self.assertEqual([(tuple(t.shape), t.stride()) for t in w.fa],
                                 [(tuple(t.shape), t.stride()) for t in flash.fa])
                self.assertEqual(w.layout['stride_order_name'], order)
                self.assertEqual(w.layout['axes_attached'], [0, 2])
                self.assertEqual(w.io.gather_axes, {(n, 0): (0, 2) for n in w.names})


@unittest.skipUnless(torch is not None, 'CPU torch required')
class WorldTests(unittest.TestCase):
    def test_the_arena_is_captured_exactly(self):
        w = world()
        g = B.CPU
        self.assertEqual(w.capture_check, dict(pages=g['dram'], mismatched_pages=0, status='PASS'))
        self.assertEqual(len(w.arena.chunks), g['dram'] // g['chunk_slots'])
        R, D = g['resident'], g['dram']
        self.assertEqual([w.catalog[p].slot for p in w.dram_pages], list(range(D)))
        self.assertEqual(w.dram_pages, list(range(R, R + D)))
        self.assertTrue(all(type(w.catalog[p]) is dict for p in w.capture_pages))
        self.assertEqual(w.free, R + g['scratch'] + 1)
        # the scratch blocks were overwritten after the captures: a host row read from the device would differ
        payload = w.catalog[w.dram_pages[-1]]
        chunk, slot = divmod(payload.slot, w.chunk_slots)
        self.assertFalse(torch.equal(B.bits(w.arena.host(chunk, w.names[0], 0)[slot]),
                                     B.bits(w.kv[0][:, w.scratch[(D - 1) % g['scratch']]])))

    def test_restore_pages_of_the_same_bytes(self):
        w = world()
        rng = random.Random(3)
        tokens = B.positions(w, 144, 'scattered', 'dram', rng)
        pages = B.restore_pages_for(w, tokens, 'scattered')
        self.assertEqual(len(pages), 9)
        self.assertEqual(len(set(pages)), 9)
        self.assertTrue(set(pages) <= set(w.dram_pages))
        self.assertGreater(len({w.catalog[p].slot // w.chunk_slots for p in pages}), 1)
        tokens = B.positions(w, 144, 'clustered', 'dram', rng)
        pages = B.restore_pages_for(w, tokens, 'clustered')
        self.assertEqual(pages, list(range(pages[0], pages[0] + 9)))
        self.assertTrue(set(pages) <= {t // P for t in tokens})


@unittest.skipUnless(torch is not None, 'CPU torch required')
class VariantTests(unittest.TestCase):
    """today_parts and DirectGather against the runtime's TensorIO.gather_tokens: two worlds of the same seed, the
    same token sets, the whole cache compared bit for bit."""

    def sets(self, w):
        rng = random.Random(11)
        count, pages = B.CPU['mixed']
        return [('mixed', B.mixed(w, count, pages, 0.4, rng)), ('resident', B.positions(w, 48, 'scattered',
                                                                                         'resident', rng)),
                ('scattered', B.positions(w, 144, 'scattered', 'dram', rng)),
                ('clustered', B.positions(w, 144, 'clustered', 'dram', rng))]

    def run_both(self, order, variant):
        tensors, TG = runtime()
        ours, theirs = world(order), world(order)
        self.assertTrue(all(torch.equal(B.bits(a), B.bits(b)) for a, b in zip(ours.kv, theirs.kv)))
        direct = B.DirectGather(ours, lambda host: host) if variant == 'direct' else None
        for name, tokens in self.sets(ours):
            with self.subTest(order=order, variant=variant, set=name):
                found_ours = TG.sources(tokens, P, ours.resident, None, ours.catalog)
                found_theirs = TG.sources(tokens, P, theirs.resident, None, theirs.catalog)
                targets = [{0: (b,)} for b in ours.fresh(len(tokens) // P)]
                self.assertEqual(targets, [{0: (b,)} for b in theirs.fresh(len(tokens) // P)])
                ours.io._http_h2d_issued = theirs.io._http_h2d_issued = 0
                if variant == 'direct':
                    direct(targets, found_ours)
                else:
                    B.today_parts(ours.io, targets, found_ours, B.Clock(ours))
                theirs.io.gather_tokens(targets, found_theirs)
                self.assertTrue(all(torch.equal(B.bits(a), B.bits(b)) for a, b in zip(ours.kv, theirs.kv)))
                self.assertEqual(ours.io._http_h2d_issued, 0 if variant == 'direct' else theirs.io._http_h2d_issued)
                host = sum(1 for s in found_theirs if s[0] == 'host')
                self.assertEqual(theirs.io._http_h2d_issued, host * theirs.token_bytes)
        return ours, direct

    def test_today_parts_is_gather_tokens(self):
        for order in ('HND', 'NHD'):
            self.run_both(order, 'parts')
        tensors, _ = runtime()
        self.assertEqual(B.gather_tokens_source(tensors), (B.GATHER_TOKENS_SHA256, True))

    def test_today_parts_marks_every_part(self):
        w = world()
        tokens = B.positions(w, 48, 'scattered', 'dram', random.Random(2))
        found = w.TG.sources(tokens, P, w.resident, None, w.catalog)
        clock = B.Clock(w)
        B.today_parts(w.io, [{0: (b,)} for b in w.fresh(3)], found, clock)
        self.assertEqual(list(clock.parts), list(B.TODAY_PARTS))
        self.assertTrue(all(v >= 0 for v in clock.parts.values()))

    def test_direct_is_gather_tokens(self):
        for order in ('HND', 'NHD'):
            ours, direct = self.run_both(order, 'direct')
            self.assertEqual(len(direct.maps), len(ours.arena.chunks) * ours.layers)
            self.assertEqual(len(direct.captured), B.CPU['captures'] * ours.layers)   # every capture, mapped once
            self.assertTrue(direct.info['zero_copy'])

    def test_direct_refusals_before_any_write(self):
        w = world()
        direct = B.DirectGather(w, lambda host: host)
        payload = w.catalog[w.dram_pages[0]]
        snapshot = [t.clone() for t in w.kv]
        good = [('device', {0: (2,)}, 0)] * (P - 1) + [('host', payload, 1)]
        cases = [([{0: (40,)}], good[:-1]),                                       # one source short
                 ([{0: (40,)}], good[:-1] + [('host', payload, P)]),              # offset outside the Page
                 ([{0: (2,)}], good),                                             # target aliases a source
                 ([{0: (40,)}, {0: (40,)}], good + good),                         # aliased targets
                 ([{0: (40,)}], good[:-1] + [('disk', payload, 0)]),              # unknown source kind
                 ([{0: (40,)}], good[:-1] + [('host', {'fa0': ()}, 0)]),          # incomplete payload
                 ([], [])]                                                        # nothing to gather
        for targets, sources in cases:
            with self.subTest(targets=targets, last=sources[-1:]), self.assertRaises(REFUSED):
                direct(targets, sources)
        released = w.io.capture_pages([{0: (3,)}])[0]
        w.arena.slots.release(released.slot, released.token)
        with self.assertRaisesRegex(REFUSED, 'foreign or released'):
            direct([{0: (40,)}], [('host', released, 0)] * P)
        self.assertTrue(all(torch.equal(B.bits(a), B.bits(b)) for a, b in zip(w.kv, snapshot)))

    def test_the_mapping_it_would_make(self):
        host = torch.zeros((3, 2, P, 2, 8), dtype=torch.bfloat16)
        face = B.interface(host)
        self.assertEqual(face, dict(shape=(3, 2, P, 2, 8), typestr='<i2', data=(host.data_ptr(), False),
                                    strides=None, version=2))
        alias = B.CudaAlias(host)
        self.assertIs(alias.host, host)
        self.assertEqual(alias.__cuda_array_interface__, face)
        for bad in (host.transpose(0, 1), torch.zeros(4, dtype=torch.float32)):
            with self.assertRaises(ValueError):
                B.interface(bad)
        with self.assertRaisesRegex(ValueError, 'page-locked'):            # an unpinned buffer is refused
            B.map_pinned(torch, host, torch.device('cpu'))


@unittest.skipUnless(torch is not None, 'CPU torch required')
class CheckTests(unittest.TestCase):
    def test_compare_finds_a_flipped_bit_and_a_write_outside(self):
        w = world()
        tokens = B.positions(w, 48, 'scattered', 'dram', random.Random(4))
        found = w.TG.sources(tokens, P, w.resident, None, w.catalog)
        truth = B.reference(w, found)
        targets = w.fresh(3)
        blocks = [{0: (b,)} for b in targets]
        _, moved, mismatched, outside = B.checked(w, targets, truth, lambda: w.io.gather_tokens(blocks, found))
        self.assertEqual((moved, mismatched, outside), (48 * w.token_bytes, [], []))

        def corrupt():
            w.io.gather_tokens(blocks, found)
            w.kv[1][1, targets[1], 5, 0, 3] = -w.kv[1][1, targets[1], 5, 0, 3] - 1     # slot 16 + 5, V
            w.kv[0][0, 7, 0, 0, 0] = 99.0                                                 # a resident block
        _, _, mismatched, outside = B.checked(w, targets, truth, corrupt)
        self.assertEqual((mismatched, outside), ([P + 5], [7]))


class RecordTests(unittest.TestCase):
    def test_summary_nearest_rank_p90(self):
        self.assertEqual(B.summary([5, 1, 3, 2, 4]), dict(n=5, best_ms=1, median_ms=3, p90_ms=5, mean_ms=3))
        ten = B.summary(list(range(1, 11)))
        self.assertEqual((ten['p90_ms'], ten['median_ms']), (9, 5.5))
        self.assertEqual(B.summary([2.5])['p90_ms'], 2.5)
        with self.assertRaises(ValueError):
            B.summary([])

    def test_g1_estimates(self):
        e = B.g1_estimates(5051)
        self.assertEqual((e['line_ms'], e['intercept_ms'], e['slope_ms_per_token']), (140.7, 50.8376, 0.017798))
        self.assertEqual(e['bytes'], 331022336)
        self.assertEqual(e['restore_throughput_ms'], [24.6, 34.5])

    @staticmethod
    def row(variant, ms, status='PASS', tokens=5051, spread='scattered', source='dram', threads='default'):
        timing = dict(n=20, best_ms=ms, median_ms=ms, p90_ms=ms, mean_ms=ms, timer='cuda_events')
        if variant == 'today_parts':
            timing['parts'] = {n: dict(median_ms=float(i)) for i, n in enumerate(B.TODAY_PARTS)}
        return dict(variant=variant, source=source, spread=spread, tokens=tokens, threads=threads, timing=timing,
                    status=status)

    def test_decide(self):
        def label(today, direct, restore, **kw):
            rows = [self.row('today', today), self.row('direct', direct), self.row('restore', restore),
                    self.row('today_parts', today)]
            return B.decide(rows, 5051, 'scattered')
        d = label(140.7, 5.0, 30.0)
        self.assertEqual((d['label'], d['today_over_direct'], d['restore_over_direct'], d['dominant_part']),
                         ('DEVICE_DIRECT_ADVISED', 28.14, 6.0, 'drain'))
        self.assertEqual(d['line'], 'TOKEN_GATHER_DECISION=DEVICE_DIRECT_ADVISED tokens=5051 slots=5056 '
                                    'spread=scattered today_ms=140.70 direct_ms=5.00 restore_ms=30.00 '
                                    'today_over_direct=28.14x dominant_part=drain')
        self.assertEqual(label(37.0, 30.0, 30.0)['label'], 'TODAY_PATH_OK')          # <= 1.25 x restore
        self.assertEqual(label(9.5, 9.0, 1.0)['label'], 'TODAY_PATH_OK')             # under G1's 10 ms
        self.assertEqual(label(140.0, 71.0, 30.0)['label'], 'NO_CLEAR_WINNER')
        self.assertEqual(label(140.0, 70.0, 30.0)['label'], 'DEVICE_DIRECT_ADVISED')
        missing = B.decide([self.row('today', 1.0), self.row('restore', 1.0)], 5051, 'scattered')
        self.assertEqual((missing['label'], missing['reason']), ('NO_DECISION', 'no direct row'))
        self.assertTrue(missing['line'].endswith('reason=no_direct_row'))
        failed = B.decide([self.row('today', 1.0), self.row('direct', 1.0, 'FAIL'), self.row('restore', 1.0)],
                          5051, 'scattered')
        self.assertEqual((failed['label'], failed['today_ms']), ('NO_DECISION', None))
        other = B.decide([self.row('today', 140.0, threads=1), self.row('direct', 5.0), self.row('restore', 30.0)],
                         5051, 'scattered')
        self.assertEqual(other['label'], 'NO_DECISION')                              # the one-thread row is not today

    def test_verdicts_and_schema(self):
        good = dict((k, None) for k in B.RECORD_KEYS)
        good.update(schema=B.SCHEMA, errors={}, capture_check={'HND': dict(status='PASS')}, rows=[],
                    copy=dict(status='PASS', sets=[]), attention=dict(today=dict(status='PASS')),
                    direct=dict(status='MAPPED'), decision=B.decide([], 5051, 'scattered'))
        self.assertEqual(B.verdict_of(good), 'TOKEN_GATHER_BENCH_PASS')
        good['verdict'] = 'TOKEN_GATHER_BENCH_PASS'
        self.assertEqual(B.validate_record(good), [])
        for change, verdict in ((dict(errors=dict(bench='x')), 'FAIL'), (dict(copy=dict(status='FAIL')), 'FAIL'),
                                (dict(attention=dict(today=dict(status='FAIL'))), 'FAIL'),
                                (dict(rows=[dict(status='FAIL')]), 'FAIL'),
                                (dict(attention=dict(today=dict(status='ATTENTION_UNAVAILABLE'))), 'ERROR'),
                                (dict(direct=dict(status='DIRECT_UNAVAILABLE')), 'ERROR'),
                                (dict(errors=dict(backend='vLLM not importable')), 'ERROR')):
            with self.subTest(change=change):
                self.assertEqual(B.verdict_of(dict(good, **change)), 'TOKEN_GATHER_BENCH_' + verdict)
        self.assertIn('verdict TOKEN_GATHER_BENCH_PASS, the record says TOKEN_GATHER_BENCH_FAIL',
                      B.validate_record(dict(good, copy=dict(status='FAIL'))))
        self.assertIn('missing key rows', B.validate_record({k: v for k, v in good.items() if k != 'rows'}))
        self.assertIn("schema 'v0'", B.validate_record(dict(good, schema='v0')))
        bad_row = dict(self.row('today', 1.0), slots=5051, pad=5, pages=1, blocks=316, host_rows=5051,
                       device_rows=0, bytes=1, gbps=1.0, exact=True, mismatched_slots=0, changed_outside_targets=0,
                       h2d_bytes_per_call=0, h2d_bytes_expected=0, bytes_ok=True)
        self.assertIn('row 0: slots 5051 for 5051 tokens', B.validate_record(dict(good, rows=[bad_row])))


@unittest.skipUnless(torch is not None, 'CPU torch required')
class RunTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def main(self, *extra, out=None):
        out = out or self.root / 'bench.json'
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = B.main(['--cpu-test', '--source', str(OVERLAY), '--out', str(out), *extra])
        record = json.loads(out.read_text(encoding='utf-8')) if out.exists() else None
        return code, record, stdout.getvalue().splitlines(), stderr.getvalue()

    def test_cpu_run_end_to_end(self):
        code, record, lines, _ = self.main()
        self.assertEqual(code, 0)
        self.assertEqual(B.validate_record(record), [])
        self.assertEqual(record['verdict'], 'TOKEN_GATHER_BENCH_PASS', record['errors'])
        self.assertEqual(lines[-1], 'TOKEN_GATHER_BENCH=TOKEN_GATHER_BENCH_PASS %s' % (self.root / 'bench.json'))
        self.assertEqual(lines[-2], record['decision']['line'])
        self.assertEqual(lines[-3], record['decisions']['clustered']['line'])
        self.assertEqual(record['decision']['spread'], 'scattered')
        self.assertIn(record['decision']['label'], B.LABELS)
        self.assertFalse((self.root / 'bench.json.partial').exists())
        sizes, spreads = [16, 48, 144], ['scattered', 'clustered']
        want = {(v, s, sp, n, 'default') for sp in spreads for n in sizes
                for v, s in (('today', 'dram'), ('today_parts', 'dram'), ('restore', 'dram'), ('today', 'resident'),
                             ('direct', 'dram'), ('direct', 'resident'))}
        want |= {(v, 'dram', 'scattered', 48, 1) for v in ('today', 'today_parts')}
        got = [(r['variant'], r['source'], r['spread'], r['tokens'], r['threads']) for r in record['rows']]
        self.assertEqual(sorted(got, key=str), sorted(want, key=str))
        for row in record['rows']:
            self.assertEqual((row['status'], row['exact'], row['bytes'], row['slots']),
                             ('PASS', True, row['slots'] * 2 * 2 * 8 * 2 * 2, B.slots_of(row['tokens'])), row)
            if row['variant'] == 'today_parts':
                self.assertEqual(set(row['timing']['parts']), set(B.TODAY_PARTS))
            if row['variant'] == 'direct':
                self.assertEqual(set(row['parts']['parts']), set(B.DIRECT_PARTS))
                self.assertEqual(row['h2d_bytes_per_call'], 0)
        copies = {(c['order'], c['name'], c['variant']) for c in record['copy']['sets']}
        names = ['mixed48', 'target144-scattered', 'target144-clustered']
        self.assertEqual(copies, {(o, n, v) for o in ('HND', 'NHD') for n in names
                                  for v in (('today', 'direct') if n == 'mixed48' else ('today', 'restore', 'direct'))})
        self.assertTrue(all(c['status'] == 'PASS' for c in record['copy']['sets']))
        mixed = next(c for c in record['copy']['sets'] if c['name'] == 'mixed48' and c['variant'] == 'direct')
        self.assertTrue(mixed['device_rows'] and mixed['capture_rows'] and len(mixed['chunks']) > 1)
        self.assertEqual({k: v['status'] for k, v in record['attention'].items()}, dict(today='PASS', direct='PASS'))
        self.assertEqual((record['direct']['status'], record['direct']['mapping']), ('MAPPED', 'cpu_stand_in'))
        self.assertEqual(record['gather_tokens_source']['replica_follows'], True)
        self.assertEqual(set(record['layout']), {'HND', 'NHD'})
        self.assertEqual(record['estimates']['tokens'], 144)
        self.assertEqual(set(record['phases']), {'world_hnd', 'checks_today_hnd', 'rows_today', 'world_nhd',
                                                 'checks_today_nhd', 'checks_direct_hnd', 'rows_direct',
                                                 'checks_direct_nhd'})

    def test_refusals(self):
        out = self.root / 'exists.json'
        out.write_text('{}', encoding='utf-8')
        code, record, lines, err = self.main(out=out)
        self.assertEqual((code, record, lines), (2, {}, []))
        self.assertIn('exists', err)
        if not torch.cuda.is_available():
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = B.main(['--source', str(OVERLAY), '--out', str(self.root / 'gpu.json')])
            self.assertEqual(code, 2)
            self.assertIn('CUDA required', stderr.getvalue())
            self.assertFalse((self.root / 'gpu.json').exists())

    def test_load_failure_is_a_failed_record(self):
        out = self.root / 'load.json'
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            code = B.main(['--cpu-test', '--source', str(self.root / 'missing'), '--out', str(out)])
        record = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual((code, record['verdict']), (1, 'TOKEN_GATHER_BENCH_FAIL'))
        self.assertTrue(record['errors']['bench'].startswith('load: '))
        self.assertEqual(record['decision']['label'], 'NO_DECISION')
        self.assertEqual(B.validate_record(record), [])

    def test_a_wrong_device_direct_copy_fails_the_bench(self):
        copy = B.DirectGather.copy

        def wrong(self, plan):
            copy(self, plan)
            for v in plan['relevant']:          # one slot of the first gathered block, K, negated
                cache = v.tensors[0]
                b = int(plan['dst_b'][v.group][0])
                o = int(plan['dst_o'][0])
                cache[b, 0, o] = -cache[b, 0, o] - 1
        with patch.object(B.DirectGather, 'copy', wrong):
            code, record, lines, _ = self.main()
        self.assertEqual((code, record['verdict']), (1, 'TOKEN_GATHER_BENCH_FAIL'))
        self.assertTrue(all(c['status'] == ('FAIL' if c['variant'] == 'direct' else 'PASS')
                            for c in record['copy']['sets']))
        self.assertTrue(all(r['status'] == ('FAIL' if r['variant'] == 'direct' else 'PASS') for r in record['rows']))
        self.assertEqual((record['decision']['label'], record['decision']['reason']), ('NO_DECISION',
                                                                                       'a check failed'))
        self.assertEqual(B.validate_record(record), [])
        self.assertTrue(lines[-1].startswith('TOKEN_GATHER_BENCH=TOKEN_GATHER_BENCH_FAIL '))

    def test_a_mapping_that_cannot_be_made_is_an_error(self):
        def refused(world):
            def mapper(host):
                raise RuntimeError('no unified addressing here')
            return mapper
        with patch.object(B, 'mapper_for', refused):
            code, record, _, _ = self.main()
        self.assertEqual((code, record['verdict']), (0, 'TOKEN_GATHER_BENCH_ERROR'))
        self.assertEqual(record['direct']['status'], 'DIRECT_UNAVAILABLE')
        self.assertIn('no unified addressing here', record['errors']['direct'])
        self.assertFalse(any(r['variant'] == 'direct' for r in record['rows']))
        self.assertEqual((record['decision']['label'], record['decision']['reason']), ('NO_DECISION',
                                                                                       'no direct row'))
        self.assertEqual(B.validate_record(record), [])


if __name__ == '__main__':
    unittest.main()

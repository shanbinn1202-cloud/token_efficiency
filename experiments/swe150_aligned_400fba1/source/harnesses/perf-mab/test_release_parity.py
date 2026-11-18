"""The harness against the sealed runtime's own code (a fresh clone of public-release.bundle; CPU only, no installed
release: serve.installed_check is stubbed as in perf-decode2).

- every planned runtime server's serve.py dry run through the sealed serve.py's own parser (never vLLM's): the vLLM
  command (K, the KV bytes, --max-model-len 262144, the layout flags, the CUDA-graph mode, --no-async-scheduling;
  --decode-fastpath / --release-idle-pages never reach vLLM), the startup config the checks expect (protocol
  kv_session_v1 and its record, page_tokens, kv_layout, cuda_graphs, max_tokens, dram_arena_pages, decode_fastpath
  and release_idle_pages only when on, the slots) and the launcher's pinned-host bytes = controller.pinned_host;
- the runner against the sealed kv-session protocol module: a whole read_next case (format error, two calls in one
  reply, a pad stop, a length-limited reply, the question) through the sealed parse_complete, registry and
  KvTranscript (as the frontend and the engine use them): every turn's prompt, stable end and observation span equal
  mab_checks' construction of the client's bodies and replies, the Goal inputs equal mab_checks.goal_source, the
  sealed info() options cover the runner's expectations; the constants the runner and the checks share (method,
  context-limit text, header, body keys);
- the evidence keys the checks read, in the sealed sources: the route's kv_session row and response fields, the
  runtime row's fields, the capacity events' fields (idle_release.py) - the same keys fake_server.py writes.
"""
import ast
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import fake_server as fs  # noqa: E402
import git_bundle  # noqa: E402
import job  # noqa: E402
import mab_checks  # noqa: E402
import mab_runner  # noqa: E402
import mab_workload  # noqa: E402

PINS = json.loads((ROOT / 'pins.json').read_bytes())


def dict_keys(source, target):
    """The keyword names of every `<target> = dict(...)` assignment in a source text."""
    keys = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == target for t in node.targets) \
                and isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', None) == 'dict':
            keys |= {k.arg for k in node.value.keywords if k.arg}
    return keys


class ReleaseParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.repo = base / 'repo'
        git_bundle.git(['-c', 'core.autocrlf=false', 'clone', '-q', '--branch', 'sealed-release',
                        ROOT / git_bundle.BUNDLE_NAME, cls.repo], stderr=subprocess.STDOUT)
        git_bundle.git(['-C', cls.repo, '-c', 'core.autocrlf=false', 'checkout', '-q', '--detach', PINS['commit']],
                       stderr=subprocess.STDOUT)
        cls.release = base / 'release'
        shutil.copytree(cls.repo / 'source-overlay', cls.release / 'source')
        cls.saved = list(sys.path)
        sys.path[:0] = [str(cls.repo), str(cls.release / 'source')]
        cls.serve = importlib.import_module('serve')
        cls.K = importlib.import_module('remote_ra_live_v2.kv_session_protocol')
        cls.WC = importlib.import_module('remote_ra_live_v2.window_capacity')
        cls.KL = importlib.import_module('remote_ra_live_v2.kv_layout')
        cls.MP = importlib.import_module('remote_ra_live_v2.model_profile')
        cls.manifest = dict(model_root='/m', model_revision='r' * 40, site='/s')

    @classmethod
    def tearDownClass(cls):
        sys.path[:] = cls.saved
        for name in [n for n in sys.modules if n.split('.')[0] in ('serve', 'install', 'remote_ra_live_v2',
                                                                    'remote_ra_http', 'runtime_renderer',
                                                                    'policy_registry')]:
            del sys.modules[name]
        cls.tmp.cleanup()

    def dry_run(self, server, name, changes=None):
        original, seen = self.serve.prepare, {}
        manifest = dict(self.manifest, model_profile=c.GEMMA_MODEL) if c.is_gemma(server) else self.manifest

        def capture(*args, **kwargs):
            seen['value'] = original(*args, **kwargs)
            return seen['value']
        argv = ['serve.py', '--release-dir', str(self.release), '--policy', server['policy'],
                '--output', str(Path(self.tmp.name) / ('dry-' + name)), '--port', '8000', '--protocol', 'kv-session']
        argv += c.launch_arguments(PINS, server)
        for flag, value in (changes or {}).items():     # a refusal probe: one flag's value changed
            argv[argv.index(flag) + 1] = value
        if c.REWRITER_SWITCH in argv:
            # The Goal rewriter checkpoint lives on DeltaAI; serve.py pins a directory with a config.json.
            local = Path(self.tmp.name) / 'goal-model'
            local.mkdir(exist_ok=True)
            (local / 'config.json').write_text('{"architectures": ["Qwen3ForCausalLM"]}', encoding='utf-8')
            argv[argv.index(c.REWRITER_SWITCH) + 1] = str(local)
        out = io.StringIO()
        with patch.object(sys, 'argv', argv), patch.object(self.serve, 'installed_check', return_value=manifest), \
                patch.object(self.serve, 'prepare', side_effect=capture), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.serve.main()
        value = json.loads(out.getvalue().splitlines()[-1])
        return value, seen['value']

    def gate_record(self, server, value):
        """controller.dry_run_record (the login node's serve-dry-run gate) on the sealed serve.py's printed value; the
        Goal rewriter is the local stand-in directory here."""
        args = list(server.get('policy_args') or ())
        if c.REWRITER_SWITCH in args:
            args[args.index(c.REWRITER_SWITCH) + 1] = str(Path(self.tmp.name) / 'goal-model')
        record = c.dry_run_record(PINS, dict(server, policy_args=args), value)
        self.assertEqual(record['status'], 'SERVICE_DRY_RUN')
        return record

    def test_every_planned_runtime_server(self):
        done = set()
        for plan in PINS['config']['plan_order']:
            for row in c.plan_servers(PINS, plan):
                server = c.server_entry(PINS, row['label'])
                if server['kind'] != 'runtime' or row['label'] in done or c.is_gemma(server):
                    continue
                done.add(row['label'])
                s = c.runtime_settings(PINS['config'], server)
                value, (manifest, config, command, env) = self.dry_run(server, row['label'])
                self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
                self.gate_record(server, value)
                goal = value.get('goal_rewriter')
                if server['policy'] == 'full':
                    self.assertIsNone(goal, row['label'])
                else:
                    self.assertIs(goal.get('used'), server['policy'] == 'goal_query', (row['label'], goal))
                if s.get('admission_overlap'):
                    self.assertEqual(config['admission']['admission_overlap']['permits'], s['admission_overlap'])
                command = [str(x) for x in value['command']]
                self.assertEqual((c.command_value(command, '--max-num-seqs'),
                                  c.command_value(command, '--kv-cache-memory-bytes'),
                                  c.command_value(command, '--max-model-len')),
                                 (str(s['max_running_sessions']), str(s['kv_cache_gib'] * c.GIB), str(s['max_model_len'])), row['label'])
                record = c.check_layout_command(command, s)
                for switch in ('--decode-fastpath', '--release-idle-pages'):
                    self.assertNotIn(switch, command)
                expect = job.expectation(PINS, server, 'm')
                self.assertEqual((config['protocol'], config.get('kv_layout'), record),
                                 ('kv_session_v1', expect['layout_record'], expect['layout_record']), row['label'])
                self.assertEqual((config['page_tokens'], config.get('cuda_graphs'), config.get('max_tokens'),
                                  config.get('dram_arena_pages'), config.get('max_running_sessions')),
                                 (16, None if s['cuda_graphs'] == 'off' else s['cuda_graphs'], 4096,
                                  expect['dram_arena_pages'], s['max_running_sessions']))
                self.assertEqual(config['kv_session'], dict(method='dram_backed_kv', budget=expect['budget'],
                                                            thinking=s['kv_session_thinking'],
                                                            header_text=c.kv_header(PINS, server),
                                                            slots=expect['slots']), row['label'])
                for key in ('decode_fastpath', 'release_idle_pages'):
                    self.assertEqual(config.get(key), True if s[key] else None, (row['label'], key))
                    self.assertEqual(key in config, s[key], (row['label'], key))
                self.assertEqual(config['effective_server_config']['max_model_len'], s['max_model_len'])
                arena = config.get('dram_arena_pages')
                pool = c.connector_pool_bytes(s)
                host = self.serve.pinned_host_bytes(s['max_running_sessions'], 16, True, s['max_model_len'], arena,
                                                    connector=pool if pool is not None else c.CONNECTOR_BYTES)
                self.assertLessEqual(host, self.serve.PINNED_HOST_GIB * c.GIB, row['label'])
                self.assertEqual((config.get('connector_pool_bytes'), config.get('language_model_only'),
                                  config['effective_server_config'].get('language_model_only')), (pool, True, True))
                c.check_text_only_command(command, True)
                self.assertEqual(host, c.pinned_host(s)['bytes'], row['label'])
                self.assertEqual(len(config['tasks']), expect['slots'])
                info = self.K.info(config)
                options = info['options']
                spec = job.runner_spec(Path('/stage'), PINS, server, 'http://x', Path('/out'), 60)
                self.assertEqual({k: options.get(k) for k in spec['expect_info']}, spec['expect_info'], row['label'])
                self.assertEqual((info['method'], info['budget']), (mab_runner.METHOD, spec['budget']))
        self.assertEqual(done, {s['label'] for s in PINS['config']['servers'] if s['kind'] == 'runtime'
                                and not c.is_gemma(s)})

    def test_shared_constants(self):
        K = self.K
        self.assertEqual((K.METHOD, K.CONTEXT_LIMIT, K.HEADERS['on']),
                         (mab_runner.METHOD, mab_runner.CONTEXT_LIMIT, mab_runner.HEADER_TEXT))
        self.assertEqual(mab_checks.CONTEXT_LIMIT, K.CONTEXT_LIMIT)
        kit = mab_runner.char_kit(fs.TOKENIZER_CONFIG)
        runner = type('R', (), dict(kit=kit, gen=dict(PINS['config']['generation']), budget=32768))()
        client = mab_runner.KvClient(runner, 'x')
        body, _, _ = client.build(list(mab_runner.WARMUP), None, 2)
        self.assertEqual(set(body), K.COMPLETE_KEYS)
        self.assertEqual(json.loads((self.repo / c.TEMPLATE_EVIDENCE).read_bytes()),
                         json.loads(fs.TOKENIZER_CONFIG.read_bytes()))
        self.assertEqual(kit.identity['template_sha256'], PINS['runtime_features']['kv_session']['template_sha256'])

    def test_evidence_keys(self):
        route = (self.repo / c.KV_SESSION_ROUTE).read_text(encoding='utf-8')
        row = dict_keys(route, 'row')
        for key in ('session_id', 'turn', 'header_tokens', 'prefix_tokens', 'boundary_tokens', 'input_tokens',
                    'max_tokens', 'effective_max_tokens', 'prompt_tokens', 'completion_tokens', 'cached_tokens',
                    'finish_reason', 'message_source', 'explicit_requirement'):
            self.assertIn(key, row)
        extra = dict_keys(route, 'extra')
        self.assertTrue({'kv_slot', 'kv_turn', 'effective_max_tokens', 'capacity_waits', 'queue_wait_s'} <= extra, extra)
        for key in ('selected_pages', 'candidate_pages', 'restored_h2d_bytes', 'prepare_h2d_bytes', 'selection_reason'):
            self.assertIn(key + '=', route)
        evidence = (self.repo / 'source-overlay/remote_ra_http/evidence.py').read_text(encoding='utf-8')
        for key in ('ordinal', 'request_id', 'session_id', 'turn', 'queue_wait_s', 'capacity_waits',
                    'prompt_token_ids', 'prompt_token_sha256', 'selected_logical_pages', 'candidate_page_count',
                    'stage_b', 'request_latency_ns', 'computed_start', 'phase2_start', 'prompt_end', 'computed_end',
                    'candidate_end', 'sealed_end'):
            self.assertRegex(evidence, r'\b%s=' % key)
        # check_route_b's after-completion rule: a turn commits exactly selected | [L/P, Q'/P) as its resident Pages
        # (allocation_owner._complete, inherited by the Gemma window owner); stage_b.sealed_end is Q'.
        owner = (self.repo / 'source-overlay/remote_ra_live_v2/allocation_owner.py').read_text(encoding='utf-8')
        self.assertIn('prefix, base = len(plan["selected"]), L // P', owner)
        self.assertIn('resident2 = tuple(sorted(set(plan["selected"]) | set(range(base, Q2 // P))))', owner)
        self.assertIn("sealed_end=len(done['dram'])*native['page_tokens']", evidence)
        self.assertIn('class WindowAllocationOwner(RunnerAllocationOwner):', (
            self.repo / 'source-overlay/remote_ra_live_v2/window_owner.py').read_text(encoding='utf-8'))
        self.assertIn("result['kv_session']", evidence)
        self.assertIn("result['idle_release']", evidence)
        release = (self.repo / c.IDLE_RELEASE_MODULE).read_text(encoding='utf-8')
        for key in ('schema=SCHEMA', "event='evict'", "event='readmit'", 'session_id=', 'session_index=', 'pages=',
                    'blocks=', 'request_id=', 'shortfall_blocks=', 'free_blocks_by_pool_before=',
                    'free_blocks_by_pool=', 'monotonic_ns='):
            self.assertIn(key, release)
        self.assertIn('evicted_sessions=[], evicted_pages=0', release)
        self.assertIn('readmitted_pages=[]', release)
        # check_route_b's window from the admission (certainly_running's admitted_ns): a session is a release victim
        # unless a lifecycle request, native request or phase-1 record exists (idle_release.idle), so its request
        # waiting for admission leaves it idle; the admission enters the lifecycle before it reserves, and the
        # reservation readmits before phase 1 - the readmission lies inside the admission.
        self.assertIn('state = lifecycle.sessions.get(sid, (0, None))[1]', release)
        self.assertIn('s.namespace in busy or state is not None or', release)
        lifecycle = (self.repo / 'source-overlay/remote_ra_live_v2/lifecycle.py').read_text(encoding='utf-8')
        entry, reserve = lifecycle.index('self.sessions[session] = (turn, rid)'), lifecycle.index('steps(request, binding)')
        self.assertLess(entry, reserve)
        endpoint = (self.repo / 'source-overlay/remote_ra_live_v2/private_endpoint.py').read_text(encoding='utf-8')
        self.assertLess(endpoint.index('release.readmit(binding, session)'), endpoint.index('phase1 = self.phase1.run('))
        # The optimized switches' echoes the checks read (mab_checks.check_fastpath_echo, check_deferred_completion_io,
        # check_goal_rewriter_batch): READY adds each pinned switch (the worker's fast-path flags compared with the
        # parent's), the completion writer and the batch with its cap; the Gemma window worker's receipt carries the
        # three fast paths, never the completion writer.
        for text in ("value.update(decode_fastpath2=True)", "value.update(decode_fastpath3=True)",
                     "value.update(deferred_completion_io=True)",
                     "value.update(goal_rewriter_batch=config['goal_rewriter_batch'],",
                     "goal_rewriter_batch_tokens=config['goal_rewriter_batch_tokens'])",
                     "require(receipt.get('decode_fastpath3')==config.get('decode_fastpath3')"):
            self.assertIn(text, evidence)
        worker = (self.repo / 'source-overlay/remote_ra_live_v2/window_worker.py').read_text(encoding='utf-8')
        for text in ('evidence[FASTPATH_KEY] = True', 'evidence[FASTPATH2_KEY] = True', 'evidence[FASTPATH3_KEY] = True'):
            self.assertIn(text, worker)
        self.assertNotIn('deferred_completion', worker)

    def test_runner_case_through_the_sealed_protocol(self):
        """One read_next case: the runner's bodies served by the sealed parse/registry/transcript (frontend and
        engine), the replies from fake_server's model with every turn shape (the global thinking switch: off)."""
        self.runner_case(None)

    def test_runner_case_thinking_on_through_the_sealed_protocol(self):
        """main ba4d1ba (decision D-B): the same case with smqt's kv-session thinking on: the runner spec of smqot
        (enable_thinking true, the thinking-on header), the sealed transcript with HEADERS['on'], the fake model's
        replies with their reasoning kept in the stream: every turn's prompt and spans as mab_checks constructs them."""
        server = c.server_entry(PINS, 'smqot')
        self.assertEqual(c.runtime_settings(PINS['config'], server)['kv_session_thinking'], 'on')
        streams = self.runner_case(server)
        self.assertTrue(any('</think>' in mab_runner.char_kit(fs.TOKENIZER_CONFIG).decode(list(s)) for s in streams))

    def runner_case(self, server):
        K = self.K
        kit = mab_runner.char_kit(fs.TOKENIZER_CONFIG)
        thinking = c.thinking_on(PINS['config'], server)            # the global switch, or the server's plan's (D-B)
        header = tuple(kit.encode(K.HEADERS['on' if thinking else 'off']))
        registry = K.KvSessionRegistry(4)
        front = [K.KvTranscript(header=header, im_start=kit.im_start, vocab=kit.vocab, decode=kit.decode,
                                stop_ids=frozenset({kit.eos, kit.pad})) for _ in range(4)]
        engine = [K.KvTranscript(header=header, im_start=kit.im_start, vocab=kit.vocab, decode=kit.decode,
                                 stop_ids=frozenset({kit.eos, kit.pad})) for _ in range(4)]
        model = fs.FakeModel(dict(format_turns={'x': [2]}, double_turns={'x': [3]}, pad_turns={'x': [1]},
                                  long_turns={'x': [5]}), 'x')
        served = []

        def serve(method, url, body=None, timeout=60):
            if url.endswith('/close'):
                return 200, dict(closed=True), 0, 0
            request = K.parse_complete(body, budget=None, bound=4096, max_model_len=262144)
            index = registry.resolve(request)
            t, e = front[index], engine[index]
            first = t.turn == 0
            t.check(request)
            limit = K.effective_max_tokens(request, t.prompt_length(request), 262144)
            new = K.new_messages(request, t.history, first)
            source = K.source(request, new, K.objective_of(request.messages) if first else None)
            t.stage(request)
            tokens, stable, span = t.render()
            echo = K.from_source(K.validate_source(json.loads(json.dumps(source))))
            self.assertEqual(e.render(echo), (tokens, stable, span))
            output, _ = model.reply(list(tokens), min(limit, 600))
            committed = t.commit(tokens, output, len(request.messages))
            e.commit(tokens, output)
            served.append(dict(tokens=tuple(tokens), stable=stable, span=span, context=K.goal_inputs(echo.messages, first),
                               output=list(output)))
            return 200, dict(text=committed['text'], output_ids=list(output), prompt_tokens=len(tokens),
                             completion_tokens=len(output), cached_tokens=len(tokens) - len(request.input_ids),
                             finish_reason=K.finish_reason(output, t.stop_ids, min(limit, 600)),
                             effective_max_tokens=limit), 1, 2
        case = mab_workload.synthetic_cases(dict(kind='synthetic', name='p', seed=1, cases=1, chunks=5,
                                                 chunk_chars=300))[0]
        with tempfile.TemporaryDirectory() as out, patch.object(mab_runner, 'http', side_effect=serve):
            base = c.runner_spec_base(PINS, server)
            self.assertEqual((base['generation']['enable_thinking'], base['header_text']),
                             (thinking, K.HEADERS['on' if thinking else 'off']))
            spec = dict(base, tokenizer=dict(kind='char', config=str(fs.TOKENIZER_CONFIG)),
                        mode='kv', label='x', base_url='http://x', out=out, agent_seconds=600, budget=None,
                        expect_info={}, workers=1, warmup=False, stop_on_failure=False, probes=False)
            spec['generation'] = dict(spec['generation'], max_tokens=600)
            runner = mab_runner.Runner(spec)
            row = mab_runner.run_case(runner, case, 0)
            runner.recorder.close()
            client = mab_checks.load_client(out)
        self.assertEqual(row['status'], 'answered', row)
        built = mab_checks.construct(mab_checks.kv_sessions(client['requests'])[row['sid']])
        self.assertEqual(len(built), len(served))
        boundaries = []
        for mine, theirs, request in zip(built, served, client['requests']):
            self.assertEqual(tuple(mine['T']), theirs['tokens'])
            geo = mine['geo']
            if geo['local_turn'] == 0:
                self.assertEqual((theirs['stable'], theirs['span']), (geo['o'], None))
            else:
                self.assertEqual((theirs['stable'], theirs['span']), (geo['o'], (geo['o'], geo['e'])))
            source, requirement = mab_checks.goal_source(request)
            self.assertEqual((theirs['context']['message_source'], theirs['context']['explicit_requirement']),
                             (source, requirement))
            boundaries.append(request['body']['boundary_tokens'])
        self.assertIn(3, boundaries)                          # a restored last sample (pad or length stop)
        self.assertEqual(served[-1]['context']['explicit_requirement'], case['query'].strip())
        return [s['tokens'] for s in served]

    def test_sealed_pins_of_the_repin(self):
        """integrate-1112 4ebff97: the sealed launcher's features read from the clone equal the pinned ones (the policy
        registry, async_fix with the resolve fix of job 3568311, async on both profiles), kv_session_protocol.py is
        289d3c60 (the pins' info() of the registry; b79d5498 at main ba4d1ba) and its chat template the one the harness
        renders (KV_SESSION_TEMPLATE_SHA256)."""
        import hashlib
        features = c.runtime_features(self.repo)
        self.assertEqual(features, PINS['runtime_features'])
        self.assertEqual(sorted(features[c.ASYNC_FIX_KEY]), ['native_probe', 'resolve', 'scheduler'])
        self.assertEqual(features[c.ASYNC_KEY]['profiles'], [c.QWEN_MODEL, c.GEMMA_MODEL])
        self.assertEqual(hashlib.sha256((self.repo / c.KV_SESSION_MODULE).read_bytes()).hexdigest()[:8], '289d3c60')
        self.assertEqual(features['kv_session']['template_sha256'], c.KV_SESSION_TEMPLATE_SHA256)

    def test_policy_registry_readers(self):
        """integrate-1112: the pinned registry rows are the sealed policy_registry.describe() (here in-process, the
        harness's reader in a child Python); the plans' policies, budgets and Goal engine come from them; the
        selection replay (postrun/selection_parity.py) reads the same rows from the source tree: POLICIES = the
        Selector V2 policies, replayed = observation_query and random_fill, the record keys from evidence_key."""
        registry = importlib.import_module('policy_registry')
        rows = PINS['runtime_features'][c.REGISTRY_KEY]
        self.assertEqual(rows, json.loads(json.dumps(registry.describe())))
        self.assertEqual(c.policy_registry_feature(self.repo), rows)
        self.assertEqual(PINS['runtime_features']['policies'], list(registry.names()))
        self.assertEqual(c.runtime_policies(PINS['runtime_features']),
                         tuple(p.name for p in registry.policies() if p.budget in ('none', 'fixed')))
        for server in PINS['config']['servers']:
            if server['kind'] == c.RUNTIME:
                policy = registry.get(server['policy'])
                self.assertEqual((c.active_budget(server) is not None, c.goal_engine(PINS, server['policy'])),
                                 (policy.budget == 'fixed', policy.goal_engine), server['label'])
        sys.path.insert(0, str(ROOT / 'postrun'))
        try:
            parity = importlib.import_module('selection_parity')
        finally:
            sys.path.remove(str(ROOT / 'postrun'))
        found = parity.registry(self.release / 'source')
        self.assertEqual(found, {row['name']: row for row in rows})
        self.assertEqual(parity.policies(found), tuple(p.name for p in registry.policies() if p.selector_v2))
        self.assertEqual([n for n in parity.policies(found) if parity.replayable(found[n])],
                         ['observation_query', 'random_fill'])
        self.assertEqual((found['random_fill']['evidence_key'], found['observation_query']['evidence_key']),
                         ('fill', None))
        with self.assertRaisesRegex(ValueError, 'is not replayed'):
            parity.replay_server((None, None, None), self.write_ready('goal_query'), found)

    def write_ready(self, policy):
        out = Path(self.tmp.name) / ('ready-' + policy)
        out.mkdir(exist_ok=True)
        (out / 'ready.json').write_text(json.dumps(dict(policy=policy, selection_contract={})), encoding='utf-8')
        return out


    # ---- Gemma 4 (the gm plans): the sealed launcher's prepare_gemma4 and the protocol's Gemma grammar

    def test_every_planned_gemma_server(self):
        """Every Gemma runtime server through the sealed serve.py (a release manifest naming the Gemma profile): the
        vLLM command (K, the KV bytes, the context, the window layout with request_pages under the selective rule,
        TRITON_ATTN, the gemma4 parsers, 8,192 batched tokens, no Mamba / GDN flag, --host-memory-shares and the
        runtime switches never reaching vLLM), the startup config the checks read (model_profile, kv_session_v1 with
        the gemma4 grammar and header, the effective config of check_gemma_profile), the pools and the pinned host of
        the sealed window_capacity against controller.gemma_pool_fit / gemma_pinned_host, info against the spec."""
        done = set()
        for server in PINS['config']['servers']:
            if server['kind'] != 'runtime' or not c.is_gemma(server):
                continue
            label = server['label']
            done.add(label)
            s = c.runtime_settings(PINS['config'], server)
            value, (manifest, config, command, env) = self.dry_run(server, label)
            self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
            gate = self.gate_record(server, value)
            self.assertEqual((gate['kv_layout'], gate['connector_pool_bytes']), (config['kv_layout'], c.GIB))
            command = [str(x) for x in value['command']]
            self.assertEqual((c.command_value(command, '--max-num-seqs'), c.command_value(command, '--kv-cache-memory-bytes'),
                              c.command_value(command, '--max-model-len')),
                             (str(s['max_running_sessions']), str(s['kv_cache_gib'] * c.GIB), str(s['max_model_len'])))
            record = c.check_gemma_layout_command(command, s)
            for switch in ('--decode-fastpath', '--release-idle-pages', c.HOST_SHARES_SWITCH):
                self.assertNotIn(switch, command)
            c.check_text_only_command(command, True)
            expect = job.expectation(PINS, server, 'm')
            self.assertEqual((config['protocol'], config['model_profile'], config['kv_layout'], record),
                             ('kv_session_v1', c.GEMMA_MODEL, expect['layout_record'], expect['layout_record']), label)
            self.assertEqual(config['kv_session'], dict(method='dram_backed_kv', budget=expect['budget'], thinking='off',
                                                        header_text=c.GEMMA_HEADER, slots=expect['slots'],
                                                        grammar='gemma4'), label)
            effective = config['effective_server_config']
            self.assertEqual({k: effective.get(k) for k in mab_checks.GEMMA_EFFECTIVE}, mab_checks.GEMMA_EFFECTIVE)
            self.assertEqual((effective['block_size'], effective['additional_config']),
                             (16, {mab_checks.LAYOUT_KEY: record}))
            self.assertEqual((config['page_tokens'], config.get('cuda_graphs'), config.get('max_tokens'),
                              config.get('dram_arena_pages'), config.get('max_running_sessions'),
                              config.get('connector_pool_bytes'), config.get('language_model_only')),
                             (16, None if s['cuda_graphs'] == 'off' else s['cuda_graphs'], 4096,
                              expect['dram_arena_pages'], s['max_running_sessions'], c.GIB, True), label)
            for key in ('decode_fastpath', 'release_idle_pages'):
                self.assertEqual(key in config, s[key], (label, key))
            # integrate-1021's optimized switches: pinned exactly when the server runs them (never a vLLM argument);
            # the rewriter batch with the Gemma profile's default token cap by --kv-cache-gib (25 GiB: 25,600).
            for key in (c.FASTPATH2_KEY, c.FASTPATH3_KEY, c.DC_KEY):
                self.assertEqual(config.get(key), True if s[key] else None, (label, key))
                self.assertEqual(expect[key], s[key], (label, key))
            for flag in c.OPTIMIZED_SWITCHES:
                self.assertNotIn(flag, command)
            batch = (config.get(c.BATCH_KEY), config.get(c.BATCH_TOKENS_KEY))
            self.assertEqual(batch, (expect[c.BATCH_KEY], expect[c.BATCH_TOKENS_KEY]), label)
            self.assertEqual(batch, (gate[c.BATCH_KEY], gate[c.BATCH_TOKENS_KEY]), label)
            if server.get(c.BATCH_KEY) is not None:
                self.assertEqual(batch, (4, self.serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS[s['kv_cache_gib']]), label)
                self.assertEqual(batch[1], 25600, label)
                self.assertNotIn(c.BATCH_TOKENS_SWITCH, c.launch_arguments(PINS, server))   # the default, not passed
            row = c.policy_row(PINS, server['policy'])   # integrate-1112: the registry's own text
            if not row['goal_engine'] and row['rewriter_unused'] is not None:
                self.assertEqual(value['goal_rewriter'], dict(used=False, reason=row['rewriter_unused'],
                                                              requested_model=None))
            # The pools and the pinned host by the sealed window_capacity.
            fit = c.gemma_pool_fit(s)
            layout = self.KL.from_vllm_config(type('V', (), dict(additional_config={mab_checks.LAYOUT_KEY: record}))())
            g = self.MP.GEMMA4_GEOMETRY
            groups, block, page = len(g.sliding_groups()), g.sliding_block_bytes(16), g.page_bytes(16)
            self.assertEqual((self.KL.window_pool_blocks(layout, groups),
                              self.WC.pool_zero_blocks(s['kv_cache_gib'] * c.GIB, layout, groups, block, page)),
                             (fit['pool_one_blocks'], fit['blocks']), label)
            self.assertEqual(expect['pools'], [fit['blocks'], fit['pool_one_blocks']])
            host = self.WC.pinned_host_bytes(s['max_running_sessions'], 16, page, s['max_model_len'],
                                             config.get('dram_arena_pages'), connector_bytes=c.GIB)
            shares = s.get('host_memory_shares') or 1
            self.assertEqual(host, c.gemma_pinned_host(s)['bytes'], label)
            self.assertAlmostEqual(c.gemma_pinned_host(s)['budget_gib'], self.WC.pinned_budget_bytes(shares) / c.GIB,
                                   places=2)
            self.assertTrue(self.WC.host_fits(host, shares) and c.gemma_pinned_host(s)['fits'], label)
            info = self.K.info(config)
            spec = job.runner_spec(Path('/stage'), PINS, server, 'http://x', Path('/out'), 60)
            self.assertEqual({k: info['options'].get(k) for k in spec['expect_info']}, spec['expect_info'], label)
            self.assertEqual((spec['grammar'], spec['header_text'], info['method'], info['budget']),
                             ('gemma4', c.GEMMA_HEADER, mab_runner.METHOD, spec['budget']))
        self.assertEqual(done, {'gm1a', 'gm1b', 'gm4o', 'gm4g', 'gm4r', 'smgo', 'smgr', 'smgg', 'smgh', 'smgoa', 'smgra',
                                'smgga'})

    def test_release_smoke_servers(self):
        """integrate-1103's release smoke servers through the sealed serve.py (smqo / smqr / smqg on Qwen, smgo / smgr
        / smgg / smgh on Gemma): every speed switch named, so no speed default applies (no speed_defaults record) and
        the three decode fast paths, the completion writer and --language-model-only are pinned; --async-scheduling
        exactly on smq's servers (vLLM's --async-scheduling, startup.json and effective_server_config); the vLLM Goal
        rewriter's record equal to controller.engine_settings on the contract's checkpoint, and the admission record
        (--admission-overlap, --goal-overlap) equal to controller.admission_expected; the login gate's record; and every
        earlier server: no speed_defaults either (its serve.py arguments name every switch)."""
        for plan in PINS['config']['plan_order']:
            for label in PINS['config']['plans'][plan]['servers']:
                server = c.server_entry(PINS, label)
                if server['kind'] != 'runtime':
                    continue
                value, _ = self.dry_run(server, 'all-' + plan + '-' + label)
                self.assertNotIn(c.SPEED_RECORD_KEY, value, (plan, label))
        for label in ('smqo', 'smqr', 'smqg', 'smgo', 'smgr', 'smgg', 'smgh', 'smqot', 'smqrt', 'smqgt', 'smgoa', 'smgra',
                      'smgga'):
            server = c.server_entry(PINS, label)
            s = c.runtime_settings(PINS['config'], server)
            value, (manifest, config, command, env) = self.dry_run(server, 'smoke-' + label)
            self.assertNotIn(c.SPEED_RECORD_KEY, value, label)
            gate = self.gate_record(server, value)
            command = [str(x) for x in value['command']]
            for key in ('decode_fastpath', c.FASTPATH2_KEY, c.FASTPATH3_KEY, c.DC_KEY, 'language_model_only',
                        'release_idle_pages'):
                self.assertIs(config.get(key), True, (label, key))
            qwen = label.startswith('smq')
            # main ba4d1ba: async on smq's / smqt's Qwen servers and on smga's Gemma servers (async-gemma, D-C)
            on = qwen or label in ('smgoa', 'smgra', 'smgga')
            self.assertEqual((config.get(c.ASYNC_KEY), config['effective_server_config'].get(c.ASYNC_KEY)),
                             (True, True) if on else (None, None), label)
            self.assertEqual((command.count('--async-scheduling'), command.count('--no-async-scheduling')),
                             (1, 0) if on else (0, 1), label)
            self.assertEqual(gate['async_scheduling'], on)
            # D-B: kv-session thinking on smqt's servers only (the sealed serve.py's record and header)
            thinking = label in ('smqot', 'smqrt', 'smqgt')
            self.assertEqual((config['kv_session']['thinking'], config['kv_session']['header_text']),
                             ('on', self.K.HEADERS['on']) if thinking else
                             ('off', c.GEMMA_HEADER if not qwen else self.K.HEADERS['off']), label)
            engine = c.engine_settings(PINS, server)
            if engine is None:
                self.assertNotIn(c.ENGINE_KEY, config)
            else:
                model = config['selection_contract']['goal_rewriter']['model']
                self.assertEqual(c.engine_record_differences(config[c.ENGINE_KEY], engine, model), [], label)
                self.assertEqual(mab_checks.engine_record_differences(config[c.ENGINE_KEY], engine, model), [])
                self.assertEqual(value[c.ENGINE_KEY], config[c.ENGINE_KEY])
                self.assertEqual((config[c.ENGINE_KEY]['batch_invariant'], config[c.ENGINE_KEY]['guided_backend'],
                                  config[c.ENGINE_KEY]['kv_cache_memory_bytes']), (True, 'lm-format-enforcer', 3 * c.GIB))
            self.assertEqual(config.get('admission'), c.admission_expected(server), label)
            self.assertEqual(job.expectation(PINS, server, 'm')['admission'], config.get('admission'))
            if server['policy'] == 'goal_query':
                self.assertEqual(config['admission']['admission_overlap']['permits'], 16 if qwen else 8)
                self.assertEqual(config['admission'].get('goal_overlap'), None if label == 'smgh' else True, label)
                self.assertEqual((config.get(c.BATCH_KEY), config.get(c.BATCH_TOKENS_KEY)),
                                 (4, 25600) if label == 'smgh' else (None, None), label)
            else:
                self.assertNotIn('admission', config)
        # Gemma at the documented concurrency only: 16 sessions do not fit 25 GiB at the 32K budget (the selective
        # rule), with or without the Goal transient; the sealed launcher refuses it as the harness does.
        for label, needed, usable in (('smgo', 5699, 4957), ('smgg', 5686, 3677)):
            server = c.server_entry(PINS, label)
            fit = c.gemma_pool_fit(dict(c.runtime_settings(PINS['config'], server), max_running_sessions=16))
            self.assertEqual((fit['fits'], fit['needed'], fit['usable']), (False, needed, usable), label)
            with self.assertRaisesRegex(ValueError, 'with 16 sessions leaves %d usable global Pages' % usable):
                self.dry_run(server, 'smoke16-' + label, {'--max-running-sessions': '16'})

    def test_gemma_refusals_agree(self):
        """The sealed launcher and the harness refuse the same Gemma settings: gm4o with 16 sessions at 25 GiB (the
        selective rule at the 32K budget: 5,699 Pages needed, 4,957 usable - why the end-to-end runs use 8), gm4g's
        rewriter batch at 32 GiB without an explicit token cap (the Gemma profile has no default there: job 3410948),
        gm4o's 155-GiB arena on one memory share (pinned budget 80.9 GiB) and gm1a at 262,144 tokens (the strict rule:
        17,325 Pages > 17,164)."""
        cases = (('gm4o', {'--max-running-sessions': '16'}, 'with 16 sessions leaves 4957 usable global Pages'),
                 ('gm4g', {'--kv-cache-gib': '32'}, 'needs an explicit --goal-rewriter-batch-tokens T'),
                 ('gm4o', {'--dram-arena-gib': '155'}, 'more than the 80.9 GiB'),
                 ('gm1a', {'--max-model-len': '262144'}, 'one session at the 262144-token limit'))
        for label, changes, text in cases:
            server = c.server_entry(PINS, label)
            with self.assertRaisesRegex(ValueError, text):
                self.dry_run(server, label + '-refused-%d' % len(changes) + list(changes)[0][2:6], changes)
            s = dict(c.runtime_settings(PINS['config'], server))
            if '--max-running-sessions' in changes:
                fit = c.gemma_pool_fit(dict(s, max_running_sessions=16))
                self.assertEqual((fit['fits'], fit['needed'], fit['usable']), (False, 5699, 4957))
            elif '--kv-cache-gib' in changes:
                self.assertIsNone(c.batch_tokens_of(PINS, dict(server, kv_cache_gib=32)))
            elif '--dram-arena-gib' in changes:
                self.assertFalse(c.gemma_pinned_host(dict(s, dram_arena_gib=155))['fits'])
            else:
                s['max_model_len'] = 262144
                self.assertFalse(c.gemma_pool_fit(s)['fits'])
                self.assertEqual(c.gemma_pool_fit(s)['needed'], 17325)

    def test_gemma_shared_constants(self):
        K, MP = self.K, self.MP
        self.assertEqual((K.GEMMA4, K.GEMMA4_HEADER, K.GEMMA4_TURN_START, K.GEMMA4_TOOL_OPEN, K.GEMMA4_TOOL_CLOSE),
                         (mab_runner.GEMMA4, mab_runner.GEMMA4_HEADER, mab_runner.GEMMA4_TURN_START,
                          mab_runner.GEMMA4_TOOL_OPEN, mab_runner.GEMMA4_TOOL_CLOSE))
        self.assertEqual((c.GEMMA_HEADER, c.GEMMA_STOP_IDS, c.GEMMA_ATTENTION_BACKEND, c.GEMMA_TOKEN_BYTES),
                         (K.GEMMA4_HEADER, sorted(MP.GEMMA4_STOP_IDS), MP.GEMMA4_ATTENTION_BACKEND,
                          MP.GEMMA4_GEOMETRY.global_token_bytes))
        self.assertEqual(MP.GEMMA4_MODEL_PINS, PINS['config']['gemma']['model']['metadata'])
        self.assertEqual(c.gemma4_feature(self.repo, (self.repo / 'serve.py').read_text(encoding='utf-8')),
                         PINS['runtime_features']['gemma4'])
        kit = fs.gemma_kit()
        self.assertEqual((kit.eos, kit.pad, kit.im_start, kit.im_end, sorted(kit.stop_ids)),
                         (MP.EOS, MP.PAD, MP.TURN_OPEN, MP.TURN_CLOSE, sorted(MP.GEMMA4_STOP_IDS)))
        self.assertEqual(kit.identity['template_sha256'], PINS['config']['gemma']['model']['template_sha256'])

    def test_gemma_runner_case_through_the_sealed_protocol(self):
        """One read_next case in Gemma's grammar: the runner's bodies served by the sealed parse_complete (with the
        empty header of a continuation after tool responses), registry and KvTranscript with Gemma's tool delimiters
        (frontend and engine): every turn's prompt, stable end and observation span equal mab_checks' construction
        (e = N - 1 after tool responses), every later body passes the Gemma turn rule, the Goal inputs equal
        mab_checks.goal_source; the replies of fake_server's Gemma model with every turn shape."""
        K = self.K
        kit = fs.gemma_kit()
        header = tuple(kit.encode(K.GEMMA4_HEADER))
        stops = frozenset(kit.stop_ids)
        tools = tuple(kit.encode(K.GEMMA4_TOOL_OPEN + K.GEMMA4_TOOL_CLOSE))
        registry = K.KvSessionRegistry(4)
        new = lambda: K.KvTranscript(header=header, im_start=kit.im_start, vocab=kit.vocab, decode=kit.decode,
                                     stop_ids=stops, tools=tools)
        front, engine = [new() for _ in range(4)], [new() for _ in range(4)]
        model = fs.GemmaFakeModel(dict(format_turns={'x': [2]}, double_turns={'x': [3]}, pad_turns={'x': [1]},
                                       long_turns={'x': [5]}), 'x')
        served = []

        def serve(method, url, body=None, timeout=60):
            if url.endswith('/close'):
                return 200, dict(closed=True), 0, 0
            request = K.parse_complete(body, budget=None, bound=4096, max_model_len=131072, empty_header=True)
            index = registry.resolve(request)
            t, e = front[index], engine[index]
            first = t.turn == 0
            t.check(request)
            limit = K.effective_max_tokens(request, t.prompt_length(request), 131072)
            messages = K.new_messages(request, t.history, first)
            source = K.source(request, messages, K.objective_of(request.messages) if first else None)
            t.stage(request)
            tokens, stable, span = t.render()
            echo = K.from_source(K.validate_source(json.loads(json.dumps(source))))
            self.assertEqual(e.render(echo), (tokens, stable, span))
            output, _ = model.reply(list(tokens), min(limit, 600))
            committed = t.commit(tokens, output, len(request.messages))
            e.commit(tokens, output)
            served.append(dict(tokens=tuple(tokens), stable=stable, span=span, context=K.goal_inputs(echo.messages, first)))
            return 200, dict(text=committed['text'], output_ids=list(output), prompt_tokens=len(tokens),
                             completion_tokens=len(output), cached_tokens=len(tokens) - len(request.input_ids),
                             finish_reason=K.finish_reason(output, t.stop_ids, min(limit, 600)),
                             effective_max_tokens=limit), 1, 2
        case = mab_workload.synthetic_cases(dict(kind='synthetic', name='p', seed=1, cases=1, chunks=5,
                                                 chunk_chars=300))[0]
        gm1a = c.server_entry(PINS, 'gm1a')
        with tempfile.TemporaryDirectory() as out, patch.object(mab_runner, 'http', side_effect=serve):
            spec = dict(c.runner_spec_base(PINS, gm1a), tokenizer=dict(kind='char', config=str(fs.GEMMA_TOKENIZER_CONFIG)),
                        mode='kv', label='x', base_url='http://x', out=out, agent_seconds=600, budget=None,
                        expect_info={}, workers=1, warmup=False, stop_on_failure=False, probes=False)
            spec['generation'] = dict(spec['generation'], max_tokens=600)
            runner = mab_runner.Runner(spec)
            row = mab_runner.run_case(runner, case, 0)
            runner.recorder.close()
            client = mab_checks.load_client(out)
        self.assertEqual(row['status'], 'answered', row)
        built = mab_checks.construct(mab_checks.kv_sessions(client['requests'])[row['sid']])
        self.assertEqual(len(built), len(served))
        headers, boundaries = [], []
        for mine, theirs, request in zip(built, served, client['requests']):
            self.assertEqual(tuple(mine['T']), theirs['tokens'])
            geo = mine['geo']
            if geo['local_turn'] == 0:
                self.assertEqual((theirs['stable'], theirs['span']), (geo['o'], None))
            else:
                self.assertEqual((theirs['stable'], theirs['span']), (geo['o'], (geo['o'], geo['e'])))
                self.assertEqual(mab_checks.gemma_boundary_problems(request['body'], len(header)), [])
            source, requirement = mab_checks.goal_source(request)
            self.assertEqual((theirs['context']['message_source'], theirs['context']['explicit_requirement']),
                             (source, requirement))
            headers.append(request['body']['header_tokens'])
            boundaries.append(request['body']['boundary_tokens'])
        self.assertIn(0, headers[1:])                        # a continuation after tool responses: no header
        self.assertIn(len(header), headers[1:])              # a new user turn: the 7-token header (here in characters)
        self.assertIn(2, boundaries)                         # a restored last sample (the pad stop)
        self.assertEqual(served[-1]['context']['explicit_requirement'], case['query'].strip())


if __name__ == '__main__':
    unittest.main()

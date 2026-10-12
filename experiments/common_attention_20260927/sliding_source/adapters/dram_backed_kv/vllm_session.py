"""Async transport around the original OnlineSelectorSession policy.

Observation prefill precedes selection, exactly as in the reference. The
assistant header is prefetched after restoring the selected historical KV.
One-token staging samples are discarded before computation by native streaming.
"""
import asyncio
from hashlib import sha256
import json
import os
import time
import uuid

from .vllm_policy import Policy, upstream


class Controller:
    def __init__(self, session, body):
        self.session = session
        self.engine = session.engine
        self.tokenizer = self.engine.get_tokenizer()
        self.options = json.loads(os.environ['VLLM_DRAM_KV_OPTIONS'])
        c = self.engine.model_config.hf_text_config
        self.policy = Policy(self.tokenizer, c, self.engine.model_config.model,
            body['budget'], self.options, session.rid)
        self.policy.root_goal = upstream._root_objective(body['messages'])
        self.policy.instance_id = sha256(self.policy.root_goal.encode()).hexdigest()[:16]
        self.policy.mandatory_end = body['prefix_tokens']
        if not hasattr(self.engine, '_dram_goal_lock'):
            self.engine._dram_goal_lock = asyncio.Semaphore(1)

    async def _stage(self, ids, body, *, max_tokens, selected=None, tail_start=None, seal=True):
        from vllm import SamplingParams
        from vllm.engine.protocol import StreamingInput
        from vllm.sampling_params import RequestOutputKind
        s = self.session
        if not ids:
            raise ValueError('Empty DRAM stage')
        meta = dict(session_id=s.rid, logical=s.logical, selected=selected)
        if selected is not None:
            meta['tail_start'] = tail_start
        params = SamplingParams(max_tokens=max_tokens, temperature=body['temperature'],
            top_p=body.get('top_p', 1.0), top_k=body.get('top_k', -1),
            repetition_penalty=body.get('repetition_penalty', 1.0), seed=body.get('seed', 0),
            skip_special_tokens=False, output_kind=RequestOutputKind.DELTA,
            extra_args={'dram_kv': meta})
        s.future = asyncio.get_running_loop().create_future()
        await s.queue.put(StreamingInput(prompt={'prompt_token_ids': ids}, sampling_params=params))
        if s.task is None:
            s.task = asyncio.create_task(s.consume(params))
        outputs, text, reason = await s.future
        s.logical += len(ids) + len(outputs) - 1
        self.policy.append(ids)
        self.policy.append(outputs[:-1])
        stats, = await self.engine.collective_rpc('kv_session_stats', args=(s.rid, seal))
        if stats['logical_computed_tokens'] != s.logical or len(self.policy.logical_tokens) != s.logical:
            raise ValueError('DRAM frontend and worker logical counters disagree')
        self.policy.ingest(stats)
        self.policy._add_page_records(s.logical)
        self.policy.sealed_end = stats['sealed_end']
        s.computed = stats['retained_kv_tokens']
        return outputs, text, reason, stats

    async def _query(self, goal, observation_vector, required):
        p = self.policy
        metrics = dict(source=p.args.query_source, required=required, published=False,
            cache_hit=False, token_count=0, forward_ms=0.0)
        if not required:
            return metrics
        if p.args.query_source == 'observation':
            raw = observation_vector
        else:
            rendered = upstream._goal_query_text(goal)
            digest = sha256(rendered.encode()).hexdigest()
            cached = p._goal_query_cache.get(goal.retrieval_goal_hash)
            if cached is not None:
                raw, count, cached_digest = cached
                if cached_digest != digest:
                    raise ValueError('Goal hash collision changed the query payload')
                metrics.update(cache_hit=True, token_count=count)
            else:
                from vllm import SamplingParams
                from vllm.sampling_params import RequestOutputKind
                ids = self.tokenizer.encode(rendered, add_special_tokens=False)
                rid = 'dram-goal-' + uuid.uuid4().hex
                params = SamplingParams(max_tokens=1, temperature=0,
                    output_kind=RequestOutputKind.FINAL_ONLY,
                    extra_args={'dram_kv': {'session_id': rid, 'query': True}})
                started = time.perf_counter()
                async with self.engine._dram_goal_lock:
                    try:
                        async for _ in self.engine.generate({'prompt_token_ids': ids}, params, rid):
                            pass
                        report, = await self.engine.collective_rpc('kv_session_stats', args=(rid, False))
                        raw = report['last_vector']
                    finally:
                        await self.engine.collective_rpc('dram_release', args=(rid,))
                metrics.update(token_count=len(ids), forward_ms=(time.perf_counter() - started) * 1000)
                p._goal_query_cache[goal.retrieval_goal_hash] = raw, len(ids), digest
        vector = upstream.CompactVector(tuple(raw['values']), raw['dtype'], raw['scale'])
        p.publisher.publish_query(p.session_id, p.turn_id, vector)
        metrics['published'] = True
        return metrics

    async def complete(self, body):
        s, p = self.session, self.policy
        ids = list(body['input_ids'])
        header = body['header_tokens']
        if not ids or not 0 < header <= len(ids):
            raise ValueError('Invalid DRAM input/header')
        if s.logical + len(ids) + body['max_tokens'] > self.engine.model_config.max_model_len:
            raise ValueError('Logical RoPE context limit reached')
        first = p.turn_id == 0
        request_start = s.logical
        observation = str(body['messages'][-1].get('content', ''))
        if first:
            p.turn_id = 1
            context = p._context(observation)
            goal = p.goal_constructor.construct(context)
            plan = p._page_plan(context, 'cold_start')
            prompt_tokens = len(ids)
            cached_tokens = 0
            query = dict(required=False, published=False, token_count=0, forward_ms=0.0)
            outputs, text, reason, stats = await self._stage(ids, body, max_tokens=body['max_tokens'])
        else:
            boundary = int(body['boundary_tokens'])
            if not 0 < boundary < len(ids) - header:
                raise ValueError('Expected previous assistant boundary and current observation')
            # Finish the previous sampled token/chat boundary with its previous
            # turn label before cataloguing the new observation.
            await self._stage(ids[:boundary], body, max_tokens=1)
            tail_start = p.sealed_end
            p.turn_id += 1
            obs_ids = ids[boundary:-header]
            _, _, _, observed = await self._stage(obs_ids, body, max_tokens=1, seal=False)
            context = p._context(observation)
            goal = p.goal_constructor.construct(context)
            query = await self._query(goal, observed['last_vector'],
                len(context.historical_page_ids) * 16 > p.args.switch_threshold)
            plan = p._plan(context, goal)
            plan.validate_against(context, p._catalog())
            if plan.standby_page_ids:
                raise ValueError('Online Selector V2 unexpectedly returned standby pages')
            selected = list(plan.ordered_page_ids)
            prompt_tokens = len(selected) * 16 + s.logical - tail_start + header
            historical_positions = [pos for pid in selected
                for pos in range(*map(int, pid.split(':')[1].split('-')))]
            cached_tokens = sum(pos < request_start for pos in historical_positions)
            cached_tokens += max(0, request_start - tail_start)
            outputs, text, reason, stats = await self._stage(ids[-header:], body,
                max_tokens=body['max_tokens'], selected=selected, tail_start=tail_start)
        p.previous_active = tuple(stats['resident'])
        p.ever_selected.update(plan.ordered_page_ids)
        p.previous_action = self.tokenizer.decode(outputs, skip_special_tokens=True)
        public_stats = {key: value for key, value in stats.items()
                        if key not in {'vectors', 'last_vector', 'resident'}}
        return dict(text=text, output_ids=outputs, finish_reason=reason,
            prompt_tokens=prompt_tokens, completion_tokens=len(outputs),
            cached_tokens=cached_tokens, prefill_tokens=len(ids),
            selected_pages=len(plan.ordered_page_ids), selected_tokens=16 * len(plan.ordered_page_ids),
            selection=plan.selector, selector_config=dict(plan.selector_config),
            goal=upstream._goal_metrics(goal),
            query_embedding=query, auxiliary_calls=int(query['published'] and not query.get('cache_hit', False)
                and p.args.query_source == 'goal'), **public_stats)

    async def close(self):
        self.policy.close()
        await self.engine.collective_rpc('dram_release', args=(self.session.rid,))

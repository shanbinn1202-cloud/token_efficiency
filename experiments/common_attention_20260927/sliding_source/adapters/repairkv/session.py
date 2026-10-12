"""Turn protocol: commit final sample, base eviction, probe, repair, replay."""
import uuid


class Controller:
    def __init__(self, session, body):
        self.session = session
        self.engine = session.engine
        self.previous_outputs = None
        self.history = []
        self.active = dict(logical=0, physical=0)

    async def stage(self, ids, body, *, max_tokens, mode='generate', restore=None, rewind=False):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        s = self.session
        if not ids:
            raise ValueError('Empty RepairKV stage')
        meta = dict(session_id=s.rid, mode=mode, **self.active)
        if restore is not None:
            meta.update(restore=True, logical=restore['logical'], physical=restore['physical'], rewind=rewind)
            if rewind:
                s.logical = restore['logical']
                del self.history[s.logical:]
        params = SamplingParams(max_tokens=max_tokens, temperature=body['temperature'],
            top_p=body.get('top_p', 1.), top_k=body.get('top_k', -1),
            repetition_penalty=body.get('repetition_penalty', 1.), seed=body.get('seed', 0),
            ignore_eos=bool(body.get('ignore_eos', False)),
            skip_special_tokens=False, output_kind=RequestOutputKind.DELTA,
            extra_args={'repairkv': meta})
        outputs, text, reason = [], '', None
        rid = s.rid + '-' + uuid.uuid4().hex
        async for result in self.engine.generate(
                {'prompt_token_ids': self.history + ids}, params, rid):
            for part in result.outputs:
                outputs.extend(part.token_ids)
                text += part.text
                reason = part.finish_reason or reason
        s.logical += len(ids) + len(outputs) - 1
        self.history.extend(ids + outputs[:-1])
        if len(self.history) != s.logical:
            raise ValueError('RepairKV token history and logical counter disagree')
        stats, = await self.engine.collective_rpc('kv_session_stats', args=(s.rid,))
        if stats['logical_computed_tokens'] != s.logical:
            raise ValueError('RepairKV frontend/worker logical counters disagree')
        self.active = dict(logical=s.logical, physical=stats['retained_kv_tokens'])
        return outputs, text, reason, stats

    async def close(self):
        await self.engine.collective_rpc('repairkv_release', args=(self.session.rid,))

    async def complete(self, body):
        s = self.session
        ids = list(body['input_ids'])
        if s.logical + len(ids) + body['max_tokens'] > self.engine.model_config.max_model_len:
            raise ValueError('Logical RoPE context limit reached')
        incoming = len(ids)
        if self.previous_outputs is None:
            cached = 0
            outputs, text, reason, stats = await self.stage(ids, body, max_tokens=body['max_tokens'])
        else:
            # Upstream generate_turn computes the last generated token too.
            # Native streaming leaves it uncomputed; commit it before eviction.
            if ids[0] != self.previous_outputs[-1]:
                raise ValueError('RepairKV continuation did not restore the exact final sampled token')
            await self.stage(ids[:1], body, max_tokens=1, mode='boundary')
            ids = ids[1:]
            base, = await self.engine.collective_rpc('repairkv_prepare',
                args=(s.rid, len(self.previous_outputs)))
            if base['candidates']:
                await self.stage(ids, body, max_tokens=1, mode='probe', restore=base)
                repaired, = await self.engine.collective_rpc('repairkv_select', args=(s.rid,))
                outputs, text, reason, stats = await self.stage(ids, body,
                    max_tokens=body['max_tokens'], restore=repaired, rewind=True)
                cached = repaired['physical']
            else:
                outputs, text, reason, stats = await self.stage(ids, body,
                    max_tokens=body['max_tokens'], restore=base)
                cached = base['physical']
        self.previous_outputs = outputs
        return dict(text=text, output_ids=outputs, finish_reason=reason,
            prompt_tokens=cached+len(ids), completion_tokens=len(outputs),
            cached_tokens=cached, prefill_tokens=incoming,
            budget_scope='resumed_history_after_each_turn',
            selection='upstream_exact_q_snapkv_base_burst_repair', **stats)

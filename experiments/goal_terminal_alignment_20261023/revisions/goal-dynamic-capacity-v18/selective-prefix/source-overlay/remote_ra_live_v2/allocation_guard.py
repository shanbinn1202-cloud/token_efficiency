"""Validate unsupported allocation semantics, not a normal capacity-check flag."""
import json
from .contract import LiveV2Error

def reject(field, value, reason):
    # Never stringify request objects, blocks, prompts or token contents.
    safe = value if value is None or type(value) in (bool, int, float) else '<redacted>'
    if field == 'request.status' and type(value) is str:
        safe = value[:64]
    if field in ('computed_tokens_by_group', 'external_computed_tokens_by_group') and type(value) in (tuple,list):
        safe = list(value) if len(value) <= 16 and all(type(v) is int for v in value) else '<redacted>'
    detail = dict(field=field, value=safe, value_type=type(value).__name__, reason=reason)
    from remote_ra_linux.native_authority import event
    event('NATIVE_ALLOCATION_REFUSED', **detail)
    raise LiveV2Error('native allocation refused: ' + json.dumps(detail, sort_keys=True))

def validate_native_allocation(values, request, binding, plan, start):
    # Stage B: phase 2 is admitted at plan['phase2_start'] (e, or 0 at local turn 0).
    for field in ('num_new_computed_tokens', 'num_external_computed_tokens',
                  'num_lookahead_tokens', 'num_encoder_tokens', 'reserved_blocks'):
        value = values.get(field, 0)
        if type(value) is not int or value != 0:
            reject(field, value, 'prefix/async/speculative/encoder reservation unsupported')
    for field in ('delay_cache_blocks', 'full_sequence_must_fit', 'has_scheduled_reqs', 'skip_cache'):
        value = values.get(field, False)
        if type(value) is not bool:
            reject(field, value, 'native boolean argument required')
    if values['delay_cache_blocks']:
        reject('delay_cache_blocks', values['delay_cache_blocks'], 'async KV load unsupported')
    computed = values.get('new_computed_blocks')
    if computed is not None:
        groups = getattr(computed, 'blocks', None)
        if groups is None or any(groups):
            reject('new_computed_blocks', computed, 'cannot borrow actual exact-prefix blocks')
    for field in ('num_preemptions',):
        value = getattr(request, field, 0)
        if type(value) is not int or value != 0:
            reject(field, value, 'preemption/asynchronous placeholder state unsupported')
    if getattr(request, 'spec_token_ids', ()):
        reject('spec_token_ids', getattr(request, 'spec_token_ids'), 'draft tokens unsupported')
    status = getattr(getattr(request, 'status', None), 'name', None)
    if status not in ('WAITING', 'RUNNING'):
        reject('request.status', status, 'only fresh WAITING admission or owned RUNNING continuation')
    placeholders = getattr(request, 'num_output_placeholders', 0)
    if type(placeholders) is not int or not 0 <= placeholders <= 2:
        reject('num_output_placeholders', placeholders, 'non-spec native async depth exceeded')
    logical, total, count = request.num_computed_tokens, request.num_tokens + placeholders, values['num_new_tokens']
    for field, value in (('request.num_computed_tokens', logical), ('request.num_tokens', total),
                         ('num_new_tokens', count)):
        if type(value) is not int:
            reject(field, value, 'native token count must be integer')
    if not (start <= logical < total and count > 0 and logical + count <= total):
        reject('num_new_tokens', count, 'unsupported partial/out-of-sequence interval')
    if status == 'WAITING':
        if logical != start or total != len(binding.tokens):
            reject('request.num_computed_tokens', logical, 'WAITING must start at the authenticated phase-2 start')
    if values['full_sequence_must_fit'] and status != 'WAITING':
        reject('full_sequence_must_fit', True, 'capacity precheck is only supported for WAITING admission')
    # True remains True: it asks the native allocator to check full input capacity.
    # It is neither a prefix hit nor permission to bypass sparse ownership.
    return True

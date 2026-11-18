"""Committed session carry: cross-request provenance and exact restore checks."""
from .contract import require

def active_record(backend, identity, generation):
    record=backend.requests.get(identity.request)
    require(record is not None and record['identity']==identity and record['generation']==generation,
            'carry request identity/allocation generation changed')
    return record

def verify_new_request(backend, identity, generation, namespace):
    """Return the committed carry this request continues, or None."""
    carry=backend.carry.get(identity.session)
    if carry is None:
        return None
    # A rebuild closes the session (and its carry) before a new namespace opens.
    require(carry.session==identity.session and carry.namespace==namespace,
            'committed carry belongs to another session namespace')
    prior,prior_generation=carry.owner,carry.owner_generation
    require((identity.session,identity.epoch,identity.source,identity.model)==
            (prior.session,prior.epoch,prior.source,prior.model) and
            identity.turn>prior.turn and identity.request!=prior.request and
            identity.generation!=prior.generation and generation>prior_generation,
            'illegal cross-request carry reuse')
    require(backend.io.payload_digest(carry.payload)==carry.digest, 'committed carry payload changed')
    return carry

def verify_carry(backend, record, computed_end, carry_digest):
    """The transaction baseline carry, unchanged, at exactly (C, digest)."""
    carry=record['carry']
    require(carry is not None and backend.carry.get(carry.session) is carry and
            carry.computed_end==computed_end and carry.digest==carry_digest and
            backend.io.payload_digest(carry.payload)==carry.digest,
            'committed carry absent/stale')
    return carry

"""Do not send cancellation to a known dead/revoked engine; retain uncertainty."""
class UnconfirmedRemoteCleanup(RuntimeError):
    pass

def require_cancel_peer(registry, record):
    client=registry.llm.engine_core
    authority=getattr(client,'_remote_ra_authority',None)
    dead=getattr(getattr(client,'resources',None),'engine_dead',False) is True
    revoked=getattr(authority,'revoked',None)
    revoked=revoked is not None and revoked.is_set()
    if not dead and not revoked:return
    rid=record['assigned'].internal_id
    failures=getattr(registry,'_unconfirmed_cancels',None)
    if failures is None:
        failures={};registry._unconfirmed_cancels=failures
    if rid not in failures:
        reason='engine_dead' if dead else 'authority_revoked_or_eof'
        from hashlib import sha256
        error=UnconfirmedRemoteCleanup('remote cancel skipped: '+reason+
            '; drain unconfirmed; ledger retained; request_sha256='+sha256(rid.encode()).hexdigest())
        failures[rid]=error
        from .diagnostics import failure
        failure('FRONTEND_CANCEL_UNCONFIRMED',error,authority)
    # Existing registry/controller cleanup retains the request and still performs
    # local abort/native shutdown. Never synthesize ACK or successful drain.
    raise failures[rid]

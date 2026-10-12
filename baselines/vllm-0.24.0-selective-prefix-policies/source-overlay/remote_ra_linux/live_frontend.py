"""Attach actual frontend owners before private F1 admission is released."""
from hashlib import sha256
import json

from dynamic_runtime import require
from entrypoint_runtime import ShutdownOwner
from lifecycle_binding import FrontendBinding
from frontend_registry import FrontendRegistry, TrustedFrontendCapture
from route_runtime import ContractBinding
from serving_controller import ServingController
from remote_ra_http.config import load
from .live_renderer import ActualRendererBinding


def bind_renderer(renderer, authority):
    """Bind the real HfRenderer owner, loaded identities and semantic tokenizer."""
    require(authority.verify_identity() is True, "loaded installation identity required")
    return ActualRendererBinding(renderer, authority)


async def bind(engine, state, args, authority):
    try:
        return await _bind(engine, state, args, authority)
    except BaseException as original:
        try:
            authority.revoke("frontend_bind_failed")
        except BaseException as cleanup:
            raise BaseExceptionGroup("frontend binding and revocation failed", [original, cleanup]) from None
        raise


async def _bind(engine, state, args, authority):
    require(getattr(engine, "_remote_ra_dynamic_mode", False) is False
            and state.engine_client is engine, "dynamic frontend engine owner differs")
    require(authority.verify_identity() is True and authority.verify_engine_handshake() is True,
            "private engine B0 handshake required before frontend binding")
    serving = state.openai_serving_chat
    renderer = engine.renderer
    require(serving is not None and serving.engine_client is engine
            and serving.renderer is renderer and serving.openai_serving_render.renderer is renderer,
            "native serving/render owners differ")
    require(authority.policy in ("full", "random50", "goal_query", "observation_query", "random_fill"), "four-policy startup required")
    require(not getattr(args, "trust_request_chat_template", False), "client templates cannot change trusted source")
    loaded = bind_renderer(renderer, authority)
    require(serving.openai_serving_render.chat_template == loaded.renderer.template,
            "start smoke with the pinned explicit chat template")
    def verify():
        require(authority.verify_identity() is True and loaded.verify() is True,
                "frontend loaded identity changed")
        require(state.engine_client is engine and state.openai_serving_chat is serving
                and engine.renderer is renderer and serving.renderer is renderer
                and serving.openai_serving_render.renderer is renderer,
                "frontend owner changed")
        return True
    from remote_ra_live_v2.mswea_protocol import PROTOCOL
    config = load()
    # Multi-session stage 1: one frontend XML transcript per startup task/session.
    transcripts = (tuple(loaded.renderer.transcript() for _ in config['tasks'])
                   if config['protocol'] == PROTOCOL else ())
    capture = TrustedFrontendCapture(renderer=renderer, source_renderer=loaded.renderer,
        verify_identity=verify, goal_probe_text=None, transcripts=transcripts)
    registry = FrontendRegistry(llm=engine, capture_factory=capture, codec=authority.codec,
        encoder=engine.engine_core.encoder, verify_identity=verify,
        verify_engine_handshake=authority.verify_engine_handshake)
    controller = ServingController(serving=serving, registry=registry, verify_identity=verify)
    binding = ContractBinding(serving, controller, authority.session_id, authority.policy,
                              authority.verify_admission)
    attachments = []
    def attach(owner, name, value):
        require(not hasattr(owner, name) or getattr(owner, name) is None,
                "refusing to replace frontend attachment: " + name)
        existed = hasattr(owner, name)
        attachments.append((owner, name, existed, getattr(owner, name, None)))
        setattr(owner, name, value)
    try:
        for owner in (engine, engine.engine_core):
            require(getattr(owner, "_remote_ra_dynamic_mode", False) is False,
                    "dynamic frontend/client already attached")
            existed = hasattr(owner, "_remote_ra_dynamic_mode")
            attachments.append((owner, "_remote_ra_dynamic_mode", existed,
                                getattr(owner, "_remote_ra_dynamic_mode", None)))
            owner._remote_ra_dynamic_mode = True
        attach(engine, "_remote_ra_frontend_registry", registry)
        attach(engine, "_remote_ra_registered_add", registry.bridge)
        for owner in (engine, serving, serving.openai_serving_render):
            attach(owner, "_remote_ra_serving_controller", controller)
        attach(state, "_remote_ra_chat_route_binding", binding)
        from remote_ra_live_v2.composition import compose_frontend
        from remote_ra_live_v2.readiness import FrontendReadiness
        live = compose_frontend(serving, controller, policy=authority.policy,
            instance_id=authority.session_id,
            readiness=FrontendReadiness(authority.live_v2_readiness),
            tasks=config['tasks'], transcripts=transcripts or None,
            max_num_seqs=config['effective_server_config']['max_num_seqs'])
        attach(engine, "_remote_ra_live_four_policy", live)
        attach(serving, "_remote_ra_live_four_policy", live)
        shutdown = ShutdownOwner(engine, controller, verify)
        attach(engine, "_remote_ra_shutdown_owner", shutdown)
        lifecycle = FrontendBinding(engine, state, registry, controller, shutdown)
        attach(state, "_remote_ra_frontend_binding", lifecycle)
        attach(engine, "_remote_ra_loaded_renderer", loaded)
        lifecycle.check()
        receipt = dict(manifest=authority.identity.manifest, epoch=authority.identity.epoch,
                       session_id=authority.session_id, policy=authority.policy,
                       tokenizer_semantics=loaded.renderer._semantic.__class__.__name__,
                       template_sha256=sha256(loaded.renderer.template.encode()).hexdigest())
        authority.bind_frontend(sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest())
        require(authority.verify_admission() is True, "private F1 acknowledgement missing")
        from remote_ra_http.evidence import FrontendEvidence
        from remote_ra_http.config import effective_server_config
        state._remote_ra_http_evidence = FrontendEvidence(authority, state, effective_server_config(args))
        return binding
    except BaseException:
        controller._closing = True
        registry._closed = True
        # No HTTP admission precedes F1. The enclosing factory revokes authority
        # after restoring the exact pre-bind attachments.
        for owner, name, existed, old in reversed(attachments):
            if existed:
                setattr(owner, name, old)
            else:
                delattr(owner, name)
        raise

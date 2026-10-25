# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.orca_metrics import metrics_header
from vllm.logger import init_logger

from route_runtime import (route_cancellation, route_load_aware, invoke_chat,
                           reject_dynamic_batch, dynamic, require_completion,
                           invoke_session_close, session_close_status, chat_refusal)

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def batch_chat(request: Request) -> OpenAIServingChatBatch | None:
    return request.app.state.openai_serving_chat_batch


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@route_cancellation(with_cancellation)
@route_load_aware(load_aware_call)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    try:
        generator = await invoke_chat(handler, request, raw_request)
    except Exception as error:
        # chat-text protocol: a refused request answers with its own status and body.
        refusal = chat_refusal(error)
        if refusal is None:
            raise
        return JSONResponse(content=refusal[1], status_code=refusal[0])

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        require_completion(raw_request)
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    if dynamic(raw_request):
        raise RuntimeError("dynamic streaming response unsupported")
    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/chat/completions/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_batch_chat_completion(
    request: BatchChatCompletionRequest, raw_request: Request
):
    reject_dynamic_batch(raw_request)
    handler = batch_chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    result = await handler.create_batch_chat_completion(request, raw_request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result.model_dump())


@router.post("/v1/sessions/close", dependencies=[Depends(validate_json_request)])
async def close_session(raw_request: Request):
    """Session pool (serve.py --max-running-sessions): close one startup task's
    session when its agent is done. Loopback only; JSON body {"instance_id"}."""
    try:
        receipt = await invoke_session_close(raw_request)
    except Exception as error:
        status = session_close_status(error)
        return JSONResponse(content=dict(error=dict(message=str(error)[:4096],
                                                    type=type(error).__name__, code=status)),
                            status_code=status)
    return JSONResponse(content=dict(receipt, object="session.close"))


# The kv-session protocol (serve.py --protocol kv-session; remote_ra_http.kv_session_route).
@router.get("/v1/kv-session/info")
async def kv_session_info(raw_request: Request):
    """The kv-session server's method, budget and options."""
    from remote_ra_http.kv_session_route import invoke_info
    status, content = await invoke_info(raw_request)
    return JSONResponse(content=content, status_code=status)


@router.post("/v1/kv-session/complete", dependencies=[Depends(validate_json_request)])
async def kv_session_complete(raw_request: Request):
    """One token-level turn of a kv-session session (only the new tail is sent)."""
    from remote_ra_http.kv_session_route import invoke_complete
    status, content = await invoke_complete(raw_request, ChatCompletionRequest)
    return JSONResponse(content=content, status_code=status)


@router.post("/v1/kv-session/close", dependencies=[Depends(validate_json_request)])
async def kv_session_close(raw_request: Request):
    """Close a kv-session session; 200 also for unknown and closed ids."""
    from remote_ra_http.kv_session_route import invoke_close
    status, content = await invoke_close(raw_request)
    return JSONResponse(content=content, status_code=status)


# The chat-text protocol (serve.py --protocol chat-text; remote_ra_http.chat_text_route).
@router.get("/v1/chat-text/info")
async def chat_text_info(raw_request: Request):
    """The chat-text server's options and its client-session counts."""
    from remote_ra_http.chat_text_route import invoke_info
    status, content = await invoke_info(raw_request)
    return JSONResponse(content=content, status_code=status)


@router.post("/v1/chat-text/close", dependencies=[Depends(validate_json_request)])
async def chat_text_close(raw_request: Request):
    """Close a chat-text client session by its X-Session-ID (or its key_sha256)."""
    from remote_ra_http.chat_text_route import invoke_close
    status, content = await invoke_close(raw_request)
    return JSONResponse(content=content, status_code=status)


def attach_router(app: FastAPI):
    app.include_router(router)

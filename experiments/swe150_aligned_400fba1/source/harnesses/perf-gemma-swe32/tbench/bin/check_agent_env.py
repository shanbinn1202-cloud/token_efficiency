"""Read-only check of what the agents do on a slow or failed request (perf-gemma-swe32; run by check_agent_env.sh under the
agent env's python). Prints one JSON line: mini-swe-agent and litellm versions; mini's model retry (the environment
variable MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT and its default as the installed source names them, the backoff, the
exceptions that abort the retry); the class chain of litellm_textbased; litellm's request_timeout default and
REQUEST_TIMEOUT, its DEFAULT_MAX_RETRIES (the OpenAI client's own retries), and whether litellm.completion accepts the
timeout / max_retries / num_retries keywords the gswe32 model override sets. Design R9: a client retry of a request still
in flight is a second request for a busy session on our server, so gswe32 runs with no retry at any level.
"""
import inspect
import json
import os
import re


def source_of(module):
    try:
        return inspect.getsource(module)
    except Exception:
        return ''


def main():
    value = {}
    from importlib.metadata import version
    for name in ('mini-swe-agent', 'litellm', 'openai', 'datasets'):
        try:
            value[name] = version(name)
        except Exception:
            value[name] = None
    try:
        from minisweagent.models.utils import retry as retry_module
        text = source_of(retry_module)
        value['mini_retry_env'] = sorted(set(re.findall(r'MSWEA_[A-Z_]+', text)))
        found = re.search(r'MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT["\']\s*,\s*["\']?([0-9]+)', text)
        value['mini_retry_default_attempts'] = int(found.group(1)) if found else None
        value['mini_retry_backoff'] = re.findall(r'wait_exponential\([^)]*\)', text)[:2]
    except Exception as exc:
        value['mini_retry_error'] = '%s: %s' % (type(exc).__name__, exc)
    try:
        from minisweagent.models import litellm_textbased_model as textbased
        cls = getattr(textbased, 'LitellmTextbasedModel', None)
        value['textbased_mro'] = [c.__module__ + '.' + c.__name__ for c in inspect.getmro(cls)] if cls else None
        from minisweagent.models import litellm_model
        text = source_of(litellm_model)
        value['abort_exceptions'] = sorted(set(re.findall(r'litellm\.exceptions\.([A-Za-z]+)', text)))
        value['litellm_model_sets_timeout'] = 'timeout' in re.findall(r'(\w+)\s*=', text)
    except Exception as exc:
        value['mini_model_error'] = '%s: %s' % (type(exc).__name__, exc)
    try:
        import litellm
        value['litellm_request_timeout'] = getattr(litellm, 'request_timeout', None)
        value['env_REQUEST_TIMEOUT'] = os.environ.get('REQUEST_TIMEOUT')
        try:
            from litellm import constants
            value['litellm_DEFAULT_MAX_RETRIES'] = getattr(constants, 'DEFAULT_MAX_RETRIES', None)
            value['litellm_DEFAULT_REQUEST_TIMEOUT_SECONDS'] = getattr(constants, 'DEFAULT_REQUEST_TIMEOUT_SECONDS', None)
        except Exception:
            pass
        params = inspect.signature(litellm.completion).parameters
        value['completion_keywords'] = {k: k in params for k in ('timeout', 'max_retries', 'num_retries')}
        value['litellm_num_retries_global'] = getattr(litellm, 'num_retries', None)
    except Exception as exc:
        value['litellm_error'] = '%s: %s' % (type(exc).__name__, exc)
    value['event'] = 'gswe32_agent_env'
    print(json.dumps(value, sort_keys=True, default=str))


if __name__ == '__main__':
    main()

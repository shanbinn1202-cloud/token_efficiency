"""Read-only report of what Terminus 2 does with the tb16 settings (perf-tb16; run by check_agent_env.sh under the
Harbor env's python, -I -B: nothing is written). One JSON line: Harbor / litellm / openai versions; Terminus 2's
constructor defaults (parser, summarization and its threshold, terminal recording, session id), the retry layers
(LiteLLM.call's and Terminus' _query_llm's tenacity stop / wait), litellm's default request timeout and whether
litellm.acompletion takes the timeout / max_retries / num_retries keywords tb16's llm_kwargs set; whether the
hosted_vllm model-info rule accepts tb16's model_info. Send it back with the stress-test summary.
"""
import inspect
import json
import re
import sys


def source(obj):
    try:
        return inspect.getsource(obj)
    except Exception:
        return ''


def main(model_info):
    value = {}
    from importlib.metadata import version
    for name in ('harbor', 'litellm', 'openai', 'tenacity'):
        try:
            value[name] = version(name)
        except Exception:
            value[name] = None
    try:
        from harbor.agents.terminus_2.terminus_2 import Terminus2
        params = inspect.signature(Terminus2.__init__).parameters
        value['terminus_defaults'] = {k: (params[k].default if params[k].default is not inspect.Parameter.empty else None)
                                      for k in ('parser_name', 'temperature', 'session_id', 'enable_summarize',
                                                'proactive_summarization_threshold', 'record_terminal_session',
                                                'interleaved_thinking', 'llm_kwargs', 'llm_call_kwargs') if k in params}
        text = source(Terminus2)
        value['query_llm_retry'] = re.findall(r'stop_after_attempt\((\d+)\)', text)[:4]
        value['terminus_version'] = Terminus2.version(None) if hasattr(Terminus2, 'version') else None
    except Exception as exc:
        value['terminus_error'] = '%s: %s' % (type(exc).__name__, exc)
    try:
        from harbor.llms import lite_llm
        text = source(lite_llm.LiteLLM)
        value['litellm_call_retry'] = dict(stop=re.findall(r'stop_after_attempt\((\d+)\)', text)[:2],
                                           wait=re.findall(r'wait_exponential\(([^)]*)\)', text)[:2])
        value['sends_session_header_only_with_session_id'] = 'X-Session-ID' in text and 'self._session_id is not None' in text
        from harbor.llms.utils import validate_hosted_vllm_model_config
        try:
            validate_hosted_vllm_model_config('hosted_vllm/selective-prefix', model_info)
            value['model_info_accepted'] = True
        except Exception as exc:
            value['model_info_accepted'] = '%s: %s' % (type(exc).__name__, exc)
    except Exception as exc:
        value['lite_llm_error'] = '%s: %s' % (type(exc).__name__, exc)
    try:
        import litellm
        value['litellm_request_timeout'] = getattr(litellm, 'request_timeout', None)
        value['litellm_num_retries_global'] = getattr(litellm, 'num_retries', None)
        params = inspect.signature(litellm.acompletion).parameters
        value['acompletion_keywords'] = {k: k in params for k in ('timeout', 'max_retries', 'num_retries',
                                                                  'max_tokens', 'temperature', 'extra_body')}
    except Exception as exc:
        value['litellm_error'] = '%s: %s' % (type(exc).__name__, exc)
    value['event'] = 'tb16_agent_env'
    print(json.dumps(value, sort_keys=True, default=str))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else
         {'max_input_tokens': 65536, 'max_output_tokens': 4096, 'input_cost_per_token': 0, 'output_cost_per_token': 0})

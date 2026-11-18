"""Mini-SWE-Agent model class of the stock keep-reasoning plans (perf-swe32, 2026-11-16).

Mini-SWE-Agent 2.4.2 through litellm 1.103.1 sends each earlier reply's reasoning back only as `reasoning_content`
(the request body captured on tbench: an assistant message carries content, reasoning_content,
provider_specific_fields and role). Stock vLLM 0.24.0 reads an assistant message's earlier reasoning only from its
`reasoning` field (vllm/entrypoints/chat_utils.py _parse_chat_message_content: reasoning = message.get("reasoning")),
so without this class the template never sees it. This class is the text-based model class
(minisweagent.models.litellm_textbased_model.LitellmTextbasedModel) unchanged, except that the messages it hands to
litellm carry `reasoning` as well: a copy of reasoning_content, on every assistant message whose reasoning_content is a
string. The trajectory's stored messages are not changed (the copy is made on the prepared request only), and nothing
else in the request differs. Neither Mini-SWE-Agent nor litellm is modified.

Installed next to the pinned runner (tbench /home/ubuntu/swe32/upstream/, the directory swebench_runner.py puts first
on sys.path for its optional adapters) by tbench/agent_ext/deploy_agent_ext.ps1; the keep plans' model override names
model_class swe32_keep_reasoning.KeepReasoningTextbasedModel. Importable without Mini-SWE-Agent (tests): then only
with_reasoning() exists.
"""


def with_reasoning(messages):
    """The request messages with `reasoning` = reasoning_content added to each assistant message whose
    reasoning_content is a string (new dicts; the input is not changed)."""
    out = []
    for message in messages:
        if message.get('role') == 'assistant' and isinstance(message.get('reasoning_content'), str):
            message = dict(message, reasoning=message['reasoning_content'])
        out.append(message)
    return out


try:
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
except ImportError:  # tests without Mini-SWE-Agent
    LitellmTextbasedModel = None

if LitellmTextbasedModel is not None:
    class KeepReasoningTextbasedModel(LitellmTextbasedModel):
        """LitellmTextbasedModel whose requests also carry each earlier reply's reasoning as `reasoning`."""

        def _prepare_messages_for_api(self, messages):
            return with_reasoning(super()._prepare_messages_for_api(messages))

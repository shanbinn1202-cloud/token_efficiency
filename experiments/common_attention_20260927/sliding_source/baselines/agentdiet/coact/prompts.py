from jinja2 import Template, StrictUndefined

AGENTDIET_REDUCTION_PROMPT = """You will analyze and compress a given step in a trajectory of an AI agent solving a software bug.

In the trajectory, each step is marked in <step id="..."></step>.
The agent will think in <think>, call external tools as marked in <call tool="..."></call>. Its result is marked in <result></result> within the <step> tag.

Your job is to compress the text within the TARGET step (id="{{ target_step_id }}") to avoid harming efficiency, typically shortening it to 20%-50% of the original length.
Meanwhile, keep the compressed text useful such that you are able to continue the trajectory as close as the original path.

- You should ONLY remove redundant texts, which are either irrelevant to future steps or duplicated by other texts in the trajectory.
- Replace the text to remove to "..." and a short takeaway, e.g. "... (same as the content below)".
- You should keep the original structure unchanged, e.g., XML tags, Python indentation, JSON formats and line numbers.
- Again, keep useful details in the original content unchanged, e.g., XML tags, Python indentation and line numbers.

Typical examples:
- If the step opens a huge file but only one part is necessary for future steps, replace other parts to "... (unrelated function XXX, YYY)".
- If the step runs a verbose test script and everything goes fine, replace the verbose part to "... (expected output)".
- If the step uses str_replace_editor to modify a file and the content can be inferred by the content after it, replace the tool call argument to "... (see results below)".

Task goal: {{ goal }}

Trajectory window:
{% for step_id, think_text, call_text, result_text in context_steps -%}
<step id="{{ step_id }}"{% if step_id == target_step_id %} [TARGET]{% endif %}>
<think>
{{ think_text }}
</think>
{{ call_text }}
<result>
{{ result_text }}
</result>
</step>
{% if not loop.last %}
{% endif -%}
{% endfor %}
You should only process the text within the <step> tag with the given id.
Please output the completely rewritten <step> block for the TARGET step. STOP OUTPUT IMMEDIATELY AFTER </step>."""

_AGENTDIET_REDUCTION_PROMPT_TEMPLATE = Template(
    AGENTDIET_REDUCTION_PROMPT,
    undefined=StrictUndefined,
)

def render_agentdiet_reduction_prompt(
    goal: str,
    context_steps: list[tuple[int, str, str, str]],
    target_step_id: int,
) -> str:
    """Render the AgentDiet reflection prompt for one reduction attempt.

    Args:
        goal: The global task goal.
        context_steps: List of ``(step_id, think_text, call_text, result_text)`` tuples
            forming the trajectory window shown to the reflector.
        target_step_id: Step id within ``context_steps`` to mark as TARGET and rewrite.

    Returns:
        The rendered reflection prompt.
    """
    return _AGENTDIET_REDUCTION_PROMPT_TEMPLATE.render(
        goal=goal,
        context_steps=context_steps,
        target_step_id=target_step_id,
    )

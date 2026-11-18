"""Transport semantic observation spans without changing model-visible text."""
import copy
import uuid


def render_observations(messages, indices, generation, render):
    original = render(messages, generation)
    marked = copy.deepcopy(messages)
    replacements = {}
    for i in sorted(indices):
        if messages[i]['role'] != 'user' or not isinstance(messages[i]['content'], str):
            raise ValueError('Expected a text Mini-SWE observation')
        marker = 'AGENTKV_OBSERVATION_' + uuid.uuid4().hex
        replacements[marker] = messages[i]['content']
        marked[i]['content'] = marker
    text = render(marked, generation)
    spans = []
    for marker in sorted(replacements, key=lambda m: text.index(m)):
        if text.count(marker) != 1:
            raise ValueError('Template lost or duplicated observation content')
        start = text.index(marker)
        content = replacements[marker]
        text = text[:start] + content + text[start + len(marker):]
        spans.append([start, start + len(content)])
    if text != original:
        raise ValueError('Observation phase annotation changed model input')
    return original, spans


def apply_observation_spans(fresh, offsets, phases, spans):
    """Override only observation content; quoted phase markers stay tool text."""
    if not isinstance(spans, list):
        raise ValueError('Observation spans must be a list')
    previous = 0
    for pair in spans:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('Malformed observation span')
        start, end = pair
        if type(start) is not int or type(end) is not int or not previous <= start <= end <= len(fresh):
            raise ValueError('Invalid observation bounds')
        previous = end
    result = list(phases)
    if len(offsets) != len(result):
        raise ValueError('Phase/token length mismatch')
    for i, (start, end) in enumerate(offsets):
        # Use the original parser's majority-character convention. On a tie,
        # preserve its structural/header classification.
        overlap = sum(max(0, min(end, b) - max(start, a)) for a, b in spans)
        if overlap > (end - start) / 2:
            result[i] = 3
    return result

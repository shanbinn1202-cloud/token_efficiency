"""Fill examples/swebench/model_override.yaml for one run_agents.sh run; writes stdout.

Usage: render_override.py <template> <port> [<max_tokens> <step_limit>]

With the defaults (max_tokens 1536, step_limit 90) the output is the template with
the port filled in, byte for byte the file the pilots used. Other values replace
only the line ``    max_tokens: 1536`` and the line ``  step_limit: 90``:
max_tokens must equal the server's --max-tokens (the server forces its own bound
anyway) and step_limit stays within the server's per-session bound of 256 requests.
"""
import re
import sys

MAX_TOKENS = ('1536', '4096')                 # serve.py --max-tokens choices
MAX_STEPS = 256                               # remote_ra_live_v2.sessions.MAX_TURNS
LINES = {'max_tokens': '    max_tokens: %s', 'step_limit': '  step_limit: %s'}
DEFAULTS = {'max_tokens': '1536', 'step_limit': '90'}


def render(template, port, max_tokens='1536', step_limit='90'):
    if not re.fullmatch(r'[0-9]{4,5}', port):
        raise ValueError('port must be 4 or 5 digits')
    if max_tokens not in MAX_TOKENS:
        raise ValueError('MAX_TOKENS must be 1536 or 4096 (the server --max-tokens)')
    if not (re.fullmatch(r'[1-9][0-9]{0,2}', step_limit) and int(step_limit) <= MAX_STEPS):
        raise ValueError('STEP_LIMIT must be an integer from 1 to %d' % MAX_STEPS)
    lines = template.split('\n')
    values = {'max_tokens': max_tokens, 'step_limit': step_limit}
    for key, form in LINES.items():
        default = form % DEFAULTS[key]
        if lines.count(default) != 1:
            raise ValueError('template must hold the line %r exactly once' % default)
        lines[lines.index(default)] = form % values[key]
    text = '\n'.join(lines)
    if text.count('<PORT>') != 1:
        raise ValueError('template must hold <PORT> exactly once')
    return text.replace('<PORT>', port)


def main(argv):
    if len(argv) not in (3, 5):
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    with open(argv[1], 'rb') as stream:
        template = stream.read().decode('utf-8')
    try:
        text = render(template, *argv[2:])
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    sys.stdout.buffer.write(text.encode('utf-8'))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

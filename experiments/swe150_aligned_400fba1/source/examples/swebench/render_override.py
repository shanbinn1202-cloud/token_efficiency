"""Fill examples/swebench/model_override.yaml for one run_agents.sh run; writes stdout.

Usage: render_override.py <template> <port> [<max_tokens> <step_limit> [<stock> [<repo_env>]]]

With the defaults (max_tokens 1536, step_limit 90) the output is the template with
the port filled in, byte for byte the file the pilots used. Other values replace
only the line ``    max_tokens: 1536`` and the line ``  step_limit: 90``:
max_tokens must equal the server's --max-tokens (the server forces its own bound
anyway) and step_limit stays within the server's per-session bound of 256 requests.
stock 1 (default 0) is for a stock vLLM control: it adds, after the max_tokens line, the two
model_kwargs lines of the client-side generation contract that the stock control ran with
(STOCK_LINES: the stop marker kept in the output, thinking off); with 0 the output is unchanged.
repo_env testbed (absent by default) runs the agent's commands with the repository's testbed
environment active: it replaces only the line ``  interpreter: ["bash", "-c"]`` with
``  interpreter: ["bash", "-lc"]`` (the image's login shell activates `testbed`;
docs/known-limits.md 5.6).
"""
import re
import sys

MAX_TOKENS = ('1536', '4096')                 # serve.py --max-tokens choices
MAX_STEPS = 256                               # remote_ra_live_v2.sessions.MAX_TURNS
LINES = {'max_tokens': '    max_tokens: %s', 'step_limit': '  step_limit: %s'}
DEFAULTS = {'max_tokens': '1536', 'step_limit': '90'}
# Stock vLLM takes sampling from the request: the stop marker (kept in the output) and thinking off.
STOCK_LINES = ('    extra_body: {"chat_template_kwargs": {"enable_thinking": false}, '
               '"include_stop_str_in_output": true}\n'
               '    stop: ["</mswea_bash_command>"]')
# REPO_ENV=testbed: the agent's commands in a login shell, which activates the image's `testbed` environment.
INTERPRETER = {'': '  interpreter: ["bash", "-c"]', 'testbed': '  interpreter: ["bash", "-lc"]'}


def render(template, port, max_tokens='1536', step_limit='90', stock='0', repo_env=''):
    if not re.fullmatch(r'[0-9]{4,5}', port):
        raise ValueError('port must be 4 or 5 digits')
    if max_tokens not in MAX_TOKENS:
        raise ValueError('MAX_TOKENS must be 1536 or 4096 (the server --max-tokens)')
    if not (re.fullmatch(r'[1-9][0-9]{0,2}', step_limit) and int(step_limit) <= MAX_STEPS):
        raise ValueError('STEP_LIMIT must be an integer from 1 to %d' % MAX_STEPS)
    if stock not in ('0', '1'):
        raise ValueError('STOCK must be 0 or 1')
    if repo_env not in INTERPRETER:
        raise ValueError('REPO_ENV must be testbed or unset')
    lines = template.split('\n')
    values = {'max_tokens': max_tokens, 'step_limit': step_limit}
    for key, form in LINES.items():
        default = form % DEFAULTS[key]
        if lines.count(default) != 1:
            raise ValueError('template must hold the line %r exactly once' % default)
        lines[lines.index(default)] = form % values[key]
        if key == 'max_tokens' and stock == '1':
            lines.insert(lines.index(form % values[key]) + 1, STOCK_LINES)
    if repo_env:
        if lines.count(INTERPRETER['']) != 1:
            raise ValueError('template must hold the line %r exactly once' % INTERPRETER[''])
        lines[lines.index(INTERPRETER[''])] = INTERPRETER[repo_env]
    text = '\n'.join(lines)
    if text.count('<PORT>') != 1:
        raise ValueError('template must hold <PORT> exactly once')
    return text.replace('<PORT>', port)


def main(argv):
    if len(argv) not in (3, 5, 6, 7):
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

"""Scripted client of the keep-rebuild edge-case smoke (perf-swe32 plan `keepedge`, 2026-11-16).

No model-driven agent: one session per task is driven by this script, which sends exactly the request sequence below
to the server and keeps or drops each reply by the script, never by its text; the observations are the script's own
text. The script covers, in every session, the request shapes of keep-rebuild-1116 (runtime 51af079):

  0  fresh                     the reply is dropped (a session whose first reply is dropped)
  1  rollback                  (kept_tokens = the system + task render); the reply is kept
  2  append                    kept
  3  append                    dropped (a rollback after a reply that closed its reasoning, as a rule)
  4  rollback                  dropped again (two rollbacks in a row)
  5  rollback                  kept
  6+ append                    a prompt that asks for long reasoning: a reply cut inside its reasoning at the reply
                               bound (no content) is dropped (a rollback after an unclosed reasoning cut), a closed one
                               is kept and the prompt sent again, at most CUT_TRIES times (`cut` recorded either way)
  .  append, then the same request again (an ordinary resend: the first reply discarded; history `rebuild`)
  .  append                    kept; the session ends

A dropped reply is followed by one user message (Mini-SWE-Agent's format error text), exactly as Mini-SWE-Agent 2.4.2
does after a reply without exactly one action. Every request carries the earlier replies as the server returned them
(content, and the reasoning as `reasoning_content`, as litellm sends it), temperature 0, the server's reply bound and
the thinking-on template switch. Each call records its request messages, the response id and usage, the reply's
content / reasoning / finish reason, its timestamp and the history label the server must write (`expected_history`).
A capacity wait during a rollback is the server's (the row's capacity_waits); nothing in the client retries.

Mini-SWE-Agent integration (the plan's model override names model_class swe32_edge_script.EdgeScriptModel): the agent
runs as for any instance (its container starts), its first query runs the whole session script through the model's
api_base and returns one assistant message whose action submits (echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT: an empty
patch) and whose extra.swe32_script holds the script's record (calls, coverage, error). The join reads that record
(postrun/join_swe32.py, scripted servers). Installed next to the pinned runner on tbench (outside the half's manifest)
by tbench/agent_ext/deploy_agent_ext.ps1. Importable without Mini-SWE-Agent (tests): then only run_session() exists.
"""
import json
import time
import urllib.error
import urllib.request

SCRIPT_VERSION = 'keepedge-v1'
CUT_TRIES = 3
FORMAT_ERROR = 'Please always provide EXACTLY ONE action in triple backticks, found 0 actions.'
SUBMIT = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
LONG_REASONING = ('Before you act, reason step by step about every one of the numbered items below: for each item, '
                  'restate it, list three ways it could fail, and say how you would check each one. Only then give '
                  'one command.\n')


def observation(step, lines):
    """The script's observation text after a kept reply (no outer whitespace; `lines` numbered lines)."""
    body = ''.join('%5d\tscripted line %d of step %d: value = %d\n' % (i, i, step, (i * 7919 + step) % 10007)
                   for i in range(lines))
    return '<returncode>0</returncode>\n<output>\n%s</output>' % body


def long_prompt(step, items):
    return LONG_REASONING + ''.join('%d. scripted item %d of step %d\n' % (i + 1, i, step) for i in range(items))


def http_post(url, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'},
                                     method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def run_session(post, base, settings):
    """One scripted session. `post(body) -> response dict`; `base`: [system, task] messages (dicts with role, content);
    `settings`: model, max_tokens, template_kwargs (the request's chat_template_kwargs or None), observation_lines,
    long_items. Returns the record: script, calls, coverage, error (None, or the exception that ended the session)."""
    messages = [dict(role=m['role'], content=m['content']) for m in base]
    calls, coverage = [], dict(first_reply_dropped=0, rollback_after_closed=0, double_rollback=0,
                               rollback_after_cut=0, cut_attempts=0, resend=0)
    record = dict(script=SCRIPT_VERSION, settings=dict(settings), calls=calls, coverage=coverage, error=None)
    state = dict(next_history='fresh', drops_in_a_row=0)

    def send(action, history):
        body = dict(model=settings['model'], messages=[dict(m) for m in messages], temperature=0.0,
                    max_tokens=settings['max_tokens'])
        if settings.get('template_kwargs') is not None:
            body['chat_template_kwargs'] = settings['template_kwargs']
        value = post(body)
        choice = value['choices'][0]
        message = choice.get('message') or {}
        reasoning = message.get('reasoning_content', message.get('reasoning'))
        reply = dict(content=message.get('content') or '', reasoning=reasoning, finish_reason=choice.get('finish_reason'))
        calls.append(dict(index=len(calls), action=action, expected_history=history, request=body['messages'],
                          response=dict(id=value.get('id'), usage=value.get('usage'), created=value.get('created')),
                          reply=reply, timestamp=time.time()))
        return reply

    def keep(reply, text):
        assistant = dict(role='assistant', content=reply['content'])
        if isinstance(reply['reasoning'], str):
            assistant['reasoning_content'] = reply['reasoning']
        messages.extend([assistant, dict(role='user', content=text)])
        state.update(next_history='append', drops_in_a_row=0)

    def drop():
        messages.append(dict(role='user', content=FORMAT_ERROR))
        state['drops_in_a_row'] += 1
        state['next_history'] = 'rollback'

    def unclosed(reply):  # cut inside its reasoning: no content, and the reply ran to the bound
        last = calls[-1]['response'].get('usage') or {}
        return not reply['content'].strip() and (reply['finish_reason'] == 'length'
                                                 or last.get('completion_tokens') == settings['max_tokens'])

    try:
        reply = send('drop', state['next_history'])  # 0: fresh, the first reply dropped
        drop()
        coverage['first_reply_dropped'] += 1
        reply = send('keep', state['next_history'])  # 1: rollback (kept = the system + task render)
        keep(reply, observation(1, settings['observation_lines']))
        reply = send('keep', state['next_history'])  # 2: append
        keep(reply, observation(2, settings['observation_lines']))
        reply = send('drop', state['next_history'])  # 3: append, dropped
        coverage['rollback_after_closed'] += not unclosed(reply)
        drop()
        reply = send('drop', state['next_history'])  # 4: rollback, dropped again
        drop()
        coverage['double_rollback'] += 1
        reply = send('keep', state['next_history'])  # 5: rollback, kept
        keep(reply, long_prompt(5, settings['long_items']))
        for attempt in range(CUT_TRIES):  # 6+: a long-reasoning prompt until a reply is cut inside its reasoning
            coverage['cut_attempts'] += 1
            reply = send('cut', state['next_history'])
            if unclosed(reply):
                drop()
                reply = send('keep', state['next_history'])  # the rollback after the unclosed cut
                coverage['rollback_after_cut'] += 1
                keep(reply, observation(20 + attempt, settings['observation_lines']))
                break
            keep(reply, long_prompt(10 + attempt, settings['long_items']))
        send('resend-first', state['next_history'])  # an ordinary resend: the first reply discarded ...
        reply = send('keep', 'rebuild')  # ... and the same request again: a rebuild
        coverage['resend'] += 1
        keep(reply, observation(30, settings['observation_lines']))
        send('final', state['next_history'])  # the last request (its reply ends the session)
    except Exception as exc:  # recorded, never retried: the join fails a scripted session with an error
        record['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:500])
    record['messages'] = messages
    return record


def settings_from(model_name, model_kwargs, observation_lines=200, long_items=60):
    """The script's settings from Mini-SWE-Agent's model config (the plan's model override)."""
    extra = dict(model_kwargs.get('extra_body') or {})
    return dict(model=model_name.split('/', 1)[1] if model_name.startswith('openai/') else model_name,
                max_tokens=int(model_kwargs['max_tokens']), template_kwargs=extra.get('chat_template_kwargs'),
                timeout=float(model_kwargs.get('timeout') or 600), observation_lines=observation_lines,
                long_items=long_items, api_base=str(model_kwargs['api_base']))


def handover(record):
    """The one assistant message the agent receives: submit, the script's record in extra.swe32_script."""
    return dict(role='assistant', content='THOUGHT: the scripted session is done.\n\n```mswea_bash_command\n%s\n```'
                % SUBMIT, extra=dict(actions=[dict(command=SUBMIT)], response=dict(id=None, usage=None),
                                     cost=0.0, timestamp=time.time(), swe32_script=record))


try:
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
except ImportError:  # tests without Mini-SWE-Agent
    LitellmTextbasedModel = None

if LitellmTextbasedModel is not None:
    class EdgeScriptModel(LitellmTextbasedModel):
        """The scripted session in place of the model: the first query runs run_session() and submits."""

        def query(self, messages, **kwargs):
            settings = settings_from(self.config.model_name, dict(self.config.model_kwargs))
            url = settings['api_base'].rstrip('/') + '/chat/completions'
            record = run_session(lambda body: http_post(url, body, settings['timeout']),
                                 [m for m in messages if m.get('role') in ('system', 'user')][:2], settings)
            return handover(record)

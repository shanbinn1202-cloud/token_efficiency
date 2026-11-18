"""Mini-SWE adapter to an unmodified, pinned Claude Code /compact executor.

The main agent stays Mini-SWE. This module serializes its text conversation into
a private native transcript, invokes only /compact, and reads the official
continuation summary verbatim. It never writes a summarization prompt.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

ROOT=Path(__file__).resolve().parent
VERSION='2.1.325'
BINARY=ROOT/'vendor'/('claude-code-'+VERSION)/'package/claude'


def native_compact(messages, *, endpoint, output, model='selective-prefix', timeout=600):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cwd=output/'workspace';cwd.mkdir()
    config=output/'config';project=config/'projects'/re.sub(r'[^a-zA-Z0-9]','-',str(cwd))
    project.mkdir(parents=True)
    sid=str(uuid.uuid4());transcript=project/(sid+'.jsonl')
    rows=[];parent=None
    now=datetime.now(timezone.utc).isoformat()
    for m in messages:
        if m['role']=='system':continue
        if m['role'] not in {'user','assistant'}:raise ValueError('Expected Mini-SWE text conversation')
        identity=str(uuid.uuid4());content=m.get('content') or ''
        if m['role']=='assistant':
            blocks=[]
            if reasoning:=m.get('reasoning_content') or m.get('reasoning'):
                blocks.append({'type':'thinking','thinking':reasoning,'signature':''})
            blocks.append({'type':'text','text':content})
            message={'role':'assistant','type':'message','id':'msg_'+uuid.uuid4().hex,
                     'model':model,'content':blocks,'stop_reason':'end_turn','stop_sequence':None,
                     'usage':{'input_tokens':0,'output_tokens':0,'cache_read_input_tokens':0,'cache_creation_input_tokens':0}}
        else:message={'role':'user','content':content}
        rows.append({'type':m['role'],'message':message,'parentUuid':parent,'uuid':identity,
                     'isSidechain':False,'timestamp':now,'userType':'external','entrypoint':'sdk-cli',
                     'cwd':str(cwd),'sessionId':sid,'version':VERSION,'gitBranch':'HEAD'})
        parent=identity
    transcript.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'input-messages.json').write_text(json.dumps(messages,indent=2))
    # Credentials/configuration from the host are intentionally not inherited.
    env={k:os.environ[k] for k in ['HOME','PATH','USER','LOGNAME','LANG','LD_LIBRARY_PATH','TMPDIR'] if k in os.environ}
    env.update(CLAUDE_CONFIG_DIR=str(config),ANTHROPIC_BASE_URL=endpoint.removesuffix('/v1'),
               ANTHROPIC_API_KEY='local-vllm-no-auth',DISABLE_AUTOUPDATER='1',DISABLE_TELEMETRY='1',
               DISABLE_ERROR_REPORTING='1',CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',
               CLAUDE_CODE_MAX_CONTEXT_TOKENS='262144')
    cmd=[str(BINARY),'-p','/compact','--resume',sid,'--model',model,'--output-format','stream-json','--verbose',
         '--safe-mode','--setting-sources','','--strict-mcp-config','--mcp-config','{"mcpServers":{}}',
         '--tools','']
    started=time.time()
    with (output/'stdout.jsonl').open('w') as out,(output/'stderr.log').open('w') as err:
        result=subprocess.run(cmd,cwd=cwd,env=env,stdout=out,stderr=err,timeout=timeout)
    if result.returncode:raise RuntimeError(f'Official /compact failed; see {output}')
    events=[json.loads(l) for l in (output/'stdout.jsonl').read_text().splitlines() if l.startswith('{')]
    boundaries=[e for e in events if e.get('subtype')=='compact_boundary']
    results=[e for e in events if e.get('type')=='result']
    if not boundaries or not results or results[-1].get('subtype')!='success' or results[-1].get('local_command')!='compact':
        raise RuntimeError(f'Official runtime did not confirm successful compaction: {output}')
    native_rows=[json.loads(l) for l in transcript.read_text().splitlines()]
    summaries=[r for r in native_rows if r.get('isCompactSummary')]
    if not summaries:raise RuntimeError(f'Missing official continuation summary: {output}')
    summary=summaries[-1]['message']['content']
    if not isinstance(summary,str) or not summary.strip():raise RuntimeError('Empty official summary')
    boundary_rows=[r for r in native_rows if r.get('subtype')=='compact_boundary']
    metadata=boundary_rows[-1].get('compactMetadata',{})
    if metadata.get('preservedSegment') and not metadata.get('preservedMessages',{}).get('allUuids'):
        raise RuntimeError('Unsupported official retained-history record; refusing to drop it')
    retained=metadata.get('preservedMessages',{}).get('allUuids',[])
    original_by_uuid={r['uuid']:m for r,m in zip(rows,[m for m in messages if m['role']!='system'])}
    continuation=[{'role':'user','content':summary}]
    # Preserve any input messages kept by the official compactor verbatim.
    # CLI-local command chatter (including "No response requested") is not
    # part of the Mini-SWE input and must not become an agent/model turn.
    continuation.extend(original_by_uuid[u] for u in retained if u in original_by_uuid)
    usage=results[-1].get('model_usage_since_last_result',results[-1].get('modelUsage',{}))
    meta={'runtime':'official Claude Code','version':VERSION,'command':'/compact','seconds':time.time()-started,
          'boundary':boundaries[-1].get('compact_metadata'),
          'usage':{'completion_tokens':sum(v.get('outputTokens',0) for v in usage.values()),
                   'prompt_tokens':sum(v.get('inputTokens',0)+v.get('cacheReadInputTokens',0)+v.get('cacheCreationInputTokens',0) for v in usage.values())},
          'transcript':str(transcript),'model_usage':usage,
          'official_preserved_input_messages':sum(u in original_by_uuid for u in retained),
          'cli_only_messages_excluded':sum(u not in original_by_uuid for u in retained)}
    (output/'result.json').write_text(json.dumps({'continuation':continuation,'metadata':meta},indent=2))
    return continuation,meta


class OfficialCompactionPolicy:
    def __init__(self,token_count,compact,threshold=8192):
        self.count=token_count;self.compact=compact;self.threshold=threshold
        self.continuation=[];self.consumed=1;self.events=[]

    def prepare(self,transcript):
        # Only the system prompt is persistent. The initial user task is part of
        # the native compactor's input, rather than an extra custom preserved copy.
        effective=[transcript[0]]
        effective.extend(self.continuation)
        effective.extend(transcript[self.consumed:])
        before=self.count(effective)
        if before<=self.threshold:return effective
        continuation,meta=self.compact(effective)
        candidate=[transcript[0],*continuation]
        # Do not impose the former custom 2K summary target or truncate/rewrite
        # the official output. The 8K setting is a trigger, not a summary cap.
        event={'before_tokens':before,'after_tokens':self.count(candidate),**meta}
        self.continuation=continuation;self.consumed=len(transcript);self.events.append(event)
        return candidate

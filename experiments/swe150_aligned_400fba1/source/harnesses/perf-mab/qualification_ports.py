"""Controller-only port ownership evidence; no signals, retries of jobs, or probes.

vLLM has no authenticated inherited-listener-FD contract here. Closing the
reservation before launch therefore leaves a race: a failed launch is recorded
and fails this job, never silently starts a second GPU server. Ephemeral port
selection itself is bounded and can skip ports used earlier in this job.
"""
import json
import os
from pathlib import Path
import socket
import time

# The server handoff after a stopped server (perf-hbm job 3428610: the 2-s LISTEN check aborted a complete job while
# the EngineCore child that inherited the listening fd was still tearing down, ~7-8 s): wait up to HANDOFF_LIMIT_S for
# the port's LISTEN (and, in job.idle, the GPU) to clear; a wait above HANDOFF_FINDING_S is the finding HANDOFF_WAIT.
HANDOFF_LIMIT_S=120
HANDOFF_FINDING_S=2.0


def listen_ports(paths=None):
    """Read Linux TCP LISTEN state only. TIME_WAIT is explicitly not a listener."""
    paths=paths if paths is not None else (Path('/proc/net/tcp'),Path('/proc/net/tcp6'))
    result=set()
    for path in paths:
        # IPv6 may be disabled. IPv4 is mandatory for this IPv4 controller.
        if str(path)=='/proc/net/tcp6' and not path.exists():continue
        for line in path.read_text(encoding='ascii').splitlines()[1:]:
            columns=line.split()
            if len(columns)>=4 and columns[3]=='0A':
                result.add(int(columns[1].rsplit(':',1)[1],16))
    return result


class Ports:
    def __init__(self,root,*,listener=listen_ports,socket_factory=socket.socket,max_attempts=32):
        self.root=Path(root);self.listener=listener;self.socket_factory=socket_factory
        self.max_attempts=max_attempts;self.used=set();self.active=None

    def event(self,policy,event,**fields):
        row=dict(policy=policy,event=event,**fields)
        with (self.root/'ports.jsonl').open('ab') as stream:
            stream.write((json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode())
            stream.flush();os.fsync(stream.fileno())

    def allocate(self,policy):
        if self.active is not None:raise RuntimeError('previous policy handoff incomplete')
        for attempt in range(1,self.max_attempts+1):
            with self.socket_factory(socket.AF_INET,socket.SOCK_STREAM) as probe:
                probe.bind(('127.0.0.1',0))
                port=probe.getsockname()[1]
            if port in self.used:
                self.event(policy,'skip_previously_selected',port=port,attempt=attempt)
                continue
            self.used.add(port);self.active=(policy,port,attempt)
            self.event(policy,'allocated',port=port,attempt=attempt,host='127.0.0.1',
                       fd_handoff=False,launch_race_action='fail_job_preserve_evidence')
            return port,attempt
        self.event(policy,'allocation_exhausted',attempts=self.max_attempts)
        raise RuntimeError('ephemeral port allocation exhausted without reusing previous ports')

    def stopped(self,policy,process,*,timeout=HANDOFF_LIMIT_S,clock=time.monotonic,sleep=time.sleep):
        """After the APIServer parent exited: wait up to `timeout` s for its port's LISTEN to disappear (the EngineCore /
        worker child that inherited the listening fd may still be tearing down: ~7-8 s in perf-hbm job 3428610, more
        with --deferred-completion-io's final writer flush); fail only past the limit. Returns the seconds waited
        (recorded in the handoff event; the job reports a wait above HANDOFF_FINDING_S as the finding HANDOFF_WAIT)."""
        if self.active is None or self.active[0]!=policy:raise RuntimeError('port handoff ownership mismatch')
        _,port,attempt=self.active
        code=process.poll()
        if code is None:
            self.event(policy,'handoff_failed',port=port,attempt=attempt,reason='process_still_running')
            raise RuntimeError('previous APIServer process has not exited')
        start=clock();deadline=start+timeout
        while port in self.listener():
            if clock()>=deadline:
                self.event(policy,'handoff_failed',port=port,attempt=attempt,reason='tcp_listen_remains',exitcode=code,
                           listen_wait_s=round(clock()-start,3),limit_s=timeout)
                raise RuntimeError('previous HTTP port still has a LISTEN socket %.0f s after the APIServer exited; no '
                                   'other process was touched'%timeout)
            sleep(.025)
        waited=round(clock()-start,3)
        self.event(policy,'handoff_complete',port=port,attempt=attempt,exitcode=code,
                   process_exited=True,listening=False,time_wait_is_not_listener=True,listen_wait_s=waited,
                   limit_s=timeout)
        self.active=None
        return waited

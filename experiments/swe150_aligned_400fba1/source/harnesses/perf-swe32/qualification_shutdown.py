"""Graceful parent-only signal; process-group signals are failed escalation."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

def stop_server(process,receipt=None):
    if receipt is None:receipt={}
    receipt.update(pid=process.pid,target='APIServer parent only',signal=None,signal_sent=False)
    before=process.poll();receipt['pre_signal_exitcode']=before
    if before is not None:
        raise RuntimeError('owned server exited before explicit SIGTERM: '+str(before))
    # Popen.send_signal may suppress ProcessLookupError or return without sending
    # if a second poll observes exit. os.kill reports kernel acceptance directly.
    os.kill(process.pid,signal.SIGTERM)
    receipt.update(signal='SIGTERM',signal_sent=True)
    try:code=process.wait(timeout=45)
    except subprocess.TimeoutExpired as original:
        errors=[original]
        try:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        except BaseException as exc:errors.append(exc)
        raise BaseExceptionGroup('owned server required group escalation; cleanup unqualified',errors)
    receipt['exitcode']=code
    if code!=0:raise RuntimeError('owned server exited abnormally: '+str(code))

def stop_preserving(process,receipt=None):
    original=sys.exception()
    try:stop_server(process,receipt=receipt)
    except BaseException as cleanup:
        if original is not None and original is not cleanup:
            raise BaseExceptionGroup('qualification and shutdown failed',[original,cleanup]) from None
        raise

def verify_closure(output,ready):
    output=Path(output);raw=(output/'engine-cleanup.json').read_bytes()
    engine=json.loads(raw);parent=json.loads((output/'parent-cleanup.json').read_bytes())
    if any(value.get('epoch')!=ready['epoch'] or value.get('session_id')!=ready['session_id']
           or value.get('joined') is not True for value in (engine,parent)):
        raise RuntimeError('closure receipt identity/join mismatch')
    if (engine.get('runtime_manifest')!=ready['runtime_manifest'] or
            parent.get('cleanup_sha256')!=hashlib.sha256(raw).hexdigest() or
            parent.get('natural_exit') is not True or parent.get('child_exitcode')!=0):
        raise RuntimeError('closure ACK/hash/natural exit mismatch')

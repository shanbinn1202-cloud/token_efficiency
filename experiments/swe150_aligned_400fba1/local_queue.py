"""Record and independently release the four local allocations. No shared Docker host is used."""
import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

from verify_manifest import ROOT,verify

METHODS=("random","sliding_window","agentkv","claude_compression")
LEDGER=ROOT/"local-jobs.json"

def save(state):
    temp=LEDGER.with_suffix(".tmp")
    temp.write_text(json.dumps(state,indent=2)+"\n")
    temp.replace(LEDGER)

def create(state):
    for method in METHODS:
        if method in state:
            continue
        name="s150-local-"+method
        cmd=["sbatch","--parsable","--hold","--partition=ghx4","--account=bcjw-dtai-gh",
             "--time=05:00:00","--mem=145G","--nodes=1","--ntasks=1","--cpus-per-task=32",
             "--gpus-per-node=1","--no-requeue","--chdir="+str(ROOT),"--job-name="+name,
             "--output="+str(ROOT/"runs/slurm-%j.out"),str(ROOT/"local-job.sh"),method]
        reply=subprocess.check_output(cmd,text=True).strip().split(";")[0]
        assert reply.isdigit(),reply
        state[method]={"job_id":reply,"name":name,"status":"held_for_local_checks",
                       "partition":"ghx4","walltime":"05:00:00","submitted":time.time()}
        save(state)
        print(method,reply,flush=True)

def retire_ec2(state):
    assert len(state)==4
    old=json.loads((ROOT/"revisions/ec2-submissions/jobs.json").read_text())
    q=subprocess.check_output(["squeue","-h","-u","ssun4","-o","%i|%j|%T|%r"],text=True)
    jobs={l.split("|")[0]:l.split("|")[1:] for l in q.splitlines()}
    ids=[]
    for method in METHODS:
        jid=old[method]["job_id"]
        if jid not in jobs:
            continue
        assert jobs[jid]==["s150-"+method,"PENDING","JobHeldUser"],(jid,jobs[jid])
        ids.append(jid)
    if ids:
        subprocess.run(["scancel",*ids],check=True)
    print("Retired superseded EC2 allocations:",ids)

def release(state):
    verify()
    subprocess.run(["/usr/bin/python3","-B",str(ROOT/"check_local_assets.py")],check=True)
    assert len(state)==4
    q=subprocess.check_output(["squeue","-h","-u","ssun4","-o","%i|%j|%T|%r"],text=True)
    jobs={l.split("|")[0]:l.split("|")[1:] for l in q.splitlines()}
    for method,row in state.items():
        if row["status"] in ("released","eligible_after_interactive"):
            continue
        jid=row["job_id"]
        assert jobs.get(jid)==[row["name"],"PENDING","JobHeldUser"],(jid,jobs.get(jid))
        detail=subprocess.check_output(["scontrol","-o","show","job",jid],text=True)
        fields=dict(x.split("=",1) for x in detail.split() if "=" in x)
        tres=set(fields["ReqTRES"].split(","))
        assert fields["CPUs/Task"]=="32" and {"cpu=32","mem=145G","gres/gpu=1"}<=tres, (jid, tres)
        pilot=json.loads((ROOT/"interactive-check/queue.json").read_text())
        dependency="afterok:"+str(pilot["job_id"])
        active_dependency=fields["Dependency"].split("(")[0]
        if active_dependency!=dependency:
            from benchmark import require_interactive_check
            require_interactive_check()
        subprocess.run(["scontrol","release",jid],check=True)
        row.update(status="eligible_after_interactive",released=time.time(),validation_job_id=pilot["job_id"],dependency=dependency)
        save(state)
        print("released",method,jid,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["create","retire-ec2","release"])
    a=p.parse_args()
    with (ROOT/"queue-control.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
        {"create":create,"retire-ec2":retire_ec2,"release":release}[a.action](state)

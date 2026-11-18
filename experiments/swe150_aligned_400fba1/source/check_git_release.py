"""Run the complete test suite on bytes exported by Git, not the worktree.

By default every test module runs in its own process (unittest's own discover with that file as the pattern) on
a pool of --jobs workers; the modules in SERIAL_ONLY run after the pool, one at a time. --serial is the former
single-process run. Either way the result line ends in one "Ran N tests in S" and one OK / FAILED line in
unittest's own format, so receipts read both alike. The test modules are the files of tests/ (TESTS), discovered
there with tests/ as the top-level directory, from the archive root as the working directory, with the root and
tests/ on PYTHONPATH (the release repository's layout)."""
import argparse
import concurrent.futures
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

# Run alone, after the pool: their timing bounds assume an otherwise idle machine.
SERIAL_ONLY={
    'test_chat_text_protocol.py':'validate_tasks of every slot under 0.05 s',
    'test_goal_rewriter_timing.py':'thread start-up and scheduling within SLACK_MS (60 ms)',
    'test_goal_quoted_literals.py':"a child process's heartbeat ticks and its 1.0 / 1.5 s bounds",
}
# Seconds per module measured in a parallel run (c07d42a, ten workers): the pool starts the longest first, so
# the tail stays short. A module not listed here starts before all of them (by file size), so a new slow module
# cannot end up last. Only the order depends on these numbers.
COSTS={'test_adm_overlap.py':79,'test_adm_phase1_step.py':17,'test_admission_cpu.py':0,'test_align_ctx.py':16,
    'test_allocation.py':1,'test_chat_text_e2e.py':69,'test_chat_text_protocol.py':3,'test_chat_text_replay.py':0,
    'test_chat_text_server.py':2,'test_checkpoint.py':4,'test_connector_pool.py':5,'test_ctx_256k.py':6,
    'test_cudagraph.py':9,'test_decode_fastpath.py':18,'test_decode_fastpath2.py':35,'test_decode_fastpath3.py':24,
    'test_decouple_e2e.py':25,'test_decouple_evidence.py':5,'test_decouple_integration.py':4,
    'test_decouple_legacy.py':32,'test_decouple_selection.py':13,'test_decouple_wp2.py':11,
    'test_decouple_wp3_arena.py':4,'test_decouple_wp3_gates.py':4,'test_decouple_wp3_owner.py':8,
    'test_decouple_wp3_rows.py':3,'test_decouple_wp3_vectors.py':8,'test_deferred_completion.py':104,
    'test_digest_once.py':18,'test_gdn_handoff.py':34,'test_gemma_assembly.py':1,'test_gemma_capacity.py':2,
    'test_gemma_chat_text.py':4,'test_gemma_chat_text_e2e.py':53,'test_gemma_chat_text_goal.py':102,
    'test_gemma_chat_text_replay.py':0,'test_gemma_chunked.py':14,'test_gemma_deferred_completion.py':141,
    'test_gemma_e2e.py':25,'test_gemma_engine.py':1,'test_gemma_fastpath.py':10,'test_gemma_fastpath3.py':19,
    'test_gemma_flags.py':37,'test_gemma_forward4k.py':21,'test_gemma_full.py':9,'test_gemma_goal.py':13,
    'test_gemma_goal_overlap.py':49,'test_gemma_goal_rewriter.py':64,'test_gemma_graphs.py':9,
    'test_gemma_install.py':2,'test_gemma_kv_pools.py':1,'test_gemma_kv_session.py':2,
    'test_gemma_kv_session_e2e.py':23,'test_gemma_overlap.py':14,'test_gemma_profile.py':0,
    'test_gemma_release.py':23,'test_gemma_renderer.py':0,'test_gemma_selection.py':8,'test_gemma_startup.py':3,
    'test_gemma_window_rows.py':3,'test_gemma_worker.py':3,'test_goal_overlap.py':58,
    'test_goal_overlap_thinking.py':423,'test_goal_overlap_vllm.py':10,'test_goal_quoted_literals.py':5,
    'test_goal_rewriter_batch.py':10,'test_goal_rewriter_timing.py':2,'test_goal_rewriter_vllm.py':54,
    'test_goal_text.py':14,'test_goal_tools.py':0,'test_host_trim.py':37,'test_http.py':2,'test_idle_release.py':32,
    'test_installer.py':28,'test_int_flags.py':14,'test_integrate_flags.py':185,'test_kv_layout.py':0,
    'test_kv_pools.py':1,'test_kv_session_e2e.py':68,'test_kv_session_protocol.py':5,'test_kv_session_server.py':1,
    'test_language_model_only.py':16,'test_lease.py':0,'test_ms16.py':18,'test_ms2_admission.py':23,
    'test_ms2_admission_timeout.py':13,'test_ms2_e2e.py':97,'test_ms2_goal_async.py':49,'test_ms2_worker.py':5,
    'test_ms_e2e.py':43,'test_ms_engine.py':16,'test_ms_frontend.py':2,'test_ms_pool.py':17,
    'test_mswea_protocol.py':0,'test_payload_token.py':3,'test_qwen_kv25.py':1,'test_random_fill.py':23,
    'test_random_fill_legacy.py':41,'test_selection_contract.py':3,'test_selection_fast.py':24,
    'test_selection_stageb.py':4,'test_sessions_contract.py':0,'test_stageb_contract.py':0,'test_stageb_e2e.py':14,
    'test_stageb_engine.py':1,'test_stageb_worker.py':4,'test_startup.py':2,'test_step_profile.py':12,
    'test_terminus2_goal.py':1,'test_thinking_cut.py':180,'test_thinking_on.py':19,
    'test_tokenizer_per_request.py':0,'test_torch_profile.py':1,'test_triton_attn.py':7}
JOBS=10;MODULE_SECONDS=1200;TOTAL_SECONDS=3600
TESTS='tests'                    # the test modules' directory, relative to the archive root
VALID_MODULE=re.compile(r'[_a-z]\w*\.py$',re.IGNORECASE)   # unittest.loader.VALID_MODULE_NAME

# One worker: unittest.main as `python -m unittest discover -s TESTS -p PATTERN` runs it (same argv, sys.path[0],
# warnings and exit codes), with a result that also records each test id's outcome. Run with -c, so a spawned
# multiprocessing child never re-runs it as its main module.
WORKER=r'''
import json,os,sys,time,types,unittest
start,out,pattern=sys.argv[1:4]
sys.path[0]=os.getcwd()
sys.argv=['python -m unittest','discover','-s',start,'-p',pattern]
outcomes=[]
class Result(unittest.TextTestResult):
    def addSuccess(self,test):super().addSuccess(test);outcomes.append([test.id(),'pass'])
    def addFailure(self,test,err):super().addFailure(test,err);outcomes.append([test.id(),'fail'])
    def addError(self,test,err):super().addError(test,err);outcomes.append([test.id(),'error'])
    def addSkip(self,test,reason):super().addSkip(test,reason);outcomes.append([test.id(),'skip'])
    def addExpectedFailure(self,test,err):super().addExpectedFailure(test,err);outcomes.append([test.id(),'expected_failure'])
    def addUnexpectedSuccess(self,test):super().addUnexpectedSuccess(test);outcomes.append([test.id(),'unexpected_success'])
    def addSubTest(self,test,subtest,err):
        super().addSubTest(test,subtest,err)
        if err is not None:outcomes.append([subtest.id(),'fail' if issubclass(err[0],test.failureException) else 'error'])
class Runner(unittest.TextTestRunner):
    resultclass=Result
    def run(self,test):
        started=time.perf_counter();result=super().run(test);seconds=time.perf_counter()-started
        # The module's own namespace, not getattr: torch.ops answers any attribute name.
        hooks=sorted(n for n,m in list(sys.modules.items()) if isinstance(m,types.ModuleType) and 'load_tests' in vars(m)
            and isinstance(vars(m).get('__file__'),str)
            and os.path.dirname(os.path.abspath(vars(m)['__file__']))==os.path.abspath(start))
        with open(out,'w',encoding='utf-8') as f:
            json.dump(dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),expected_failures=len(result.expectedFailures),
                unexpected_successes=len(result.unexpectedSuccesses),successful=result.wasSuccessful(),
                seconds=seconds,outcomes=outcomes,load_tests=hooks),f)
        return result
unittest.main(module=None,argv=sys.argv,testRunner=Runner)
'''

def extract(raw,root):
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            path=root/member.name
            if not path.resolve().is_relative_to(root.resolve()):raise ValueError('archive path escape')
            if member.isdir():path.mkdir(parents=True,exist_ok=True);continue
            if not member.isfile():raise ValueError('nonregular Git release entry: '+member.name)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())

def modules(root):
    """The files discover loads from the start directory (root/TESTS). A package directory there fails closed:
    discover would import it and search it, which one process per top-level file does not reproduce."""
    start=root/TESTS
    if not start.is_dir():raise ValueError('no %s/ directory in the Git release'%TESTS)
    names=[]
    for name in sorted(os.listdir(start)):
        path=start/name
        if path.is_dir():
            if (path/'__init__.py').is_file():raise ValueError('package directory %s in the release: run --serial'%name)
        elif path.is_file() and VALID_MODULE.match(name) and fnmatch.fnmatch(name,'test_*.py'):names.append(name)
    # No module is no pass (a script that looked elsewhere would otherwise report an empty suite as passed).
    if not names:raise ValueError('no test_*.py modules in %s/ of the Git release'%TESTS)
    return names

def tree(root):
    return {p.relative_to(root).as_posix():(p.stat().st_size,p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}

def kill(process):
    if os.name=='nt':subprocess.run(['taskkill','/F','/T','/PID',str(process.pid)],capture_output=True)
    else:os.killpg(process.pid,signal.SIGKILL)

def run_module(root,pattern,out,env,seconds):
    """One worker process (discover in root/TESTS, from root); the whole process tree is killed when it outlives
    `seconds`."""
    started=time.monotonic()
    process=subprocess.Popen([sys.executable,'-B','-c',WORKER,str(root/TESTS),str(out),pattern],cwd=root,env=env,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,errors='replace',start_new_session=os.name!='nt')
    try:stdout,stderr=process.communicate(timeout=seconds);timed_out=False
    except subprocess.TimeoutExpired:kill(process);stdout,stderr=process.communicate();timed_out=True
    record=json.loads(out.read_text('utf-8')) if out.is_file() and not timed_out else None
    ok=(record is not None and record['successful'] and
        (process.returncode==0 or (process.returncode==5 and record['run']==0 and not record['skipped'])))
    return dict(name=pattern,returncode=process.returncode,timed_out=timed_out,seconds=time.monotonic()-started,
        record=record,ok=ok,output=stdout+stderr)

def summary(records,seconds):
    """TextTestRunner.run's closing lines for the summed records."""
    total={k:sum(r[k] for r in records) for k in ('run','failures','errors','skipped','expected_failures',
        'unexpected_successes')}
    run=total['run'];infos=[]
    if not all(r['successful'] for r in records):
        word='FAILED'
        if total['failures']:infos.append('failures=%d'%total['failures'])
        if total['errors']:infos.append('errors=%d'%total['errors'])
    elif run==0 and not total['skipped']:word='NO TESTS RAN'
    else:word='OK'
    if total['skipped']:infos.append('skipped=%d'%total['skipped'])
    if total['expected_failures']:infos.append('expected failures=%d'%total['expected_failures'])
    if total['unexpected_successes']:infos.append('unexpected successes=%d'%total['unexpected_successes'])
    return '-'*70+'\nRan %d test%s in %.3fs\n\n'%(run,run!=1 and 's' or '',seconds)+word+(' (%s)'%', '.join(infos) if infos else '')

def outcome_digest(outcomes):
    pairs=sorted(map(tuple,outcomes))
    return len(pairs),hashlib.sha256(json.dumps(pairs).encode('utf-8')).hexdigest()

def run_tests(root,env,mode,jobs,module_seconds,total_seconds):
    """'parallel': one process per module, `jobs` at a time, SERIAL_ONLY after them; 'record': one process for
    the whole suite, as --serial, with each test's outcome. Returns (module names, results, wall seconds)."""
    names=modules(root)
    with tempfile.TemporaryDirectory(prefix='selective-prefix-results-') as results_dir:
        out=lambda name:Path(results_dir)/(name.replace('*','all')+'.json')
        started=time.monotonic();deadline=started+total_seconds
        if mode=='record':
            return names,[run_module(root,'test_*.py',out('test_*.py'),env,total_seconds)],time.monotonic()-started
        def one(name):
            left=deadline-time.monotonic()
            if left<=0:return dict(name=name,returncode=None,timed_out=True,seconds=0.0,record=None,ok=False,
                output='not started: the %d s total limit has passed'%total_seconds)
            return run_module(root,name,out(name),env,min(module_seconds,left))
        pooled=sorted((n for n in names if n not in SERIAL_ONLY),
            key=lambda n:(n in COSTS,-(COSTS[n] if n in COSTS else (root/TESTS/n).stat().st_size),n))
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:results=list(pool.map(one,pooled))
        results+=[one(n) for n in names if n in SERIAL_ONLY]
        return names,results,time.monotonic()-started

def check(repo,ref='HEAD',serial=False,record=False,jobs=JOBS,module_seconds=MODULE_SECONDS,
          total_seconds=TOTAL_SECONDS,outcomes=None):
    repo=Path(repo).resolve()
    commit=subprocess.run(['git','-C',str(repo),'rev-parse','--verify',ref+'^{commit}'],
        capture_output=True,text=True,check=True).stdout.strip()
    raw=subprocess.run(['git','-C',str(repo),'archive','--format=tar',commit],capture_output=True,check=True).stdout
    with tempfile.TemporaryDirectory(prefix='selective-prefix-git-') as directory:
        root=Path(directory)
        extract(raw,root)
        # One OpenMP thread, forced: without it the suite outran its former 840 s
        # under load. 1800 s (was 1200 s): with the Gemma profile and the Goal
        # rewriter work merged (integrate-1015) the 1,065 tests ran 1,191 s on the
        # development machine and the archive run timed out at 1200 s. 3600 s (was
        # 1800 s): integrate-1031's 1,643 tests ran 1,697 s and keep-overlap's 1,652
        # ran 1,722 s (its slower tests gated), and with keep-overlap's keep-mode tests
        # in the archive suite the run timed out at 1800 s three times.
        env=dict(os.environ,SELECTIVE_PREFIX_ARCHIVE_TEST='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
                 PYTHONPATH=os.pathsep.join((str(root),str(root/TESTS))))
        if serial:
            run=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',TESTS,'-t',TESTS,'-p','test_*.py'],
                cwd=root,env=env,capture_output=True,text=True,timeout=3600)
            if run.returncode:raise RuntimeError('Git archive tests failed:\n'+run.stdout+run.stderr)
            return dict(status='GIT_ARCHIVE_TESTS_PASS',commit=commit,tests=run.stderr.strip())
        before=tree(root)
        mode='record' if record else 'parallel'
        names,results,wall=run_tests(root,env,mode,jobs,module_seconds,total_seconds)
        after=tree(root)
        changed=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
    records=[r['record'] for r in results if r['record'] is not None]
    hooks=sorted({h for r in records for h in r['load_tests']})
    all_outcomes=[o for r in records for o in r['outcomes']]
    count,digest=outcome_digest(all_outcomes)
    if outcomes is not None:
        totals={k:sum(r[k] for r in records) for k in ('run','failures','errors','skipped','expected_failures',
            'unexpected_successes')}
        Path(outcomes).write_text(json.dumps(dict(commit=commit,mode=mode,jobs=jobs if mode=='parallel' else 1,
            wall_seconds=wall,totals=totals,modules={r['name']:dict(seconds=r['seconds'],ok=r['ok']) for r in results},
            outcomes=sorted(map(tuple,all_outcomes))),indent=1)+'\n','utf-8')
    if mode=='record':head='mode: record, one process for %d modules'%len(names)
    else:head='mode: parallel, %d modules, %d workers, %d alone after the pool'%(
        len(results),jobs,sum(r['name'] in SERIAL_ONLY for r in results))
    lines=[head,'outcomes: %d test ids, sha256 %s'%(count,digest),
        'archive tree unchanged' if not changed else 'archive tree changed: %d paths: %s'%(len(changed),', '.join(changed[:10]))]
    if mode=='parallel':lines+=['module %s: %d run, %d skipped, %.1f s'%(r['name'],r['record']['run'] if r['record'] else 0,
        r['record']['skipped'] if r['record'] else 0,r['seconds']) for r in sorted(results,key=lambda r:r['name'])]
    seconds=wall if mode=='parallel' else (records[0]['seconds'] if records else wall)
    tests='\n'.join(lines)+'\n'+summary(records,seconds) if records else '\n'.join(lines)
    failed=[r for r in results if not r['ok']]
    if failed or hooks or len(records)!=len(results):
        detail=''.join('\n==== %s: returncode %s%s\n%s'%(r['name'],r['returncode'],', timed out' if r['timed_out'] else '',
            r['output']) for r in failed)
        if hooks:detail+='\n==== load_tests in %s: the per-module pattern differs from test_*.py; run --serial'%', '.join(hooks)
        raise RuntimeError('Git archive tests failed:'+detail+'\n'+tests)
    return dict(status='GIT_ARCHIVE_TESTS_PASS',commit=commit,tests=tests)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--ref',default='HEAD')
    parser.add_argument('--serial',action='store_true',help='the former single-process run (unittest discover)')
    parser.add_argument('--record',action='store_true',help='one process as --serial, with per-test outcomes')
    parser.add_argument('--jobs',type=int,default=JOBS,help='parallel workers (default %d)'%JOBS)
    parser.add_argument('--outcomes',type=Path,help='write per-test outcomes and per-module seconds (not --serial)')
    args=parser.parse_args()
    if args.serial and (args.record or args.outcomes):parser.error('--serial takes neither --record nor --outcomes')
    print(json.dumps(check(args.repo,args.ref,serial=args.serial,record=args.record,jobs=args.jobs,
        outcomes=args.outcomes),sort_keys=True))

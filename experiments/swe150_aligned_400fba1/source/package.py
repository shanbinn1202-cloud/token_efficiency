"""Create a deterministic core-only source archive; never consult Git or network."""
import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile
from install import ROOT,digest,encoded,need,save,verify_package

PUBLIC_FILES=('README.md','README.zh-CN.md','LICENSE','.gitattributes','INSTALL_RELEASE.json','SOURCE_SHA256SUMS.txt',
    'install.py','restore.py','serve.py','package.py','tests/test_installer.py','tests/test_http.py',
    'tests/test_allocation.py','tests/test_checkpoint.py','tests/test_startup.py','tests/test_goal_tools.py','tests/test_selection_contract.py','tests/test_mswea_protocol.py','tests/test_selection_stageb.py','tests/test_stageb_contract.py','tests/test_stageb_worker.py','tests/test_stageb_engine.py','tests/test_stageb_e2e.py','check_git_release.py',
    'THIRD_PARTY_NOTICES.txt','patches/v0.24.0.diff',
    'examples/http_client.py','examples/task.json','examples/messages.json',
    'tools/docs_set.py','docs/status.md','docs/install.md','docs/running.md','docs/reference.md','docs/design.md','docs/benchmarks/swe-bench-mini-swe-agent.md','docs/history.md','docs/status.zh-CN.md','docs/install.zh-CN.md','docs/running.zh-CN.md','docs/reference.zh-CN.md','docs/design.zh-CN.md','docs/benchmarks/swe-bench-mini-swe-agent.zh-CN.md','docs/history.zh-CN.md',
    'docs/known-limits.md','docs/known-limits.zh-CN.md')
def build(root=ROOT):
    definition=verify_package(root)
    names=sorted((*PUBLIC_FILES,*('source-overlay/'+n for n in definition['source_files'])))
    buffer=io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',filename='',mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as archive:
            for name in names:
                path=root/name;need(path.is_file() and not path.is_symlink(),'regular public source required: '+name)
                raw=path.read_bytes();member=tarfile.TarInfo('selective-prefix/'+name)
                member.size=len(raw);member.mode=0o644;member.mtime=member.uid=member.gid=0
                member.uname=member.gname='';archive.addfile(member,io.BytesIO(raw))
    return buffer.getvalue()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raw=build()
    need(not args.output.exists() and args.output.parent.is_dir(),'new archive under existing parent required')
    save(args.output,raw)
    print(json.dumps(dict(status='CORE_SOURCE_ARCHIVE',archive_sha256=digest(raw),bytes=len(raw),gpu_tested=False),sort_keys=True))
if __name__=='__main__':main()

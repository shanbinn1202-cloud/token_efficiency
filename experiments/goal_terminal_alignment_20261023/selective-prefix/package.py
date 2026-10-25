"""Create a deterministic core-only source archive; never consult Git or network."""
import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile
from install import ROOT,digest,encoded,need,save,verify_package

PUBLIC_FILES=('README.md','README.zh-CN.md','LICENSE','.gitattributes','INSTALL_RELEASE.json','SOURCE_SHA256SUMS.txt',
    'install.py','restore.py','serve.py','package.py','test_installer.py','test_http.py',
    'test_allocation.py','test_checkpoint.py','test_startup.py','test_goal_tools.py','test_selection_contract.py','test_mswea_protocol.py','test_selection_stageb.py','test_stageb_contract.py','test_stageb_worker.py','test_stageb_engine.py','test_stageb_e2e.py','check_git_release.py',
    'THIRD_PARTY_NOTICES.txt','patches/v0.24.0.diff',
    'examples/http_client.py','examples/task.json','examples/messages.json')
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

"""Safe extraction of one tar.xz into a new directory (local only): regular files and directories, relative POSIX
names, no traversal, links or duplicates; prints one JSON line {files, bytes, root}. Used by fetch_run.ps1 for the tbench
package (the DeltaAI archives are extracted by verify_evidence.py).

Usage: python -B unpack.py <archive.tar.xz> <new directory>
"""
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile


def main(archive, dest):
    dest = Path(dest)
    dest.mkdir(parents=True)
    files, total, roots = 0, 0, set()
    with tarfile.open(archive, 'r:xz') as stream:
        seen = set()
        for member in stream:
            name = member.name.rstrip('/')
            parts = PurePosixPath(name).parts
            if (not name or name.startswith('/') or '\\' in name or ':' in name or any(p in ('', '.', '..') for p in parts)
                    or not (member.isreg() or member.isdir())):
                raise SystemExit('unsafe package member: %r' % member.name)
            if name in seen:
                raise SystemExit('duplicate package member: ' + name)
            seen.add(name)
            roots.add(parts[0])
            target = dest.joinpath(*parts)
            if os.name == 'nt' and len(str(target.resolve())) >= 250:
                raise SystemExit('local path too long for Windows: ' + name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            data = stream.extractfile(member).read()
            with target.open('xb') as out:
                out.write(data)
            files += 1
            total += len(data)
    print(json.dumps(dict(files=files, bytes=total, roots=sorted(roots), dest=str(dest))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2]))

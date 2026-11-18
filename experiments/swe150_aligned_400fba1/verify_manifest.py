"""Verify the frozen benchmark code before staging or releasing the four allocations."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent

def sha(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()

def verify():
    manifest=json.loads((ROOT/"files.json").read_text())
    if not manifest:
        raise ValueError("Missing benchmark file digests")
    for name,expected in manifest.items():
        if sha(ROOT/name)!=expected:
            raise ValueError(f"Benchmark file changed: {name}")
    return len(manifest)

if __name__=="__main__":
    print(f"Verified {verify()} files")

"""Derive the AgentKV release from the node's clean 400fba1 installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
COMMIT = "400fba1ca414904d8a48222de4315ff678e18446"
PIN = hashlib.sha256((ROOT / "derived-pins.json").read_bytes()).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_pair(root, filename, value):
    path = root / filename
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    # Unlink the shared hard link first. The original release remains immutable.
    path.unlink(missing_ok=True)
    path.write_bytes(encoded)
    pin = path.with_suffix(".sha256")
    pin.unlink(missing_ok=True)
    h = hashlib.sha256(encoded).hexdigest()
    pin.write_text(h + "  " + filename + "\n")
    return h


def stage(local):
    goal = local / "engines/goal"
    target = goal.parent / "baselines"
    original = json.loads((goal / "runtime-manifest.json").read_text())
    latest = json.loads((ROOT / "upstream-pins.json").read_text())
    if original["source_files"] != latest:
        raise ValueError("Expected exactly the 400fba1 installed upstream overlay")
    changes = json.loads((ROOT / "derived-pins.json").read_text())
    source_identity = digest(goal / "release-index.json")
    if target.exists():
        installed = json.loads((target / "runtime-manifest.json").read_text())
        derivation = installed.get("derivation", {})
        if derivation.get("upstream_index") != source_identity or derivation.get("adapter_sha256") != PIN:
            raise RuntimeError("A different derived release already occupies this allocation")
        for rel, pins in changes.items():
            if digest(target / "source" / rel) != pins["derived"]:
                raise RuntimeError("Existing derived release differs: " + rel)
        return target
    with tempfile.TemporaryDirectory(dir=goal.parent) as temp:
        work = Path(temp) / "release"
        shutil.copytree(goal, work, copy_function=os.link)
        for folder, _, _ in os.walk(work):
            os.chmod(folder, 0o700)
        index = json.loads((goal / "release-index.json").read_text())
        for rel, pins in changes.items():
            dest = work / "source" / rel
            overlay = ROOT / "overlay" / rel
            old_hash = digest(dest) if dest.exists() else None
            if old_hash != pins["base"] or digest(overlay) != pins["derived"]:
                raise ValueError("Upstream or adapter version mismatch: " + rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.unlink(missing_ok=True)
            shutil.copyfile(overlay, dest)
            original["source_files"][rel] = pins["derived"]
            index["files"]["source/" + rel] = pins["derived"]
        original["derivation"] = {"source_commit": COMMIT, "upstream_index": source_identity,
                                  "adapter_sha256": PIN, "adapted_files": {name: v["derived"] for name,v in changes.items()}}
        index["manifest_sha256"] = write_pair(work, "runtime-manifest.json", original)
        for name in ["runtime-manifest.json", "runtime-manifest.sha256"]:
            index["files"][name] = digest(work / name)
        write_pair(work, "release-index.json", index)
        work.rename(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", required=True, type=Path)
    print(stage(parser.parse_args().local), flush=True)

"""Check the published code against the manifest from the interactive run."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERATED = {"assets/dataset-first150.json", "assets/tasks/tasks-pool.json"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, help="verify the two generated inputs in this assets directory")
    args = parser.parse_args()
    release = json.loads((ROOT / "release.json").read_text())
    manifest = ROOT / "files.json"
    if sha256(manifest) != release["cluster_manifest_sha256"]:
        raise ValueError("The original cluster manifest has changed")
    files = json.loads(manifest.read_text())
    source = release["upstream"]["source_path"] + "/"
    source_count = sum(name.startswith(source) for name in files)
    if source_count != release["upstream"]["file_count"]:
        raise ValueError("The number of upstream files differs from the published release")
    checked = 0
    for name, expected in files.items():
        path = Path(name)
        if path.is_absolute():
            continue  # Site-specific runtime packages have their own frozen digests.
        path = ROOT / path
        if name in GENERATED and args.assets:
            path = args.assets / Path(name).relative_to("assets")
        elif name in GENERATED and not path.exists():
            continue
        if sha256(path) != expected:
            raise ValueError(f"Frozen file differs: {name}")
        checked += 1
    print(f"Verified {checked} files, including {source_count} upstream files, against the cluster manifest.")


if __name__ == "__main__":
    main()

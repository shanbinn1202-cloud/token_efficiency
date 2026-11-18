"""Restore the pinned sources and runtimes used by the SWE-150 agent clients."""
import argparse
from base64 import b64encode
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen
import zipfile

HERE = Path(__file__).resolve().parent


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm)


def checked(path, expected):
    if not path.is_file() or digest(path).hexdigest() != expected:
        raise ValueError(f"Missing or different frozen file: {path}")


def install(stream, path, expected, mode=0o644):
    if path.exists():
        checked(path, expected)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = Path(tmp.name)
    try:
        with name.open("wb") as dst:
            shutil.copyfileobj(stream, dst, length=1024 * 1024)
        checked(name, expected)
        name.chmod(mode)
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)


def restore_files(root, checkout, records):
    for target, (source, expected) in records.items():
        with (checkout / source).open("rb") as stream:
            install(stream, root / target, expected)


def restore_mini(root, pin, wheel, tmp):
    if wheel is None:
        subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:",
                        "--no-deps", "--dest", str(tmp), f"{pin['distribution']}=={pin['version']}"],
                       check=True)
        wheel, = tmp.glob("*.whl")
    checked(wheel, pin["wheel_sha256"])
    with zipfile.ZipFile(wheel) as archive:
        for target, expected in pin["files"].items():
            with archive.open(target) as src:
                install(src, root / "mini242" / target, expected)


def restore_claude(root, pin, archive, tmp):
    stored = root / pin["archive_destination"]
    binary = root / pin["destination"]
    if stored.is_file():
        checked(stored, pin["archive_sha256"])
        archive = stored
    elif archive is None:
        archive = tmp / "claude-code.tgz"
        with urlopen(pin["url"], timeout=120) as src, archive.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    integrity = "sha512-" + b64encode(digest(archive, "sha512").digest()).decode()
    if integrity != pin["integrity"]:
        raise ValueError("Claude Code package SHA-512 integrity check failed")
    if archive != stored:
        with archive.open("rb") as source:
            install(source, stored, pin["archive_sha256"])
    with tarfile.open(stored, "r:gz") as package:
        for name, expected in pin["files"].items():
            path = binary.parent / name
            if path.exists():
                checked(path, expected)
                continue
            member = package.getmember("package/" + name)
            if not member.isfile():
                raise ValueError(f"Invalid archive member: {name}")
            with package.extractfile(member) as src:
                install(src, path, expected, mode=0o755 if name == "claude" else 0o644)
    if not os.access(binary, os.X_OK):
        raise PermissionError(f"Claude Code executable is not executable: {binary}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path, help="token_efficiency checkout")
    parser.add_argument("--upstream", required=True, type=Path, help="selective-prefix checkout at 400fba1")
    parser.add_argument("--destination", default=HERE, type=Path, help="support directory to build")
    parser.add_argument("--mini-wheel", type=Path, help="local mini-swe-agent 2.4.2 wheel for offline setup")
    parser.add_argument("--claude-archive", type=Path, help="local official Linux ARM64 2.1.325 npm tarball")
    parser.add_argument("--skip-claude", action="store_true", help="build support for the three KV methods only")
    args = parser.parse_args()
    pin = json.loads((HERE / "support-lock.json").read_text())
    root = args.destination.resolve()
    for target, expected in pin["published"].items():
        with (HERE / target).open("rb") as src:
            install(src, root / target, expected)
    restore_files(root, args.repository.resolve(), pin["repository_files"])
    restore_files(root, args.upstream.resolve(), pin["upstream_files"])
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        restore_mini(root, pin["mini_wheel"], args.mini_wheel, tmp)
        if not args.skip_claude:
            restore_claude(root, pin["claude"], args.claude_archive, tmp)
    print(f"Verified client support in {root}")


if __name__ == "__main__":
    main()

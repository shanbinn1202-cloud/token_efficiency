"""Local only (never delivered): keep every temporary file of the perf-mab tools off drive C:.

The user's rule: nothing on C:. Git Bash exports TMPDIR=/tmp and a plain Windows shell has TEMP on
C:\\Users\\...\\AppData\\Local\\Temp; build.py (git scratch clones), the release-parity test (a fresh clone of the
sealed bundle), the tests and the synthetic stages all use `tempfile`. `ensure()` points TEMP, TMP, TMPDIR and
tempfile.tempdir at D:\\Remote-RA-temp\\tmp\\decode-speed2\\harness when the current temporary directory is on C: (or is a POSIX
path such as /tmp that Windows resolves onto C:); it leaves a temporary directory elsewhere (e.g. the local validation's
own -Scratch on D:) unchanged. Linux (the stage) never imports this module.
"""
import os
from pathlib import Path
import tempfile

DEFAULT = r'D:\Remote-RA-temp\tmp\perf-mab'


def on_c(path):
    """Whether `path` resolves onto drive C: on Windows (or is unset)."""
    if not path:
        return True
    try:
        resolved = str(Path(path).resolve())
    except OSError:
        return True
    return resolved[:2].upper() == 'C:' or not resolved[1:2] == ':'


def ensure(default=DEFAULT):
    """On Windows: move TEMP/TMP/TMPDIR and tempfile's directory off C: (to `default`); returns the directory used."""
    if os.name != 'nt':
        return tempfile.gettempdir()
    current = os.environ.get('TEMP') or os.environ.get('TMP') or tempfile.gettempdir()
    if on_c(current) or on_c(os.environ.get('TMPDIR') or current) or on_c(tempfile.gettempdir()):
        if not on_c(current):
            default = current  # TEMP is already off C: (only TMPDIR or tempfile's cache pointed there)
        Path(default).mkdir(parents=True, exist_ok=True)
        for name in ('TEMP', 'TMP', 'TMPDIR'):
            os.environ[name] = default
        tempfile.tempdir = default
    return tempfile.gettempdir()

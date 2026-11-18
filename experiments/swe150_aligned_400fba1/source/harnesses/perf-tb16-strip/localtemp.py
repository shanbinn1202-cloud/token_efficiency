"""Local only (never delivered): keep every temporary file of the perf-tb16 tools off drive C:.

The user's rule: nothing on C:. Git Bash exports TMPDIR=/tmp and a plain Windows shell has TEMP on
C:\\Users\\...\\AppData\\Local\\Temp; build.py (git scratch clones), the verifier (a fresh clone of the sealed bundle for
the selection replay), the tests and the synthetic stages all use `tempfile`. `ensure()` points TEMP, TMP, TMPDIR and
tempfile.tempdir at D:\\Remote-RA-temp\\tmp\\perf-tb16-strip when the current temporary directory is on C: (or is a POSIX
path such as /tmp that Windows resolves onto C:); it leaves a temporary directory elsewhere (e.g. the local validation's
own -Scratch on D:) unchanged. Linux (the stage) never imports this module.
2026-11-12 (perf-swe32's of 2026-11-11: console windows kept flashing while agents ran): `hide_windows()` (also called
by `ensure()`) makes every child process the interpreter starts on Windows run without a console window of its own
(CREATE_NO_WINDOW); a no-op elsewhere. The real submission's visible SSH console (transport.ps1) is PowerShell and
unaffected.
"""
import os
from pathlib import Path
import tempfile

DEFAULT = r'D:\Remote-RA-temp\tmp\perf-tb16-strip'
CREATE_NO_WINDOW = 0x08000000
CHOSEN_CONSOLE = 0x00000008 | 0x00000010  # DETACHED_PROCESS | CREATE_NEW_CONSOLE: the call chose its console


def hide_windows():
    """Windows: subprocess.Popen (and so run / check_output / call) starts a child with CREATE_NO_WINDOW unless the
    call chose DETACHED_PROCESS or CREATE_NEW_CONSOLE; a call that names no stdio gets the parent's fds 0-2 named when
    they are pipes or files (else it shares the parent's console as before: a hidden console of its own would take
    its output). Output, pipes and exit codes are unchanged. Idempotent; returns whether it applies (False off
    Windows)."""
    if os.name != 'nt':
        return False
    import subprocess
    if getattr(subprocess.Popen, 'tb16_no_window', False):
        return True
    original = subprocess.Popen

    def plain_std_fds():
        """fds 0-2 are all open and none is a console or NUL (character devices): pipes or files."""
        try:
            return all(not os.isatty(fd) and os.fstat(fd) is not None for fd in (0, 1, 2))
        except OSError:
            return False

    class Popen(original):
        tb16_no_window = True

        def __init__(self, *args, **kwargs):
            flags = kwargs.get('creationflags') or 0
            if len(args) <= 1 and not flags & CHOSEN_CONSOLE:
                if all(kwargs.get(k) is None for k in ('stdin', 'stdout', 'stderr')):
                    # a child that names no stdio shares the parent's console and so its std handles; a hidden
                    # console of its own needs them named: only pipes / files are passed, else it stays as it was
                    if plain_std_fds():
                        kwargs.update(stdin=0, stdout=1, stderr=2, creationflags=flags | CREATE_NO_WINDOW)
                else:
                    kwargs['creationflags'] = flags | CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)
    subprocess.Popen = Popen
    return True




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
    hide_windows()
    current = os.environ.get('TEMP') or os.environ.get('TMP') or tempfile.gettempdir()
    if on_c(current) or on_c(os.environ.get('TMPDIR') or current) or on_c(tempfile.gettempdir()):
        if not on_c(current):
            default = current  # TEMP is already off C: (only TMPDIR or tempfile's cache pointed there)
        Path(default).mkdir(parents=True, exist_ok=True)
        for name in ('TEMP', 'TMP', 'TMPDIR'):
            os.environ[name] = default
        tempfile.tempdir = default
    return tempfile.gettempdir()

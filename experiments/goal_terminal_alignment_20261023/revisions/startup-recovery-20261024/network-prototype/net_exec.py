"""Prototype: private user/network namespace with rootless outbound networking."""
import os
import select
import subprocess
import sys

def main():
    gate_read, gate_write = os.pipe()
    ready_read, ready_write = os.pipe()
    child = network = None
    try:
        child = subprocess.Popen([
            'unshare', '--user', '--map-root-user', '--net',
            sys.executable, '-c',
            'import os,sys; fd=int(sys.argv[1]); assert os.read(fd,1)==b"1"; os.close(fd); os.execvp(sys.argv[2],sys.argv[2:])',
            str(gate_read), *sys.argv[1:],
        ], pass_fds=(gate_read,))
        os.close(gate_read); gate_read = None
        # Wait for unshare to create its namespace, not merely fork.
        import time
        original = os.readlink('/proc/self/ns/net')
        for _ in range(100):
            if child.poll() is not None:
                raise RuntimeError('Namespace launcher exited')
            if os.readlink(f'/proc/{child.pid}/ns/net') != original:
                break
            time.sleep(.02)
        else:
            raise TimeoutError('Private network namespace not created')
        network = subprocess.Popen([
            'slirp4netns', '--configure', '--disable-host-loopback',
            '--ready-fd', str(ready_write),
            '--userns-path', f'/proc/{child.pid}/ns/user',
            str(child.pid), 'tap0',
        ], pass_fds=(ready_write,))
        os.close(ready_write); ready_write = None
        if not select.select([ready_read], [], [], 20)[0] or os.read(ready_read,1) != b'1':
            raise RuntimeError('Rootless network failed readiness')
        os.write(gate_write,b'1'); os.close(gate_write); gate_write = None
        return child.wait()
    finally:
        for proc in [child,network]:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        for fd in [gate_read,gate_write,ready_read,ready_write]:
            if fd is not None:
                os.close(fd)

if __name__ == '__main__':
    raise SystemExit(main())

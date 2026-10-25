"""Loopback-only TCP relay on the agent host (stdlib only): 127.0.0.1:<port> -> the reverse tunnel's Unix socket.

The GPU job's reverse tunnel ends in a Unix socket on the agent host (`ssh -R <socket>:127.0.0.1:<server port>`).
A Unix-socket forward never opens a TCP listener, so it stays private even when the agent host's sshd has
`GatewayPorts yes` (which binds every TCP remote forward on all interfaces, whatever address the client asks for).
This relay is then the only TCP listener: it binds 127.0.0.1 explicitly and forwards each accepted connection to
the socket (one Unix connection each, both directions pumped until either side closes). The agents reach the model
at http://127.0.0.1:<port>/v1 and nothing off the host can. The multi-session SWE-bench pilots (Slurm jobs 3299922
and 3300434) used this relay unchanged.

Usage: relay.py <socket path> <port>
"""
import socket
import sys
import threading

CHUNK = 1 << 16


def pump(source, target):
    try:
        while True:
            data = source.recv(CHUNK)
            if not data:
                break
            target.sendall(data)
    except OSError:
        pass
    finally:
        try:
            target.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def handle(client, path):
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        upstream.connect(path)
    except OSError as exc:
        print('RELAY_UPSTREAM_ERROR %s: %s' % (type(exc).__name__, exc), flush=True)
        client.close(); upstream.close()
        return
    back = threading.Thread(target=pump, args=(upstream, client), daemon=True)
    back.start()
    pump(client, upstream)
    back.join()
    client.close(); upstream.close()


def main(path, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', port))
    server.listen(64)
    print('RELAY_LISTENING 127.0.0.1:%d -> %s' % (port, path), flush=True)
    while True:
        client, _ = server.accept()
        threading.Thread(target=handle, args=(client, path), daemon=True).start()


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))

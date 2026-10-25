"""Real stdlib spawn driver for CPU protocol gates, NOT a vLLM launcher.

Only fixed module-level engine_entry is spawned. Scenario controls are synthetic
fault injection and must never become production configuration. No key, epoch,
challenge, callback/factory or authority object is passed in spawn arguments.
"""
from dataclasses import dataclass
import multiprocessing as mp
import os
import queue
import secrets
import threading
import time

from .protocol import (AuthorityError, Capability, Channel, EngineFence, MAX_FRAME,
    Topology, child_provision, digest, hex64, parent_provision, require, send_raw)
from .evidence import PINS, verify_sources

CPU_MANIFEST = digest({"kind": "CPU_SYNTHETIC_ONLY", "protocol": 1,
                       "source_pins": PINS})


@dataclass(frozen=True)
class Scenario:
    mode: str = "normal"
    hello_delay: float = 0.0
    prepared_delay: float = 0.0
    ack_delay: float = 0.0
    lease: float = 2.0

    def validate(self):
        require(type(self) is Scenario and self.mode in {"normal", "never_reads", "partial_json",
            "oversized", "exit_before_b0", "exit_after_b0", "exit_after_f1", "lost_f1_ack",
            "blocked_worker", "partial_transport", "long_owner"}, "unknown CPU fault scenario")
        require(all(type(x) in (int, float) and 0 <= x <= 5 for x in (
            self.hello_delay, self.prepared_delay, self.ack_delay)), "invalid synthetic delay")
        require(type(self.lease) in (int, float) and 0.2 <= self.lease <= 10, "invalid lease")


def _same(actual, expected):
    require(actual == expected, "phase receipt/nonce differs")


def engine_entry(pipe, telemetry, native_init, scenario):
    """One CPU engine child; every fence mutation occurs on this owner thread."""
    scenario.validate()
    parent = mp.parent_process()
    require(parent is not None and mp.get_start_method() == "spawn", "actual spawn parent required")
    stop = threading.Event()
    revoke = threading.Event()
    fence = EngineFence(revoke)
    deadline = [time.monotonic() + scenario.lease]
    report = {"kind": "CPU_SYNTHETIC_ONLY", "child_pid": os.getpid(), "parent_pid": parent.pid,
              "start_method": mp.get_start_method(), "ordinary_before_f1_denied": False,
              "registered_before_f1_denied": False, "failure": None}

    def watch_parent():
        while not stop.wait(0.02):
            if not parent.is_alive() or time.monotonic() >= deadline[0]:
                revoke.set()  # no scheduler/provider/controller calls here
                # The owner gets a bounded chance to observe revocation and clean
                # CPU state. A stuck transport/worker gets process-level failure.
                if not stop.wait(0.5):
                    os._exit(73)
                return

    monitor = threading.Thread(target=watch_parent, name="cpu-engine-peer-monitor")
    monitor.start()
    try:
        if scenario.mode == "never_reads":
            revoke.wait(5)
            raise AuthorityError("synthetic blocked peer")
        time.sleep(scenario.hello_delay)
        channel = child_provision(pipe, parent_pid=parent.pid, manifest=CPU_MANIFEST,
                                  deadline=deadline[0])
        if scenario.mode == "partial_json":
            pipe.send_bytes(b'{"version":1')
            return
        if scenario.mode == "oversized":
            pipe.send_bytes(b"x" * (MAX_FRAME + 1))
            return
        if scenario.mode == "partial_transport":
            # POSIX byte-stream Connection: a declared frame with no body. On
            # Windows message pipes, test the equivalent peer loss during write
            # separately; do not claim this branch is portable wire evidence.
            require(os.name != "nt", "partial byte-stream fault requires POSIX")
            import struct
            os.write(pipe.fileno(), struct.pack("!i", 100))
            revoke.wait(5)  # declared frame body never arrives; watchdog must unblock it
            return
        # Native HELLO/INIT is independent of sideband PREPARED. Tests control
        # INIT from the parent to expose waiting-inside-yield deadlock.
        fence.events.append("HELLO")
        require(native_init.wait(max(0, deadline[0] - time.monotonic())), "native INIT deadline")
        fence.check()
        fence.events.append("INIT")
        if scenario.mode == "blocked_worker":
            revoke.wait(5)
            raise AuthorityError("synthetic worker RPC timeout")
        time.sleep(scenario.prepared_delay)
        fence.check()
        if scenario.mode == "exit_before_b0":
            os._exit(71)
        # Explicitly synthetic; the real seam must use capability_from_worker_gate.
        capability = Capability(secrets.token_hex(32), digest({"cpu_worker": os.getpid()}))
        prepared = capability.payload()
        channel.send("PREPARED", prepared)
        fence.events.append("WORKER_CAPABILITY_ACK")
        receipt = {"receipt_hash": digest(prepared)}
        _, payload = channel.receive({"COMMIT"}, deadline[0])
        _same(payload, receipt)
        time.sleep(scenario.ack_delay)
        fence.check()
        channel.send("COMMIT_ACK", receipt)
        _, payload = channel.receive({"RELEASE"}, deadline[0])
        _same(payload, receipt)
        fence.capable()
        fence.native_ready("INPUT_READY")
        fence.native_ready("HANDSHAKE_READY")
        channel.send("RELEASE_ACK", receipt)
        channel.handoff_to_monitor()
        deadline[0] = time.monotonic() + scenario.lease
        for kind in ("ordinary", "registered"):
            try:
                fence.require_add(kind)
            except AuthorityError:
                report[kind + "_before_f1_denied"] = True
        if scenario.mode == "exit_after_b0":
            os._exit(72)
        bound = False
        while True:
            kind, payload = channel.receive({"PING", "FRONTEND_BOUND", "CLOSE"}, deadline[0])
            fence.check()
            deadline[0] = time.monotonic() + scenario.lease
            if kind == "CLOSE":
                fence.revoke_and_cleanup()
                channel.send("CLOSED", {})
                break
            if kind == "PING":
                channel.send("PONG", payload)
                continue
            require(not bound and payload["worker_nonce"] == capability.worker_nonce,
                    "repeated F1 or changed worker nonce")
            bound = True
            fence.bind_frontend()
            if scenario.mode == "exit_after_f1":
                os._exit(74)
            if scenario.mode == "lost_f1_ack":
                # Partial release: engine opened, frontend MUST remain fenced.
                # No response; parent deadline/child lease permanently revoke.
                revoke.wait(5)
                fence.check()
            channel.send("FRONTEND_ACK", payload)
            if scenario.mode == "long_owner":
                # Simulates an owner unable to dispatch while doing work. The
                # observer may fence, then process-level timeout ends this CPU
                # child; no claim about interrupting real GPU kernels follows.
                time.sleep(5)
                fence.check()
            fence.require_add("ordinary")
            fence.require_add("registered")
            fence.events.append("CPU_ADMISSION_ONLY_NO_PUBLICATION")
    except BaseException as exc:
        report["failure"] = type(exc).__name__  # do not echo messages/credentials
    finally:
        fence.revoke_and_cleanup()
        report["events"] = list(fence.events)
        report["revoked"] = fence.state == "REVOKED"
        report["owner_thread_cleanup"] = True
        stop.set()
        monitor.join(0.5)
        pipe.close()
        try:
            send_raw(telemetry, report)
        except (BrokenPipeError, EOFError, OSError):
            pass
        telemetry.close()


class CpuAuthorityLaunch:
    """Parent/frontend simulator. NOT_INSTALLABLE and no request success API."""
    def __init__(self, *, topology=None, context=None, scenario=None, timeout=3.0):
        context = context or mp.get_context("spawn")
        topology = topology or Topology()
        topology.validate(context)  # before key, pipe, process, thread creation
        scenario = scenario or Scenario()
        scenario.validate()
        require(type(timeout) in (int, float) and 0.2 <= timeout <= 10, "invalid launch deadline")
        self.context, self.scenario, self.timeout = context, scenario, timeout
        self._owner = threading.get_ident()
        self._lock = threading.RLock()
        self._commands = queue.Queue(maxsize=1)
        self._b0 = threading.Event()
        self._f1 = threading.Event()
        self._done = threading.Event()
        self._revoked = threading.Event()
        self._error = None
        self._channel = None
        self._deadline = time.monotonic() + timeout
        self._started = False
        self._closed = False
        self._process = self._sideband = self._watchdog = self._pipe = self._telemetry = None
        self.identity = None
        self.reader_phases = []
        self.report = None

    def _check_owner(self):
        require(threading.get_ident() == self._owner, "frontend owner thread changed")

    def start(self):
        self._check_owner()
        require(not self._started and not self._closed, "launch is single use")
        verify_sources()  # read-only pins; still no loaded identity claim
        self._started = True
        parent_pipe = child_pipe = receive_report = send_report = None
        try:
            parent_pipe, child_pipe = self.context.Pipe(duplex=True)
            receive_report, send_report = self.context.Pipe(duplex=False)
            self.native_init = self.context.Event()
            process = self.context.Process(target=engine_entry,
                args=(child_pipe, send_report, self.native_init, self.scenario),
                name="cpu-private-engine")
            require(process._start_method == "spawn", "Process context differs")
            self._process, self._pipe, self._telemetry = process, parent_pipe, receive_report
            process.start()
            child_pipe.close()
            send_report.close()
            self._deadline = time.monotonic() + self.timeout
            self._sideband = threading.Thread(target=self._run, name="cpu-parent-sideband")
            self._watchdog = threading.Thread(target=self._watch, name="cpu-parent-watchdog")
            self._sideband.start()
            self._watchdog.start()
            return self
        except BaseException:
            self._fence("launch construction failed")
            if self._process is not None and self._process.pid is not None:
                self._reap()
            # Reaping releases peer I/O; closing also wakes a pending local
            # Windows poll. Serialize against the reader's finally close.
            self._close_transport()
            for pipe in (child_pipe, receive_report, send_report):
                if pipe is not None:
                    pipe.close()
            self._done.set()
            for thread in (self._sideband, self._watchdog):
                if thread is not None and thread.ident is not None:
                    thread.join(1.0)
                    require(not thread.is_alive(), "failed-launch thread cleanup incomplete")
            if self._process is not None and self._process.pid is not None:
                self._process.close()
            self._closed = True
            raise

    def advance_native_init(self):
        self._check_owner()
        require(self._started and not self._revoked.is_set(), "launch unavailable")
        self.native_init.set()

    def _fence(self, reason):
        with self._lock:
            self._revoked.set()
            if self._error is None:
                self._error = reason

    def _progress(self):
        with self._lock:
            self._deadline = time.monotonic() + self.timeout

    def _watch(self):
        while not self._done.wait(0.01):
            if self._revoked.is_set():
                break
            if not self._process.is_alive():
                self._fence("engine peer died")
                break
            if time.monotonic() >= self._deadline:
                self._fence("absolute phase or I/O deadline expired")
                break
        if self._revoked.is_set() and self._error is not None:
            self._reap()
            # Termination closes the peer endpoint and unblocks partial receive
            # or blocked send before joining our sole reader.
            self._sideband.join(1.0)
            if self._sideband.is_alive():
                self._close_transport()
                self._sideband.join(1.0)
            if self._sideband.is_alive():
                self._fence("transport thread failed to stop")

    def _run(self):
        try:
            self.reader_phases.append("STARTUP")
            channel = parent_provision(self._pipe, child_pid=self._process.pid,
                manifest=CPU_MANIFEST, deadline=self._deadline)
            self._channel = channel
            self.identity = channel.identity
            _, prepared = channel.receive({"PREPARED"}, self._deadline)
            receipt = {"receipt_hash": digest(prepared)}
            channel.send("COMMIT", receipt)
            _, payload = channel.receive({"COMMIT_ACK"}, self._deadline)
            _same(payload, receipt)
            channel.send("RELEASE", receipt)
            _, payload = channel.receive({"RELEASE_ACK"}, self._deadline)
            _same(payload, receipt)
            require(not self._revoked.is_set(), "B0 after revoke")
            self._b0.set()
            channel.handoff_to_monitor()
            self.reader_phases.append("MONITOR")
            self._progress()
            f1_deadline = time.monotonic() + self.timeout
            bound = False
            while not self._revoked.is_set():
                try:
                    command, binding_hash = self._commands.get(timeout=0.05)
                except queue.Empty:
                    command, binding_hash = "PING", None
                if command == "CLOSE":
                    channel.send("CLOSE", {})
                    channel.receive({"CLOSED"}, self._deadline)
                    break
                if command == "BIND":
                    require(not bound and time.monotonic() < f1_deadline, "F1 repeated or expired")
                    payload = dict(frontend_nonce=secrets.token_hex(32), binding_hash=binding_hash,
                                   worker_nonce=prepared["worker_nonce"])
                    channel.send("FRONTEND_BOUND", payload)
                    _, returned = channel.receive({"FRONTEND_ACK"}, self._deadline)
                    _same(returned, payload)
                    require(not self._revoked.is_set(), "F1 ACK after revoke")
                    bound = True
                    self._f1.set()
                else:
                    require(bound or time.monotonic() < f1_deadline, "frontend binding deadline expired")
                    payload = {"ping_nonce": secrets.token_hex(32)}
                    channel.send("PING", payload)
                    _, returned = channel.receive({"PONG"}, self._deadline)
                    _same(returned, payload)
                self._progress()
        except BaseException:
            self._fence("private protocol failed")
        finally:
            self._revoked.set()
            try:
                self._close_transport()
            finally:
                self._done.set()

    def _close_transport(self):
        # The sole reader and watchdog may finish concurrently. Serialize close
        # itself, not merely a preceding closed check.
        with self._lock:
            if self._pipe is not None and not self._pipe.closed:
                self._pipe.close()

    def wait_b0(self):
        self._wait(self._b0)

    def _wait(self, event):
        self._check_owner()
        while not event.wait(0.01):
            require(not self._revoked.is_set(), self._error or "authority revoked")
        require(not self._revoked.is_set(), self._error or "authority revoked")

    def bind_frontend(self, binding_hash):
        self._check_owner()
        require(hex64(binding_hash) and self._b0.is_set() and not self._f1.is_set()
                and not self._revoked.is_set(), "frontend binding unavailable")
        self._commands.put_nowait(("BIND", binding_hash))
        self._wait(self._f1)

    def require_frontend_admission(self):
        self._check_owner()
        require(self._f1.is_set() and not self._revoked.is_set(), "frontend F1 ACK absent/revoked")

    def cancel(self):
        self._check_owner()
        self._fence("frontend cancelled")

    def _reap(self):
        # Serialize concurrent watchdog/close cleanup. No process-tree/global kill.
        with self._lock:
            process = self._process
            if process is None or process.pid is None:
                return
            if process.is_alive():
                process.terminate()
            process.join(1.0)
            if process.is_alive():
                process.kill()
                process.join(1.0)
            require(not process.is_alive(), "engine process did not terminate")

    def close(self):
        self._check_owner()
        if self._closed:
            return
        if self._started and self._sideband is not None:
            if self._b0.is_set() and not self._revoked.is_set():
                try:
                    self._commands.put_nowait(("CLOSE", None))
                except queue.Full:
                    self._fence("close raced pending binding")
            else:
                self._fence("close before B0")
            self._done.wait(self.timeout + 0.1)
            if self._done.is_set() and self._error is None:
                self._process.join(0.7)
            self._reap()
            self._sideband.join(1.0)
            self._done.set()
            self._watchdog.join(3.0)
            require(not self._sideband.is_alive() and not self._watchdog.is_alive(),
                    "authority thread cleanup incomplete")
            try:
                from .protocol import receive_raw
                self.report = receive_raw(self._telemetry, time.monotonic() + 0.2)
            except AuthorityError:
                self.report = None
            self._close_transport()
            self._telemetry.close()
            self._process.close()
        self._revoked.set()
        self._closed = True

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

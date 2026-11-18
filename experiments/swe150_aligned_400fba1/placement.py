"""Keep vLLM and local testbeds on separate physical cores in one Slurm allocation."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import time

CORES_PER_GROUP = 16
SYS_CPU = Path("/sys/devices/system/cpu")


def cpu_numbers(spec):
    numbers = set()
    for group in spec.strip().split(","):
        if not group:
            continue
        lo, sep, hi = group.partition("-")
        numbers.update(range(int(lo), int(hi if sep else lo) + 1))
    return numbers


def select_cpus(allowed, root=SYS_CPU):
    representatives = {}
    for cpu in sorted(allowed):
        group = tuple(sorted(cpu_numbers((root / ("cpu%d" % cpu) / "topology/thread_siblings_list").read_text())))
        if not group or cpu not in group:
            raise RuntimeError("Invalid hardware sibling information for CPU %s" % cpu)
        representatives.setdefault(group, cpu)
    if len(representatives) != CORES_PER_GROUP:
        raise RuntimeError("Expected a Slurm step with 16 physical cores; found %d among %s" %
                           (len(representatives), sorted(allowed)))
    physical = [",".join(map(str, group)) for group in representatives]
    return list(representatives.values()), physical


def gpu_locality():
    """Query CUDA-visible ordinal 0, which need not match host nvidia-smi ordinal 0."""
    record = {"pci_address": None, "numa_node": None, "local_cpus": []}
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        device = ctypes.c_int()
        bus_id = ctypes.create_string_buffer(32)
        if (lib.cuInit(0) or lib.cuDeviceGet(ctypes.byref(device), 0)
                or lib.cuDeviceGetPCIBusId(bus_id, len(bus_id), device)):
            raise ValueError("CUDA driver could not identify visible device 0")
        domain, bus, slot = bus_id.value.decode("ascii").lower().split(":")
        address = "%04x:%s:%s" % (int(domain, 16), bus, slot)
        path = Path("/sys/bus/pci/devices") / address
        node = int((path / "numa_node").read_text())
        record.update(pci_address=address, numa_node=node)
        locality = (Path("/sys/devices/system/node/node%d" % node) / "cpulist"
                    if node >= 0 else path / "local_cpulist")
        record["local_cpus"] = sorted(cpu_numbers(locality.read_text()))
    except (OSError, ValueError, AttributeError) as error:
        record["unavailable_reason"] = str(error)
    return record


def nvme_leaves(device):
    lower = list((device / "slaves").glob("*"))
    if lower:
        return sorted({leaf for block in lower for leaf in nvme_leaves(block.resolve())})
    match = re.fullmatch(r"(nvme[0-9]+)n[0-9]+(?:p[0-9]+)?", device.name)
    if not match:
        raise RuntimeError("Scratch filesystem does not resolve to a local NVMe drive: %s" % device)
    transport = (Path("/sys/class/nvme") / match[1] / "transport").read_text().strip()
    if transport != "pcie":
        raise RuntimeError("Scratch uses remote NVMe transport %s" % transport)
    return [device.name]


def scratch_info(requested):
    path = Path(requested).resolve(strict=True)
    tmp_root = Path("/tmp").resolve(strict=True)
    if not path.is_dir() or (path != tmp_root and tmp_root not in path.parents):
        raise RuntimeError("Sandbox scratch must be a directory on node-local /tmp: %s" % path)
    dev = path.stat().st_dev
    major_minor = "%d:%d" % (os.major(dev), os.minor(dev))
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields = before.split()
        if fields[2] != major_minor:
            continue
        target = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
        if target == path or target in path.parents:
            fs, source = after.split()[:2]
            mounts.append((len(target.parts), int(fields[0]), fs, source, str(target)))
    if not mounts:
        raise RuntimeError("Cannot identify scratch mount: %s" % path)
    _, _, fs, source, target = max(mounts)
    if fs not in {"xfs", "ext4", "btrfs"}:
        raise RuntimeError("Scratch must be a disk-backed local filesystem; %s is %s" % (path, fs))
    device = (Path("/sys/dev/block") / major_minor).resolve(strict=True)
    return {"path": str(path), "mount": target, "filesystem": fs, "source": source,
            "nvme_devices": nvme_leaves(device), "available_bytes": shutil.disk_usage(path).free}


def memory_limits():
    groups = {}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, location = line.split(":", 2)
        if not controllers:
            groups["cgroup2"] = Path(location)
        elif "memory" in controllers.split(","):
            groups["cgroup"] = Path(location)
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        kind, _, options = right.split()[:3]
        if kind in groups and (kind == "cgroup2" or "memory" in options.split(",")):
            fields = left.split()
            mounts.append((kind, Path(fields[3]), Path(fields[4])))
    # On hybrid systems the v1 hierarchy controls memory if it is mounted.
    for kind, root, mount in sorted(mounts):
        location = groups[kind]
        try:
            here = mount / location.relative_to(root)
        except ValueError:
            continue
        if not here.is_dir():
            continue
        version = 2 if kind == "cgroup2" else 1
        filename = "memory.max" if version == 2 else "memory.limit_in_bytes"
        results = []
        while here == mount or mount in here.parents:
            setting = here / filename
            if setting.exists():
                value = setting.read_text().strip()
                finite = None if value == "max" or int(value) >= 2**60 else int(value)
                results.append({"group": str(root / here.relative_to(mount)), "max_bytes": finite})
            if here == mount:
                break
            here = here.parent
        if not results:
            continue
        def limit(prefix):
            for record in results:
                if re.match(prefix + r"[_-](?:[0-9]+|batch|extern|interactive)(?:$|[._-])",
                            Path(record["group"]).name):
                    return record["max_bytes"]
            return None
        bounds = [item["max_bytes"] for item in results if item["max_bytes"] is not None]
        return {"version": version, "step_limit_bytes": limit("step"), "job_limit_bytes": limit("job"),
                "effective_limit_bytes": min(bounds) if bounds else None, "limits": results}
    return {"version": None, "step_limit_bytes": None, "job_limit_bytes": None,
            "effective_limit_bytes": None, "limits": []}


def step_receipt(role, scratch, peer):
    job_id, step_id = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
    if not job_id or not step_id or step_id in {"batch", "extern"}:
        raise RuntimeError("Run inside the explicit 16-core srun step")
    before = sorted(os.sched_getaffinity(0))
    cpus, cores = select_cpus(before)
    os.sched_setaffinity(0, cpus)
    if sorted(os.sched_getaffinity(0)) != cpus:
        raise RuntimeError("Failed to apply verified step CPU affinity")
    if role != "engine":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        gpu = gpu_locality()
    scratch_record = scratch_info(scratch)
    temp_dir = scratch_info(tempfile.gettempdir())
    if (temp_dir["source"], temp_dir["nvme_devices"]) != (scratch_record["source"], scratch_record["nvme_devices"]):
        raise RuntimeError("TMPDIR and declared scratch are on different devices")
    memory = memory_limits()
    expected_mib = 113560 if role == "engine" else 30720
    effective = memory["effective_limit_bytes"]
    # Slurm RAM fractions and page rounding vary. Disallow ceilings materially
    # below the requested share and expose job-only or unlimited setups explicitly.
    if effective is not None and effective < 0.97 * expected_mib * 1024**2:
        raise RuntimeError("%s step has only %d bytes; expected approximately %d MiB" %
                           (role, effective, expected_mib))
    job_limit = memory["job_limit_bytes"]
    if job_limit is not None and job_limit < 0.97 * (113560 + 30720) * 1024**2:
        raise RuntimeError("Job memory cgroup cannot cover both shares: %d bytes" % job_limit)
    step_limit = memory["step_limit_bytes"]
    if step_limit is None:
        memory["enforcement"] = "job_only" if job_limit is not None else "no_identifiable_step_or_job_cap"
    elif abs(step_limit - expected_mib * 1024**2) <= 0.03 * expected_mib * 1024**2:
        memory["enforcement"] = "step_cap_matches_request"
    elif job_limit is not None and step_limit >= job_limit:
        memory["enforcement"] = "job_only"
    else:
        memory["enforcement"] = "step_cap_differs_from_request"
    record = {"role": role, "hostname": socket.gethostname(), "job_id": job_id, "step_id": step_id,
              "affinity_before": before, "cpus": cpus, "physical_cores": cores, "scratch": scratch_record,
              "tmpdir": temp_dir, "cgroup": memory, "expected_step_memory_mib": expected_mib,
              "time": time.time()}
    if role == "engine":
        record["gpu"] = gpu
    if role in {"agent", "probe"} and not peer:
        raise RuntimeError("A concurrent client step must check the service step receipt")
    if peer:
        other = json.loads(Path(peer).read_text())
        if (other["role"] != "engine" or other["job_id"] != job_id or other["hostname"] != record["hostname"]
                or other["step_id"] == step_id or len(set(other["physical_cores"])) != CORES_PER_GROUP
                or set(other["physical_cores"]) & set(cores)):
            raise RuntimeError("Service and client steps do not occupy disjoint physical cores in the same allocation")
        record["peer_step_id"] = other["step_id"]
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("engine", "agent", "grade", "probe"), required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--peer", type=Path)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if len(args.cmd) < 2 or args.cmd[0] != "--":
        p.error("Expected -- followed by the program to run")
    record = step_receipt(args.role, args.scratch, args.peer)
    tmp = args.receipt.with_name(args.receipt.name + ".tmp.%d" % os.getpid())
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(args.receipt)
    os.execvp(args.cmd[1], args.cmd[1:])


if __name__ == "__main__":
    main()

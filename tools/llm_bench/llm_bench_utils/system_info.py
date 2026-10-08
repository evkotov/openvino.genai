# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Software and machine state recorded in the reports, so that results from different runs can be compared."""

import glob
import hashlib
import importlib.metadata
import os
import platform
import sys
from collections import Counter

ENV_PREFIXES = ("OV_", "OPENVINO_", "GGML_", "LLAMA_", "OMP_", "KMP_", "GOMP_", "TBB_", "ONEDNN_", "DNNL_")
# module name -> distribution name
PACKAGES = {"openvino": "openvino", "openvino_genai": "openvino-genai", "openvino_tokenizers": "openvino-tokenizers",
            "torch": "torch", "transformers": "transformers", "optimum": "optimum-intel", "llama_cpp": "llama_cpp_python"}


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def parse_cpu_list(text):
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]"""
    cpus = []
    for part in (text or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-")
            cpus.extend(range(int(first), int(last) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def format_cpu_list(cpus):
    """[0, 1, 2, 3, 8] -> '0-3,8'"""
    cpus = sorted(set(cpus))
    ranges = []
    for cpu in cpus:
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)


def _histogram(values):
    counts = Counter(v for v in values if v is not None)
    return dict(counts) if len(counts) > 1 else (next(iter(counts)) if counts else None)


def cpu_topology(sys_root="/sys"):
    cpu_dir = os.path.join(sys_root, "devices/system/cpu")
    online = parse_cpu_list(read_text(os.path.join(cpu_dir, "online")))
    topo = {"online": format_cpu_list(online), "logical_cpus": len(online)}
    # hybrid Intel CPUs expose the core types as separate PMUs
    core_types = {}
    for name, label in (("cpu_core", "performance"), ("cpu_atom", "efficient")):
        cpus = read_text(os.path.join(sys_root, "devices", name, "cpus"))
        if cpus:
            core_types[label] = cpus
    if core_types:
        topo["core_types"] = core_types
    siblings = set()
    for cpu in online:
        text = read_text(os.path.join(cpu_dir, f"cpu{cpu}/topology/thread_siblings_list"))
        if text:
            siblings.add(text)
    if siblings:
        topo["physical_cores"] = len(siblings)
        topo["smt_active"] = any(len(parse_cpu_list(s)) > 1 for s in siblings)
    return topo


def cpu_power(sys_root="/sys"):
    cpu_dir = os.path.join(sys_root, "devices/system/cpu")
    cpus = sorted(glob.glob(os.path.join(cpu_dir, "cpu[0-9]*")), key=lambda p: int(p.rsplit("cpu", 1)[1]))
    freq = [os.path.join(c, "cpufreq") for c in cpus if os.path.isdir(os.path.join(c, "cpufreq"))]
    state = {
        "scaling_driver": _histogram(read_text(os.path.join(f, "scaling_driver")) for f in freq),
        "governor": _histogram(read_text(os.path.join(f, "scaling_governor")) for f in freq),
        "energy_performance_preference": _histogram(read_text(os.path.join(f, "energy_performance_preference")) for f in freq),
        "scaling_min_khz": _histogram(read_text(os.path.join(f, "scaling_min_freq")) for f in freq),
        "scaling_max_khz": _histogram(read_text(os.path.join(f, "scaling_max_freq")) for f in freq),
        "cpuinfo_max_khz": _histogram(read_text(os.path.join(f, "cpuinfo_max_freq")) for f in freq),
    }
    no_turbo = read_text(os.path.join(cpu_dir, "intel_pstate/no_turbo"))
    boost = read_text(os.path.join(cpu_dir, "cpufreq/boost"))
    if no_turbo is not None:
        state["turbo"] = no_turbo == "0"
    elif boost is not None:
        state["turbo"] = boost == "1"
    disabled = {}
    for c in cpus:
        names = [read_text(os.path.join(s, "name")) for s in sorted(glob.glob(os.path.join(c, "cpuidle/state*")))
                 if read_text(os.path.join(s, "disable")) == "1"]
        if names:
            disabled[os.path.basename(c)[3:]] = ",".join(names)
    state["cstates_disabled"] = {format_cpu_list(int(c) for c in cpus_): names for names, cpus_ in _group(disabled).items()}
    return {k: v for k, v in state.items() if v not in (None, {})}


def _group(per_cpu):
    groups = {}
    for cpu, value in per_cpu.items():
        groups.setdefault(value, []).append(cpu)
    return groups


def throttle_counters(sys_root="/sys"):
    """Sum of the package and core thermal throttle event counters (Intel)."""
    total = {"core": 0, "package": 0}
    found = False
    for path in glob.glob(os.path.join(sys_root, "devices/system/cpu/cpu[0-9]*/thermal_throttle")):
        for kind, name in (("core", "core_throttle_count"), ("package", "package_throttle_count")):
            value = read_text(os.path.join(path, name))
            if value is not None and value.isdigit():
                total[kind] += int(value)
                found = True
    return total if found else None


def memory_settings(sys_root="/sys", proc_root="/proc"):
    def selected(text):
        if text and "[" in text:
            return text.split("[", 1)[1].split("]", 1)[0]
        return text

    meminfo = {}
    for line in (read_text(os.path.join(proc_root, "meminfo")) or "").splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "SwapTotal"):
            meminfo[key + "_kb"] = int(value.split()[0])
    return {
        "transparent_hugepage": selected(read_text(os.path.join(sys_root, "kernel/mm/transparent_hugepage/enabled"))),
        "numa_balancing": read_text(os.path.join(proc_root, "sys/kernel/numa_balancing")),
        **meminfo,
    }


def cpu_model(proc_root="/proc"):
    for line in (read_text(os.path.join(proc_root, "cpuinfo")) or "").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def os_release(root="/etc/os-release"):
    for line in (read_text(root) or "").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return None


def package_versions():
    """Versions of the loaded modules (they may come from a local build), else of the installed distributions."""
    versions = {}
    for name, dist in PACKAGES.items():
        module = sys.modules.get(name)
        version = getattr(module, "__version__", None) if module is not None else None
        if version is None and module is not None and hasattr(module, "get_version"):
            version = module.get_version()
        if version is None:
            try:
                version = importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                continue
        versions[name] = str(version)
    llama_cpp = sys.modules.get("llama_cpp")
    if llama_cpp is not None:
        lib = getattr(getattr(llama_cpp, "llama_cpp", None), "_lib", None)
        if lib is not None and getattr(lib, "_name", None):
            versions["llama_cpp_library"] = lib._name
    return versions


def env_vars():
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}


def busy_thread_cpus(pid=None, proc_root="/proc", share=0.1):
    """CPUs that the busy threads of a process last ran on; a thread is busy with >= share of the busiest CPU time."""
    pid = pid or os.getpid()
    threads = []
    for stat_path in glob.glob(os.path.join(proc_root, str(pid), "task/*/stat")):
        text = read_text(stat_path)
        if not text:
            continue
        fields = text.rsplit(")", 1)[1].split()
        # fields after the command: state is fields[0], utime = 14th, stime = 15th, processor = 39th field of stat
        threads.append((int(fields[11]) + int(fields[12]), int(fields[36])))
    if not threads:
        return None
    busiest = max(t for t, _ in threads)
    if busiest == 0:
        return None
    return format_cpu_list(cpu for t, cpu in threads if t >= share * busiest)


def gguf_info(path):
    try:
        from gguf import GGUFReader, GGUFValueType
    except Exception:
        return None
    try:
        reader = GGUFReader(path)
    except Exception:
        return None

    def field(name):
        f = reader.fields.get(name)
        if f is None or not f.data:
            return None
        value = f.parts[f.data[0]]
        if f.types and f.types[0] == GGUFValueType.STRING:
            return bytes(value).decode("utf-8", "replace")
        return value.tolist()[0]

    sizes = Counter()
    for tensor in reader.tensors:
        sizes[tensor.tensor_type.name] += int(tensor.n_bytes)
    total = sum(sizes.values()) or 1
    return {
        "architecture": field("general.architecture"),
        "file_type": field("general.file_type"),
        "tensor_types": {name: round(size / total, 4) for name, size in sizes.most_common()},
    }


def file_sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def model_info(model_path, with_hash=False):
    path = str(model_path)
    info = {"path": os.path.abspath(path)}
    if os.path.isfile(path):
        info["size_bytes"] = os.path.getsize(path)
        if with_hash:
            info["sha256"] = file_sha256(path)
        if path.endswith(".gguf"):
            gguf = gguf_info(path)
            if gguf:
                info["gguf"] = gguf
    return info


def host_info(sys_root="/sys", proc_root="/proc"):
    info = {
        "hostname": platform.node(),
        "cpu_model": cpu_model(proc_root),
        "os": os_release() or platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
    }
    if hasattr(os, "confstr"):
        try:
            info["libc"] = os.confstr("CS_GNU_LIBC_VERSION")
        except (ValueError, OSError):
            pass
    if hasattr(os, "sched_getaffinity"):
        info["process_affinity"] = format_cpu_list(os.sched_getaffinity(0))
    if sys.platform.startswith("linux"):
        info["topology"] = cpu_topology(sys_root)
        info["power"] = cpu_power(sys_root)
        info["memory"] = memory_settings(sys_root, proc_root)
    return {k: v for k, v in info.items() if v not in (None, {}, "")}


def loadavg(proc_root="/proc"):
    text = read_text(os.path.join(proc_root, "loadavg"))
    return [float(x) for x in text.split()[:3]] if text else None


class EnvironmentRecorder:
    """Snapshot at the start of the benchmark and the changes until the end."""

    def __init__(self, sys_root="/sys", proc_root="/proc"):
        self.sys_root = sys_root
        self.proc_root = proc_root
        self.start = {"loadavg": loadavg(proc_root), "throttle": throttle_counters(sys_root)}

    def collect(self, model_path=None, with_hash=False, argv=None):
        throttle_end = throttle_counters(self.sys_root)
        run = {"loadavg_start": self.start["loadavg"], "loadavg_end": loadavg(self.proc_root),
               "busy_thread_cpus": busy_thread_cpus(proc_root=self.proc_root)}
        if self.start["throttle"] is not None and throttle_end is not None:
            run["throttle_events"] = {k: throttle_end[k] - self.start["throttle"][k] for k in throttle_end}
        env = {
            "command_line": list(sys.argv if argv is None else argv),
            "packages": package_versions(),
            "env": env_vars(),
            "host": host_info(self.sys_root, self.proc_root),
            "run": {k: v for k, v in run.items() if v is not None},
        }
        if model_path is not None:
            env["model"] = model_info(model_path, with_hash)
        return env

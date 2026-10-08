# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU affinity of the run and the run_info.json record of versions and settings."""

import json
import logging
import os
import platform
import sys

logger = logging.getLogger(__name__)

APPLIED_ENV = "WWB_CPU_AFFINITY_APPLIED"
ENV_PREFIXES = ("OV_", "OPENVINO_", "GGML_", "LLAMA_", "OMP_", "KMP_", "GOMP_", "TBB_")


def parse_cpu_spec(spec):
    """'1,3,5-9:2' -> [1, 3, 5, 7, 9]; a range may have a step after ':'."""
    cpus = set()
    for item in str(spec).replace(" ", "").split(","):
        if not item:
            continue
        try:
            rng, _, step = item.partition(":")
            first, _, last = rng.partition("-")
            first, last, step = int(first), int(last or first), int(step or 1)
        except ValueError as e:
            raise ValueError(f"invalid CPU list item '{item}'") from e
        if first < 0 or last < first or step < 1:
            raise ValueError(f"invalid CPU list item '{item}'")
        cpus.update(range(first, last + 1, step))
    if not cpus:
        raise ValueError(f"empty CPU list '{spec}'")
    return sorted(cpus)


def apply_cpu_affinity(spec, use_llamacpp=False):
    """Runs the process on the given CPUs. Libraries already started threads when wwb parses its arguments,
    and those keep the old mask, so the process is restarted with the new mask, which all threads inherit."""
    if spec is None or os.environ.get(APPLIED_ENV) == spec:
        return
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("--cpu-affinity is supported only on Linux")
    cpus = parse_cpu_spec(spec)
    os.sched_setaffinity(0, cpus)
    os.environ[APPLIED_ENV] = spec
    # llama.cpp CPU builds use OpenMP: bind its thread i to the i-th CPU
    if use_llamacpp and "GOMP_CPU_AFFINITY" not in os.environ:
        os.environ["GOMP_CPU_AFFINITY"] = " ".join(str(cpu) for cpu in cpus)
    logger.info("Restarting on CPUs %s", spec)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, sys.orig_argv)


def _package_versions():
    versions = {}
    for name in ("openvino", "openvino_genai", "openvino_tokenizers", "torch", "transformers", "optimum", "llama_cpp"):
        module = sys.modules.get(name)
        if module is None:
            continue
        version = getattr(module, "__version__", None)
        if version is None and hasattr(module, "get_version"):
            version = module.get_version()
        versions[name] = str(version)
    lib = getattr(getattr(sys.modules.get("llama_cpp"), "llama_cpp", None), "_lib", None)
    if lib is not None and getattr(lib, "_name", None):
        versions["llama_cpp_library"] = lib._name
    return versions


def _openmp_runtimes():
    libs = set()
    try:
        with open("/proc/self/maps", "r") as f:
            for line in f:
                path = line.split()[-1] if line.split() else ""
                if os.path.basename(path).startswith(("libgomp", "libiomp", "libomp")):
                    libs.add(path)
    except OSError:
        pass
    return sorted(libs)


def collect_run_info(args, extra=None):
    info = {
        "command_line": list(sys.orig_argv),
        "arguments": vars(args),
        "packages": _package_versions(),
        "openmp_runtimes": _openmp_runtimes(),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)},
        "host": {"hostname": platform.node(), "processor": platform.processor(), "kernel": platform.release(),
                 "python": platform.python_version()},
    }
    if hasattr(os, "sched_getaffinity"):
        info["host"]["process_affinity"] = sorted(os.sched_getaffinity(0))
    if extra:
        info.update(extra)
    return info


def dump_run_info(path, args, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collect_run_info(args, extra), f, indent=2, default=str)

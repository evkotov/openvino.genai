# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""--cpu_affinity is applied when benchmark.py imports this module, before openvino or torch start their thread pools:
threads created earlier would keep the old CPU mask. Only the standard library may be imported here."""

import os
import sys
import threading

OPTION = "--cpu_affinity"


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


def option_value(argv, names=(OPTION,)):
    for i, arg in enumerate(argv):
        for name in names:
            if arg == name and i + 1 < len(argv):
                return argv[i + 1]
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
    return None


def apply_cpu_affinity(cpus):
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError(f"{OPTION} is not supported on this platform")
    if threading.active_count() > 1:
        print(f"[ WARNING ] {OPTION}: {threading.active_count() - 1} threads already exist and keep their CPU mask",
              file=sys.stderr)
    try:
        os.sched_setaffinity(0, cpus)
    except OSError as e:
        raise RuntimeError(f"{OPTION}: can not run on CPUs {cpus} ({e}); allowed: {sorted(os.sched_getaffinity(0))}") from e
    return sorted(os.sched_getaffinity(0))


def apply_from_argv(argv=None):
    argv = sys.argv if argv is None else argv
    spec = option_value(argv)
    if spec is None:
        return None
    cpus = apply_cpu_affinity(parse_cpu_spec(spec))
    # llama.cpp CPU builds use OpenMP; bind its thread i to the i-th CPU. libgomp reads this when it is loaded
    # (torch loads it too), so it is set here. OpenVINO TBB builds ignore it.
    if option_value(argv, ("-f", "--framework")) == "llamacpp" and "GOMP_CPU_AFFINITY" not in os.environ:
        os.environ["GOMP_CPU_AFFINITY"] = " ".join(str(cpu) for cpu in cpus)
    return cpus


try:
    apply_from_argv()
except (ValueError, RuntimeError) as e:
    sys.exit(f"[ ERROR ] {OPTION}: {e}")

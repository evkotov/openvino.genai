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


def option_value(argv):
    for i, arg in enumerate(argv):
        if arg == OPTION and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(OPTION + "="):
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
    spec = option_value(sys.argv if argv is None else argv)
    if spec is None:
        return None
    return apply_cpu_affinity(parse_cpu_spec(spec))


try:
    apply_from_argv()
except (ValueError, RuntimeError) as e:
    sys.exit(f"[ ERROR ] {OPTION}: {e}")

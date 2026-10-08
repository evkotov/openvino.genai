# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest

from llm_bench_utils.cpu_affinity import apply_from_argv, option_value, parse_cpu_spec

LLM_BENCH_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("spec, cpus", [("3", [3]), ("0-3", [0, 1, 2, 3]), ("1,3,5-9:2", [1, 3, 5, 7, 9]),
                                        ("1-15:2", list(range(1, 16, 2))), ("4, 2,2", [2, 4])])
def test_parse_cpu_spec(spec, cpus):
    assert parse_cpu_spec(spec) == cpus


@pytest.mark.parametrize("spec", ["", "a", "3-1", "1-4:0", "-1", "1,,x"])
def test_parse_cpu_spec_errors(spec):
    with pytest.raises(ValueError):
        parse_cpu_spec(spec)


@pytest.mark.parametrize("argv, value", [(["b.py", "--cpu_affinity", "1,3"], "1,3"), (["b.py", "--cpu_affinity=0-2"], "0-2"),
                                         (["b.py", "-m", "x"], None), (["b.py", "--cpu_affinity"], None)])
def test_option_value(argv, value):
    assert option_value(argv) == value


def test_no_option_keeps_mask():
    before = os.sched_getaffinity(0)
    assert apply_from_argv(["benchmark.py"]) is None
    assert os.sched_getaffinity(0) == before


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux only")
def test_mask_is_applied_on_import_and_inherited_by_threads():
    cpu = min(os.sched_getaffinity(0))
    code = ("import os, threading, llm_bench_utils.cpu_affinity\n"
            "masks = []\n"
            "t = threading.Thread(target=lambda: masks.append(os.sched_getaffinity(0))); t.start(); t.join()\n"
            "print(sorted(os.sched_getaffinity(0)), sorted(masks[0]))\n")
    out = subprocess.run([sys.executable, "-c", code, "--cpu_affinity", str(cpu)], cwd=LLM_BENCH_DIR,
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == f"[{cpu}] [{cpu}]"


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux only")
@pytest.mark.parametrize("spec, message", [("x", "invalid CPU list item"), ("100000", "can not run on CPUs")])
def test_bad_value_exits_with_message(spec, message):
    result = subprocess.run([sys.executable, "-c", "import llm_bench_utils.cpu_affinity", "--cpu_affinity", spec],
                            cwd=LLM_BENCH_DIR, capture_output=True, text=True)
    assert result.returncode != 0
    assert message in result.stderr

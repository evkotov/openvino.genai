# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

from llm_bench_utils import system_info
from llm_bench_utils.system_info import (EnvironmentRecorder, busy_thread_cpus, cpu_power, cpu_topology, format_cpu_list,
                                         parse_cpu_list, throttle_counters)


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_hybrid_sys(root):
    """4 P-core threads (2 cores with SMT) + 2 E-cores, intel_pstate."""
    cpu = "devices/system/cpu"
    write(root, f"{cpu}/online", "0-5\n")
    write(root, "devices/cpu_core/cpus", "0-3\n")
    write(root, "devices/cpu_atom/cpus", "4-5\n")
    siblings = {0: "0-1", 1: "0-1", 2: "2-3", 3: "2-3", 4: "4", 5: "5"}
    for i in range(6):
        write(root, f"{cpu}/cpu{i}/topology/thread_siblings_list", siblings[i])
        write(root, f"{cpu}/cpu{i}/cpufreq/scaling_driver", "intel_pstate")
        write(root, f"{cpu}/cpu{i}/cpufreq/scaling_governor", "performance")
        write(root, f"{cpu}/cpu{i}/cpufreq/energy_performance_preference", "performance")
        write(root, f"{cpu}/cpu{i}/cpufreq/scaling_min_freq", "800000")
        write(root, f"{cpu}/cpu{i}/cpufreq/scaling_max_freq", "5000000" if i < 4 else "3900000")
        write(root, f"{cpu}/cpu{i}/thermal_throttle/core_throttle_count", "2")
        write(root, f"{cpu}/cpu{i}/thermal_throttle/package_throttle_count", "1")
        for s, name in enumerate(("POLL", "C1E", "C6")):
            write(root, f"{cpu}/cpu{i}/cpuidle/state{s}/name", name)
            write(root, f"{cpu}/cpu{i}/cpuidle/state{s}/disable", "1" if (name == "C6" and i < 4) else "0")
    write(root, f"{cpu}/intel_pstate/no_turbo", "0")
    write(root, "kernel/mm/transparent_hugepage/enabled", "always [madvise] never")


def make_amd_sys(root):
    """2 cores without SMT, acpi-cpufreq with boost, no hybrid PMUs and no throttle counters."""
    cpu = "devices/system/cpu"
    write(root, f"{cpu}/online", "0-1")
    for i in range(2):
        write(root, f"{cpu}/cpu{i}/topology/thread_siblings_list", str(i))
        write(root, f"{cpu}/cpu{i}/cpufreq/scaling_driver", "acpi-cpufreq")
        write(root, f"{cpu}/cpu{i}/cpufreq/scaling_governor", "schedutil")
    write(root, f"{cpu}/cpufreq/boost", "1")


@pytest.mark.parametrize("text, cpus", [("0-3,8,10-11", [0, 1, 2, 3, 8, 10, 11]), ("5", [5]), ("", []), (None, [])])
def test_parse_cpu_list(text, cpus):
    assert parse_cpu_list(text) == cpus


def test_format_cpu_list_round_trip():
    assert format_cpu_list([1, 3, 5, 6, 7, 0]) == "0-1,3,5-7"
    assert parse_cpu_list(format_cpu_list([1, 3, 5, 6, 7])) == [1, 3, 5, 6, 7]


def test_hybrid_topology_and_power(tmp_path):
    make_hybrid_sys(tmp_path)
    topo = cpu_topology(str(tmp_path))
    assert topo == {"online": "0-5", "logical_cpus": 6, "core_types": {"performance": "0-3", "efficient": "4-5"},
                    "physical_cores": 4, "smt_active": True}
    power = cpu_power(str(tmp_path))
    assert power["governor"] == "performance"
    assert power["scaling_max_khz"] == {"5000000": 4, "3900000": 2}
    assert power["turbo"] is True
    assert power["cstates_disabled"] == {"0-3": "C6"}
    assert throttle_counters(str(tmp_path)) == {"core": 12, "package": 6}


def test_non_hybrid_without_intel_pstate(tmp_path):
    make_amd_sys(tmp_path)
    topo = cpu_topology(str(tmp_path))
    assert "core_types" not in topo and topo["smt_active"] is False
    power = cpu_power(str(tmp_path))
    assert power["scaling_driver"] == "acpi-cpufreq" and power["turbo"] is True
    assert "energy_performance_preference" not in power and "cstates_disabled" not in power
    assert throttle_counters(str(tmp_path)) is None


def test_empty_sysfs(tmp_path):
    assert cpu_topology(str(tmp_path)) == {"online": "", "logical_cpus": 0}
    assert cpu_power(str(tmp_path)) == {}


def test_busy_thread_cpus(tmp_path):
    # stat fields after the command name: state, ..., utime (14), stime (15), ..., processor (39)
    def stat(utime, stime, cpu):
        fields = ["S"] + ["0"] * 10 + [str(utime), str(stime)] + ["0"] * 23 + [str(cpu)] + ["0"] * 5
        return "123 (python (x)) " + " ".join(fields)

    for tid, (ut, st, cpu) in {"1": (1000, 10, 1), "2": (900, 0, 3), "3": (5, 0, 7)}.items():
        write(tmp_path, f"42/task/{tid}/stat", stat(ut, st, cpu))
    assert busy_thread_cpus(42, str(tmp_path)) == "1,3"


def test_recorder_collects_json_serializable_data(tmp_path, monkeypatch):
    make_hybrid_sys(tmp_path / "sys")
    write(tmp_path / "proc", "loadavg", "0.50 0.40 0.30 1/100 5")
    write(tmp_path / "proc", "meminfo", "MemTotal: 1000 kB\nMemAvailable: 500 kB\n")
    monkeypatch.setenv("OV_TEST_VARIABLE", "1")
    model = tmp_path / "model.bin"
    model.write_bytes(b"abc")
    recorder = EnvironmentRecorder(str(tmp_path / "sys"), str(tmp_path / "proc"))
    env = recorder.collect(model, with_hash=True, argv=["benchmark.py", "-m", "model.bin"])
    json.dumps(env)
    assert env["command_line"] == ["benchmark.py", "-m", "model.bin"]
    assert env["env"]["OV_TEST_VARIABLE"] == "1"
    assert env["run"]["loadavg_start"] == [0.5, 0.4, 0.3]
    assert env["run"]["throttle_events"] == {"core": 0, "package": 0}
    assert env["model"]["size_bytes"] == 3
    assert env["model"]["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert env["host"]["memory"]["transparent_hugepage"] == "madvise"
    if hasattr(os, "sched_getaffinity"):
        assert env["host"]["process_affinity"]


def test_package_versions_do_not_import_modules(monkeypatch):
    monkeypatch.delitem(system_info.sys.modules, "llama_cpp", raising=False)
    system_info.package_versions()
    assert "llama_cpp" not in system_info.sys.modules


def test_gguf_model_info(tmp_path):
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    path = tmp_path / "tiny.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_file_type(2)
    writer.add_tensor("a", np.zeros(48, dtype=np.float32))
    writer.add_tensor("b", np.zeros(32, dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    info = system_info.model_info(path)["gguf"]
    assert info["architecture"] == "llama"
    assert info["file_type"] == 2
    assert info["tensor_types"] == {"F32": 0.75, "F16": 0.25}

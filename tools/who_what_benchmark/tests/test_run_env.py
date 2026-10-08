# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json

import pytest

from whowhatbench.run_env import collect_run_info, dump_run_info, parse_cpu_spec


@pytest.mark.parametrize("spec, cpus", [("3", [3]), ("0-3", [0, 1, 2, 3]), ("1,3,5-9:2", [1, 3, 5, 7, 9])])
def test_parse_cpu_spec(spec, cpus):
    assert parse_cpu_spec(spec) == cpus


@pytest.mark.parametrize("spec", ["", "a", "3-1", "1-4:0"])
def test_parse_cpu_spec_errors(spec):
    with pytest.raises(ValueError):
        parse_cpu_spec(spec)


def test_run_info_is_json(tmp_path, monkeypatch):
    monkeypatch.setenv("OV_SOMETHING", "1")
    args = argparse.Namespace(base_model="m", llamacpp=True, path=tmp_path)
    path = tmp_path / "out" / "run_info_base.json"
    dump_run_info(str(path), args)
    info = json.loads(path.read_text())
    assert info["arguments"]["base_model"] == "m"
    assert info["env"]["OV_SOMETHING"] == "1"
    assert "python" in info["host"]
    assert collect_run_info(args, {"role": "base"})["role"] == "base"

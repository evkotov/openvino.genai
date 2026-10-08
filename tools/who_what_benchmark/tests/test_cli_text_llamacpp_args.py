# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

from conftest import run_wwb
from huggingface_hub import hf_hub_download
from whowhatbench import model_loaders


pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="llama.cpp tests run on Linux only")


def _assert_wwb_cli_error(args: list[str], expected_message: str) -> str:
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_wwb(args)

    output = error.value.output
    assert expected_message in output
    return output


def test_text_llamacpp_chat_requires_llamacpp(tmp_path):
    gt_data = tmp_path / "gt.csv"
    gt_data.write_text("stub\n", encoding="utf-8")

    _assert_wwb_cli_error(
        [
            "--gt-data",
            str(gt_data),
            "--model-type",
            "text",
            "--llamacpp-chat",
        ],
        "--llamacpp-chat requires --llamacpp",
    )


def test_text_llamacpp_n_ctx_requires_llamacpp():
    with pytest.raises(ValueError, match="--llamacpp-n-ctx requires --llamacpp"):
        model_loaders.load_model(
            "text",
            "dummy_model",
            "CPU",
            None,
            False,
            False,
            False,
            llamacpp_n_ctx=4096,
        )


def test_text_non_llamacpp_run_not_blocked_by_n_ctx_default(tmp_path):
    output = _assert_wwb_cli_error(
        [
            "--gt-data",
            str(tmp_path / "missing.csv"),
            "--model-type",
            "text",
        ],
        "Ground-truth data file",
    )

    assert "--llamacpp-n-ctx requires --llamacpp" not in output


def test_cli_rejects_mutually_exclusive_backend_flags():
    output = _assert_wwb_cli_error(
        [
            "--base-model",
            "dummy_base_model",
            "--model-type",
            "text",
            "--llamacpp",
            "--hf",
        ],
        "Options --hf, --genai and --llamacpp are mutually exclusive",
    )

    assert "Options --hf, --genai and --llamacpp are mutually exclusive" in output


def test_loader_raises_for_llamacpp_n_ctx_when_hf_backend_selected():
    with pytest.raises(
        ValueError,
        match="--llamacpp-n-ctx is supported only when llama.cpp is the selected text backend",
    ):
        model_loaders.load_model(
            "text",
            "dummy_model",
            "CPU",
            None,
            True,
            False,
            True,
            llamacpp_n_ctx=4096,
        )


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {"llamacpp_n_ctx": None},
        {},
    ],
    ids=["explicit_none", "key_omitted"],
)
def test_loader_sets_llamacpp_n_ctx_default_for_llamacpp_backend(monkeypatch, extra_kwargs):
    captured = {}

    def fake_load_text_llamacpp_pipeline(model_dir, **kwargs):
        captured["model_dir"] = model_dir
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(model_loaders, "load_text_llamacpp_pipeline", fake_load_text_llamacpp_pipeline)

    model_loaders.load_model(
        "text",
        "dummy_model",
        "CPU",
        None,
        False,
        False,
        True,
        **extra_kwargs,
    )

    assert captured["kwargs"]["llamacpp_n_ctx"] == 8192


def test_cli_rejects_llamacpp_for_text_chat(tmp_path):
    gt_data = tmp_path / "gt.csv"
    gt_data.write_text("stub\n", encoding="utf-8")

    _assert_wwb_cli_error(
        [
            "--gt-data",
            str(gt_data),
            "--model-type",
            "text-chat",
            "--llamacpp",
        ],
        "--llamacpp is supported only with --model-type text",
    )


def _get_tiny_llamacpp_model() -> Path:
    return Path(
        hf_hub_download(
            repo_id="afrideva/Tinystories-gpt-0.1-3m-GGUF",
            filename="tinystories-gpt-0.1-3m.Q2_K.gguf",
        )
    )


def test_text_llamacpp_real_gguf_logs_backend_and_n_ctx(tmp_path):
    pytest.importorskip("llama_cpp")

    gguf_path = _get_tiny_llamacpp_model()
    gt_data = tmp_path / "llamacpp_gt.csv"
    n_ctx = 2048

    output = run_wwb(
        [
            "--base-model",
            str(gguf_path),
            "--llamacpp",
            "--llamacpp-n-ctx",
            str(n_ctx),
            "--model-type",
            "text",
            "--gt-data",
            str(gt_data),
            "--num-samples",
            "1",
            "--max_new_tokens",
            "8",
        ]
    )

    assert "Using llama.cpp API" in output
    assert f"Using llama.cpp API (n_ctx={n_ctx})" in output
    assert gt_data.exists()


def test_cli_llamacpp_config_requires_llamacpp(tmp_path):
    gt_data = tmp_path / "gt.csv"
    gt_data.write_text("stub\n", encoding="utf-8")

    _assert_wwb_cli_error(
        ["--gt-data", str(gt_data), "--model-type", "text", "--llamacpp-config", '{"n_threads": 2}'],
        "--llamacpp-config requires --llamacpp",
    )


def test_cli_rejects_bad_cpu_affinity(tmp_path):
    gt_data = tmp_path / "gt.csv"
    gt_data.write_text("stub\n", encoding="utf-8")

    _assert_wwb_cli_error(
        ["--gt-data", str(gt_data), "--model-type", "text", "--cpu-affinity", "3-1"],
        "invalid CPU list item",
    )


def test_loader_passes_llamacpp_config_to_llama(monkeypatch):
    captured = {}

    class FakeLlama:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "llama_cpp", type(sys)("llama_cpp"))
    sys.modules["llama_cpp"].Llama = FakeLlama

    model_loaders.load_model(
        "text", "model_dir", "CPU", None, False, False, True,
        gguf_file="model.gguf", llamacpp_config={"n_threads": 4, "flash_attn": True},
    )

    assert captured["model_path"].endswith("model.gguf")
    assert captured["kwargs"] == {"n_threads": 4, "n_threads_batch": 4, "flash_attn": True, "n_ctx": 8192}


@pytest.mark.parametrize(
    "config, n_ctx, message",
    [
        ({"n_ctx": 1024}, 2048, "conflicts with n_ctx"),
        ({"model_path": "x"}, None, "must not set model_path"),
        ([1, 2], None, "must be a JSON object"),
    ],
)
def test_loader_rejects_bad_llamacpp_config(config, n_ctx, message):
    with pytest.raises(ValueError, match=message):
        model_loaders.load_model(
            "text", "model_dir", "CPU", None, False, False, True, llamacpp_n_ctx=n_ctx, llamacpp_config=config,
        )


def test_loader_takes_n_ctx_from_llamacpp_config(monkeypatch):
    captured = {}

    def fake_load_text_llamacpp_pipeline(model_dir, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(model_loaders, "load_text_llamacpp_pipeline", fake_load_text_llamacpp_pipeline)
    model_loaders.load_model("text", "model_dir", "CPU", None, False, False, True, llamacpp_config={"n_ctx": 1024})
    assert captured["kwargs"]["llamacpp_n_ctx"] == 1024


def test_text_llamacpp_real_gguf_config_affinity_and_run_info(tmp_path):
    pytest.importorskip("llama_cpp")
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("CPU affinity is Linux only")

    gguf_path = _get_tiny_llamacpp_model()
    cpu = min(os.sched_getaffinity(0))
    run_wwb(
        [
            "--base-model", str(gguf_path), "--llamacpp", "--llamacpp-config", '{"n_threads": 1}',
            "--cpu-affinity", str(cpu), "--model-type", "text", "--gt-data", str(tmp_path / "gt.csv"),
            "--num-samples", "1", "--max_new_tokens", "4",
        ]
    )

    run_info = json.loads((tmp_path / "gt.run_info.json").read_text())
    assert run_info["arguments"]["llamacpp_config"] == '{"n_threads": 1}'
    assert run_info["host"]["process_affinity"] == [cpu]
    assert "llama_cpp" in run_info["packages"]

# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from llm_bench_utils.model_utils import apply_ov_config_defaults


def test_pa_is_injected_with_warning(caplog):
    config = {}
    with caplog.at_level(logging.WARNING):
        injected = apply_ov_config_defaults(config, None, "text_gen", "CPU", optimum=False)
    assert config == {"CACHE_DIR": "", "ATTENTION_BACKEND": "PA"}
    assert injected == config
    assert "ATTENTION_BACKEND is not set" in caplog.text


@pytest.mark.parametrize("backend", ["PA", "SDPA"])
def test_explicit_backend_is_not_reported_as_injected(backend):
    config = {}
    injected = apply_ov_config_defaults(config, backend, "text_gen", "CPU", optimum=False)
    assert config["ATTENTION_BACKEND"] == backend
    assert injected == {"CACHE_DIR": ""}


def test_backend_from_ov_config_is_kept():
    config = {"ATTENTION_BACKEND": "SDPA", "CACHE_DIR": "cache"}
    assert apply_ov_config_defaults(config, None, "text_gen", "CPU", optimum=False) == {}
    assert config == {"ATTENTION_BACKEND": "SDPA", "CACHE_DIR": "cache"}


def test_same_value_in_option_and_ov_config():
    config = {"ATTENTION_BACKEND": "SDPA"}
    apply_ov_config_defaults(config, "SDPA", "text_gen", "CPU", optimum=False)
    assert config["ATTENTION_BACKEND"] == "SDPA"


def test_conflicting_backend_raises():
    with pytest.raises(RuntimeError, match="conflicts"):
        apply_ov_config_defaults({"ATTENTION_BACKEND": "PA"}, "SDPA", "text_gen", "CPU", optimum=False)


@pytest.mark.parametrize("task, device, optimum", [("image_gen", "CPU", False), ("text_gen", "NPU", False),
                                                   ("text_gen", "CPU", True)])
def test_no_pa_where_it_does_not_apply(task, device, optimum):
    config = {}
    apply_ov_config_defaults(config, None, task, device, optimum)
    assert "ATTENTION_BACKEND" not in config

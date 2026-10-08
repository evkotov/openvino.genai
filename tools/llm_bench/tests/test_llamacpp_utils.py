# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from llm_bench_utils import llamacpp_utils
from llm_bench_utils.token_prompt import TokenIdsPrompt


class FakeModel:
    """Records the eval() calls; the next token is the number of evaluated positions."""

    def __init__(self, n_ctx=64):
        self._n_ctx = n_ctx
        self.calls = []
        self.n_tokens = 0
        self.resets = 0

    def n_ctx(self):
        return self._n_ctx

    def n_vocab(self):
        return 100

    def reset(self):
        self.resets += 1
        self.n_tokens = 0

    def eval(self, tokens):
        self.calls.append(list(tokens))
        self.n_tokens += len(tokens)


@pytest.fixture
def fake_greedy(monkeypatch):
    monkeypatch.setattr(llamacpp_utils, "_greedy_token", lambda model, n_vocab: model.n_tokens)


def test_generate_prefill_then_one_token_steps(fake_greedy):
    model = FakeModel()
    tokens, durations = llamacpp_utils.generate(model, [5, 6, 7], max_new_tokens=4)
    assert model.calls == [[5, 6, 7], [3], [4], [5]]
    assert tokens == [3, 4, 5, 6]
    assert len(durations) == 4 and all(d >= 0 for d in durations)


def test_generate_starts_from_empty_cache_every_time(fake_greedy):
    model = FakeModel()
    first, _ = llamacpp_utils.generate(model, [1, 2], 3)
    second, _ = llamacpp_utils.generate(model, [1, 2], 3)
    assert first == second and model.resets == 2


def test_generate_checks_context_size(fake_greedy):
    with pytest.raises(RuntimeError, match="exceed n_ctx"):
        llamacpp_utils.generate(FakeModel(n_ctx=8), list(range(6)), 4)


def test_required_context():
    assert llamacpp_utils.required_context(["text"], None, 128) == 4096
    assert llamacpp_utils.required_context([TokenIdsPrompt(range(4096))], None, 129) == 4352
    assert llamacpp_utils.required_context([TokenIdsPrompt(range(8000))], 4000, 100) == 4352


def test_model_config_defaults_and_overrides():
    pytest.importorskip("llama_cpp")
    params = llamacpp_utils.model_config({"n_threads": 4, "type_k": 8}, prompts=[TokenIdsPrompt(range(5000))],
                                         infer_count=16)
    assert params["n_threads"] == 4 and params["n_threads_batch"] == 4
    assert params["type_k"] == 8 and params["flash_attn"] is True and params["n_batch"] == 2048
    assert params["n_ctx"] == 5120
    assert params["verbose"] is False


def test_model_config_threads_from_cpu_affinity():
    pytest.importorskip("llama_cpp")
    params = llamacpp_utils.model_config({}, cpu_affinity="1-15:2")
    assert params["n_threads"] == 8 and params["n_threads_batch"] == 8


def test_model_config_rejects_unknown_keys():
    pytest.importorskip("llama_cpp")
    with pytest.raises(RuntimeError, match="Unknown llama.cpp parameters"):
        llamacpp_utils.model_config({"INFERENCE_NUM_THREADS": 8})


def test_llamacpp_defaults_to_text_generation(tmp_path):
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    pytest.importorskip("gguf_parser")
    from llm_bench_utils.get_use_case import get_use_case
    from llm_bench_utils.model_utils import select_task

    path = tmp_path / "tiny-gemma4.gguf"
    writer = gguf.GGUFWriter(str(path), "gemma4")
    writer.add_tensor("t", np.zeros(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert select_task(None, "llamacpp") == "text_gen"
    assert select_task("text_gen_chat", "llamacpp") == "text_gen_chat"
    assert select_task(None, "ov") is None
    assert get_use_case(path, select_task(None, "llamacpp"))[0].task == "text_gen"

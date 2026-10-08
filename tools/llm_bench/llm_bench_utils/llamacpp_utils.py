# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""llama.cpp (llama-cpp-python) as a text generation framework, to compare it with OpenVINO on the same GGUF files."""

import inspect
import logging as log
import os
import time

import numpy as np

from llm_bench_utils.cpu_affinity import parse_cpu_spec
from llm_bench_utils.token_prompt import TokenIdsPrompt

# llama-bench defaults (CPU); flash attention "auto" resolves to enabled on CPU
DEFAULT_CONFIG = {"n_ctx": 4096, "n_batch": 2048, "n_ubatch": 512, "flash_attn": True}
DEFAULT_OUTPUT_TOKEN_SIZE = 512  # llm_bench default of -ic


def import_llama_cpp():
    try:
        import llama_cpp
    except ImportError as e:
        raise RuntimeError("Framework llamacpp requires llama-cpp-python: pip install llama-cpp-python") from e
    return llama_cpp


def required_context(prompts, num_prefill_tokens, infer_count):
    """KV cache size for the longest token id prompt (text prompts are not tokenized yet), rounded up to 256."""
    lengths = [min(len(p), num_prefill_tokens or len(p)) for p in prompts if isinstance(p, TokenIdsPrompt)]
    need = max(lengths, default=0) + (infer_count or DEFAULT_OUTPUT_TOKEN_SIZE)
    return max(DEFAULT_CONFIG["n_ctx"], -(-need // 256) * 256)


def model_config(config, cpu_affinity=None, prompts=(), num_prefill_tokens=None, infer_count=None):
    """Llama() arguments: llama-bench like defaults, then the user's --ov-config / -lc values."""
    llama_cpp = import_llama_cpp()
    allowed = set(inspect.signature(llama_cpp.Llama.__init__).parameters) - {"self", "model_path", "kwargs"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise RuntimeError(f"Unknown llama.cpp parameters in the config: {unknown}")
    params = dict(DEFAULT_CONFIG)
    params["n_ctx"] = required_context(prompts, num_prefill_tokens, infer_count)
    params.update(config)
    if params.get("n_threads") is None:
        if cpu_affinity is not None:
            # not os.sched_getaffinity(): with GOMP_CPU_AFFINITY libgomp binds the main thread to the first CPU
            params["n_threads"] = len(parse_cpu_spec(cpu_affinity))
        else:
            params["n_threads"] = max(os.cpu_count() // 2, 1)
            log.warning(f"llama.cpp n_threads is not set, using {params['n_threads']}; "
                        "set it with -lc '{\"n_threads\": N}' or --cpu_affinity")
    # llama-bench uses the same thread count for prompt processing and generation
    params.setdefault("n_threads_batch", params["n_threads"])
    params["verbose"] = bool(params.get("verbose", False))
    return params


def resolve_model_args(model_args):
    """Checks the use case and replaces model_args['config'] with the full Llama() arguments."""
    from llm_bench_utils.prompt_utils import get_text_prompt

    task = model_args["use_case"].task
    if task != "text_gen":
        raise RuntimeError(f"Framework llamacpp supports only text generation, not {task}")
    if model_args["batch_size"] != 1 or model_args["num_beams"] != 1:
        raise RuntimeError("Framework llamacpp supports only batch size 1 and greedy search")
    model_args["config"] = model_config(model_args["config"], model_args.get("cpu_affinity"),
                                        get_text_prompt(model_args, allow_token_ids=True),
                                        model_args.get("num_prefill_tokens"), model_args.get("infer_count"))


def create_text_gen_model(model_path, device, memory_data_collector, **kwargs):
    if str(device).upper() != "CPU" and "n_gpu_layers" not in kwargs["config"]:
        raise RuntimeError("Framework llamacpp runs on CPU unless n_gpu_layers is set in the config")
    llama_cpp = import_llama_cpp()
    params = kwargs["config"]
    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()
    model = llama_cpp.Llama(model_path=str(model_path), **params)
    end = time.perf_counter()
    log.info(f"llama.cpp model load time: {end - start:.2f}s, parameters: {params}")
    system_info = llama_cpp.llama_print_system_info().decode(errors='replace').strip()
    log.info(f"llama.cpp system info: {system_info}")
    if "OPENMP = 1" in system_info and "OMP_WAIT_POLICY" not in os.environ:
        log.warning("OMP_WAIT_POLICY is not set: llama.cpp OpenMP threads use the default wait policy of the loaded "
                    "libgomp (torch bundles its own), which changes the decode speed noticeably. Set OMP_WAIT_POLICY "
                    "explicitly (and the same way for the reference llama.cpp runs).")
    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data("compilation")
        memory_data_collector.log_data(compilation=True)
    return model, None, end - start, None, False


def tokenize(model, text):
    return model.tokenize(text.encode("utf-8"), add_bos=True, special=True)


def detokenize(model, tokens):
    return model.detokenize(tokens).decode("utf-8", errors="replace")


def _greedy_token(model, n_vocab):
    llama_cpp = import_llama_cpp()
    logits = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(model.ctx, -1), shape=(n_vocab,))
    return int(np.argmax(logits))


def generate(model, input_ids, max_new_tokens):
    """Greedy generation of exactly max_new_tokens tokens (EOS is ignored) from an empty KV cache.

    Returns the tokens and the duration of every step in seconds; the first step is the prompt processing."""
    if len(input_ids) + max_new_tokens > model.n_ctx():
        raise RuntimeError(f"Prompt ({len(input_ids)}) + new tokens ({max_new_tokens}) exceed n_ctx ({model.n_ctx()}); "
                           "set a larger n_ctx in the config")
    n_vocab = model.n_vocab()
    # reset() + eval() removes every cached position, so no prefix is reused between iterations
    model.reset()
    tokens, durations = [], []
    step_input = list(input_ids)
    while len(tokens) < max_new_tokens:
        start = time.perf_counter()
        model.eval(step_input)
        token = _greedy_token(model, n_vocab)
        durations.append(time.perf_counter() - start)
        tokens.append(token)
        step_input = [token]
    return tokens, durations

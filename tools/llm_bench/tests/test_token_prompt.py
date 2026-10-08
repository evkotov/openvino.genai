# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest

from llm_bench_utils.parse_json_data import parse_text_json_data
from llm_bench_utils.prompt_utils import get_text_prompt
from llm_bench_utils.token_prompt import (TokenIdsPrompt, parse_token_ids, read_token_ids_file, token_ids_batch,
                                          token_ids_digest)


def prompt_args(task="text_gen", **kwargs):
    args = {"prompt": None, "prompt_file": None, "prompt_token_ids": None, "use_case": SimpleNamespace(task=task)}
    args.update(kwargs)
    return args


def test_read_token_ids_file(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("1 22\n333\t4\n")
    ids = read_token_ids_file(path)
    assert isinstance(ids, TokenIdsPrompt)
    assert ids == [1, 22, 333, 4]


@pytest.mark.parametrize("content", ["", "1 x 3", "1 -2"])
def test_read_token_ids_file_rejects_bad_content(tmp_path, content):
    path = tmp_path / "ids.txt"
    path.write_text(content)
    with pytest.raises(RuntimeError):
        read_token_ids_file(path)


@pytest.mark.parametrize("value", [[], "1 2", [1, 2.0], [True], None])
def test_parse_token_ids_rejects_bad_values(value):
    with pytest.raises(RuntimeError):
        parse_token_ids(value)


def test_token_ids_batch():
    prompt = TokenIdsPrompt([5, 6, 7, 8])
    batch = token_ids_batch(prompt, batch_size=2, num_prefill_tokens=3)
    assert batch.dtype == np.int64
    assert batch.tolist() == [[5, 6, 7], [5, 6, 7]]
    assert token_ids_batch(prompt, 1).tolist() == [[5, 6, 7, 8]]
    with pytest.raises(RuntimeError, match="larger than"):
        token_ids_batch(prompt, 1, num_prefill_tokens=5)


def test_token_ids_digest_depends_on_ids_only():
    assert token_ids_digest([1, 2, 3]) == token_ids_digest(TokenIdsPrompt([1, 2, 3]))
    assert token_ids_digest([1, 2, 3]) != token_ids_digest([1, 2, 4])
    assert str(TokenIdsPrompt([1, 2, 3])).startswith("<3 token ids")


def test_prompt_file_mixes_text_and_ids():
    prompts = parse_text_json_data([{"prompt": "hello"}, {"input_ids": [1, 2, 3]}])
    assert prompts[0] == "hello"
    assert isinstance(prompts[1], TokenIdsPrompt) and prompts[1] == [1, 2, 3]


def test_get_text_prompt_from_token_ids_file(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("7 8 9")
    prompts = get_text_prompt(prompt_args(prompt_token_ids=str(path)), allow_token_ids=True)
    assert prompts == [[7, 8, 9]] and isinstance(prompts[0], TokenIdsPrompt)


def test_get_text_prompt_from_jsonl_ids(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"input_ids": [3, 4]}) + "\n")
    prompts = get_text_prompt(prompt_args(prompt_file=[str(path)]), allow_token_ids=True)
    assert prompts == [[3, 4]]


def test_token_ids_file_conflicts_with_prompt(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("1 2")
    with pytest.raises(RuntimeError, match="can not be combined"):
        get_text_prompt(prompt_args(prompt="text", prompt_token_ids=str(path)), allow_token_ids=True)


def test_token_ids_rejected_outside_text_generation(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("1 2")
    with pytest.raises(RuntimeError, match="supported only for text generation"):
        get_text_prompt(prompt_args(task="text_gen_chat", prompt_token_ids=str(path)))

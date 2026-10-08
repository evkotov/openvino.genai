# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib

import numpy as np


class TokenIdsPrompt(list):
    """A text generation prompt given as token ids; it is fed to the model without tokenization."""

    def __str__(self):
        return f"<{len(self)} token ids, sha1 {token_ids_digest(self)}>"


def parse_token_ids(value, source="prompt"):
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise RuntimeError(f"== {source}: input_ids must be a non-empty list of token ids ==")
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise RuntimeError(f"== {source}: invalid token id {token!r} ==")
    return TokenIdsPrompt(value)


def read_token_ids_file(path):
    """Reads whitespace separated token ids, the format of llama-bench --prompt-tokens-file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        ids = [int(token) for token in text.split()]
    except ValueError as e:
        raise RuntimeError(f"== {path}: token ids must be integers ==") from e
    return parse_token_ids(ids, str(path))


def token_ids_digest(ids):
    return hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes(), usedforsecurity=False).hexdigest()[:16]


def token_ids_batch(prompt, batch_size, num_prefill_tokens=None):
    """Returns int64 input ids of shape [batch_size, n]; n = num_prefill_tokens if set (must fit the prompt)."""
    ids = list(prompt)
    if num_prefill_tokens is not None:
        if num_prefill_tokens > len(ids):
            raise RuntimeError(f"== num_prefill_tokens({num_prefill_tokens}) is larger than the token id prompt({len(ids)}) ==")
        ids = ids[:num_prefill_tokens]
    return np.array([ids] * batch_size, dtype=np.int64)

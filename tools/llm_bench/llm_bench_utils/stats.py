# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

# These statistics describe the iterations of one process. Run-to-run (process-to-process)
# variation is usually larger, so comparisons should repeat the benchmark in several processes.


def summarize(values, ci=0.95, n_boot=2000, seed=0):
    """Median, spread and a bootstrap confidence interval of the median of the samples."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return None
    stdev = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    mean = float(np.mean(x))
    res = {
        "n": int(x.size),
        "mean": mean,
        "median": float(np.median(x)),
        "stdev": stdev,
        "cv_pct": 100.0 * stdev / mean if mean != 0 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p90": float(np.percentile(x, 90)),
    }
    if x.size > 1:
        rng = np.random.default_rng(seed)
        medians = np.median(x[rng.integers(0, x.size, (n_boot, x.size))], axis=1)
        alpha = (1.0 - ci) / 2.0
        res["median_ci"] = [float(np.quantile(medians, alpha)), float(np.quantile(medians, 1.0 - alpha))]
    else:
        res["median_ci"] = [res["median"], res["median"]]
    return {key: [round(v, 5) for v in value] if isinstance(value, list) else round(value, 5) if isinstance(value, float) else value
            for key, value in res.items()}


def _valid(value):
    return value not in ("", None) and value >= 0


def prompt_stats(iter_data_list, batch_size):
    """Per prompt statistics of the measured (non warm-up) iterations, keyed by prompt index."""
    stats = {}
    for prompt_idx in sorted({data["prompt_idx"] for data in iter_data_list}, key=str):
        iters = [data for data in iter_data_list if data["prompt_idx"] == prompt_idx and data["iteration"] != 0]
        first = [d["first_token_latency"] for d in iters if _valid(d["first_token_latency"])]
        second = [d["other_tokens_avg_latency"] for d in iters if _valid(d["other_tokens_avg_latency"])]
        entry = {}
        if first:
            entry["first_latency"] = summarize(first)
            # prefill throughput, input tokens per second
            pairs = [(d["input_size"], d["first_token_latency"]) for d in iters
                     if _valid(d["first_token_latency"]) and d["first_token_latency"] > 0
                     and isinstance(d["input_size"], (int, float)) and d["input_size"] > 0]
            if pairs:
                entry["first_token_throughput"] = summarize([size * 1000.0 / latency for size, latency in pairs])
        if second:
            entry["second_avg_latency"] = summarize(second)
            if all(s > 0 for s in second):
                entry["second_token_throughput"] = summarize([batch_size * 1000.0 / s for s in second])
        if entry:
            stats[str(prompt_idx)] = entry
    return stats

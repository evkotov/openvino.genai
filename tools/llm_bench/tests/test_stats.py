# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from llm_bench_utils import output_json
from llm_bench_utils.gen_output_data import gen_iterate_data
from llm_bench_utils.output_csv import output_avg_min_median
from llm_bench_utils.stats import prompt_stats, summarize


def test_summarize_known_values():
    stat = summarize([10.0, 12.0, 14.0])
    assert stat["n"] == 3
    assert stat["median"] == 12.0
    assert stat["stdev"] == 2.0
    assert stat["cv_pct"] == round(100 * 2.0 / 12.0, 5)
    assert stat["min"] == 10.0 and stat["max"] == 14.0
    low, high = stat["median_ci"]
    assert 10.0 <= low <= 12.0 <= high <= 14.0


def test_summarize_single_and_empty():
    stat = summarize([5.0])
    assert stat["stdev"] == 0.0 and stat["median_ci"] == [5.0, 5.0]
    assert summarize([]) is None


def test_summarize_is_deterministic():
    values = np.random.default_rng(1).normal(100, 5, 20)
    assert summarize(values) == summarize(values)


def test_median_ci_coverage():
    # the 95% CI of the median covers the true median in most of the repeated samples
    rng = np.random.default_rng(2)
    covered = 0
    for _ in range(200):
        low, high = summarize(rng.normal(50.0, 3.0, 15), n_boot=500)["median_ci"]
        covered += low <= 50.0 <= high
    assert covered >= 170


def make_iter(iteration, prompt_idx, in_size, first, second):
    data = gen_iterate_data(iter_idx=iteration, in_size=in_size, infer_count=4, out_size=4, gen_time=1.0, latency=1.0,
                            res_md5=["md5"], prompt_idx=prompt_idx)
    data.update(first_token_latency=first, other_tokens_avg_latency=second,
                first_token_infer_latency=first, other_tokens_infer_avg_latency=second)
    return data


ITERS = [make_iter(0, 0, 512, 900.0, 90.0), make_iter(1, 0, 512, 500.0, 20.0), make_iter(2, 0, 512, 520.0, 22.0),
         make_iter(3, 0, 512, 510.0, 21.0), make_iter(0, 1, 8, 50.0, 30.0), make_iter(1, 1, 8, 4.0, 10.0)]


def test_prompt_stats_excludes_warmup():
    stats = prompt_stats(ITERS, batch_size=1)
    assert set(stats) == {"0", "1"}
    assert stats["0"]["first_latency"]["n"] == 3
    assert stats["0"]["first_latency"]["median"] == 510.0
    assert stats["0"]["first_token_throughput"]["median"] == round(512 * 1000.0 / 510.0, 5)
    assert stats["0"]["second_token_throughput"]["median"] == round(1000.0 / 21.0, 5)
    assert stats["1"]["second_avg_latency"]["n"] == 1


def test_prompt_stats_skips_missing_values():
    stats = prompt_stats([make_iter(0, 0, 8, 1.0, 1.0), make_iter(1, 0, 8, 2.0, "")], batch_size=1)
    assert "second_avg_latency" not in stats["0"]


def test_json_report_has_stats(tmp_path):
    report = tmp_path / "report.json"
    output_json.write_result(str(report), "model", "ov", "CPU", {"num_beams": 1, "batch_size": 1}, ITERS, 1.0, "", {}, None)
    stats = json.loads(report.read_text())["perfdata"]["results_stats"]
    assert stats["0"]["second_avg_latency"]["median"] == 21.0


def test_csv_rows_have_stdev_and_cv():
    rows = output_avg_min_median(ITERS)[0]
    assert [row["iteration"] for row in rows] == ["avg", "mini", "median", "stdev", "cv(%)"]
    assert rows[3]["first_token_latency"] == pytest.approx(10.0)
    assert rows[4]["first_token_latency"] == pytest.approx(100 * 10.0 / 510.0)

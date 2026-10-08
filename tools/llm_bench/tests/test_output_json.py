# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

from llm_bench_utils import output_json
from llm_bench_utils.gen_output_data import gen_iterate_data


def make_iter(iteration, prompt_idx, first_latency, second_latency):
    data = gen_iterate_data(iter_idx=iteration, in_size=8, infer_count=4, out_size=4, gen_time=1.0,
                            latency=10.0, res_md5=["md5"], prompt_idx=prompt_idx)
    data["first_token_latency"] = first_latency
    data["other_tokens_avg_latency"] = second_latency
    data["first_token_infer_latency"] = first_latency
    data["other_tokens_infer_avg_latency"] = second_latency
    return data


def test_results_averaged_skips_warmup_of_every_prompt(tmp_path):
    # warm-up (iteration 0) values are far off, so including any of them changes the average
    iters = []
    for prompt_idx in (0, 1):
        iters.append(make_iter(0, prompt_idx, 1000.0, 1000.0))
        iters.append(make_iter(1, prompt_idx, 10.0, 2.0))
        iters.append(make_iter(2, prompt_idx, 20.0, 4.0))
    report = tmp_path / "report.json"
    output_json.write_result(str(report), "model", "ov", "CPU", {"num_beams": 1, "batch_size": 1}, iters,
                             1.0, "", {}, None)
    averaged = json.loads(report.read_text())["perfdata"]["results_averaged"]
    assert averaged["first_latency"] == 15.0
    assert averaged["second_avg_latency"] == 3.0
    assert averaged["second_token_throughput"] == round(1000.0 / 3.0, 5)


def test_no_results_averaged_for_warmup_only(tmp_path):
    report = tmp_path / "report.json"
    output_json.write_result(str(report), "model", "ov", "CPU", {"num_beams": 1, "batch_size": 1},
                             [make_iter(0, 0, 1000.0, 1000.0)], 1.0, "", {}, None)
    assert "results_averaged" not in json.loads(report.read_text())["perfdata"]

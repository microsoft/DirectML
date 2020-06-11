#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.

import matplotlib.pyplot as plt
import json
import os
import re
import argparse

script_root = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--traces_dir", "-t",
    default=os.path.join(script_root, "traces"),
    help="Path to traces directory")
args = parser.parse_args()

# Parse trace summaries
trace_environments = {}
for trace_dir in os.listdir(args.traces_dir):
    trace_dir_full = os.path.join(args.traces_dir, trace_dir)
    if os.path.isdir(trace_dir_full):
        trace_env = re.sub("(\w+)_(\d+)_(NCHW|NHWC).*", "\\1", trace_dir)
        trace_batch_size = re.sub("(\w+)_(\d+)_(NCHW|NHWC).*", "\\2", trace_dir)
        trace_data_format = re.sub("(\w+)_(\d+)_(NCHW|NHWC).*", "\\3", trace_dir)

        trace_env_with_format = f"{trace_env}_{trace_data_format}"
        if not(trace_env_with_format in trace_environments):
            trace_environments[trace_env_with_format] = {"medians":[], "batch_sizes":[], "measurements":[]}

        trace_summary_path = os.path.join(trace_dir_full, "summary.json")
        with open(trace_summary_path, "r") as f:
            trace_summary = json.load(f)

        trace_environments[trace_env_with_format]["batch_sizes"].append(int(trace_batch_size))
        trace_environments[trace_env_with_format]["medians"].append(float(trace_summary["median"]))
        trace_environments[trace_env_with_format]["measurements"].append(trace_summary["measurements"])

# Plot median times
for trace_environment_name in trace_environments:
    trace_environment = trace_environments[trace_environment_name]
    medians, batch_sizes = zip(*sorted(zip(trace_environment["medians"], trace_environment["batch_sizes"])))
    line, = plt.plot(batch_sizes, medians)
    line.set_label(trace_environment_name)

    print(trace_environment_name)
    print(medians)

plt.xlabel("Batch Size")
plt.ylabel("ms")
plt.title("Training Step Time")
plt.legend()
plt.show()
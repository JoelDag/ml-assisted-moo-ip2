import re
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Groups result JSON files by parameter setup, averages metrics across seeds, and saves aggregated results.
input_dir = Path("../../src/runs/NSGA_3_paper_setup_run_20250728_143625")
agg_dir = Path(".") 

# match and extract parameter setups and group
pattern = re.compile(r'^(.*?_tp\d+_tf\d+_jut[0-9.]+)_seed\d+_.+\.json$')
grouped_files = defaultdict(list)

for file in input_dir.glob("*.json"):
    match = pattern.match(file.name)
    if match:
        grouped_files[match.group(1)].append(file)
        
for group, files in grouped_files.items():
    print(f"{group}:")
    for file in files:
        print(f"  {file.name}")

# Function to average lists element-wise
def average_lists(lists):
    min_len = min(len(lst) for lst in lists)
    trimmed = [lst[:min_len] for lst in lists]
    return list(np.mean(trimmed, axis=0))

# Process each group and save into corresponding *_agg.json
for param_setup, files in grouped_files.items():
    hv_ip2_all = []
    hv_nsga2_all = []
    igd_ip2_all = []
    igd_nsga2_all = []

    for file in files:
        with file.open() as f:
            data = json.load(f)
            hv_ip2_all.append(data["hv_ip2"])
            hv_nsga2_all.append(data["hv_nsga2"])
            igd_ip2_all.append(data["igd_ip2"])
            igd_nsga2_all.append(data["igd_nsga2"])

    # Compute averages
    avg_hv_ip2 = average_lists(hv_ip2_all)
    avg_hv_nsga2 = average_lists(hv_nsga2_all)
    avg_igd_ip2 = average_lists(igd_ip2_all)
    avg_igd_nsga2 = average_lists(igd_nsga2_all)

    # Determine output file name
    output_file = agg_dir / f"{param_setup}_nestNone_mdepthNone_agg.json"

    if output_file.exists():
        with output_file.open() as f:
            output_data = json.load(f)
    else:
        output_data = {"problem": param_setup}

    output_data.update({
        "hv_ip2": avg_hv_ip2,
        "hv_nsga2": avg_hv_nsga2,
        "igd_ip2": avg_igd_ip2,
        "igd_nsga2": avg_igd_nsga2,
        "problem": param_setup
    })

    with output_file.open("w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved averaged results to {output_file}")

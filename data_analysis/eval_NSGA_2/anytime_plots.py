import json, re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.stats import ttest_rel

input_dir = Path("../../src/runs/NSGA_2_paper_setup_run_20250728_143908")
output_dir = Path("new_plots")
output_dir.mkdir(exist_ok=True)
pattern = re.compile(r'^(.*?_tp\d+_tf\d+_jut[0-9.]+)_seed\d+_.+\.json$')

grouped_files = defaultdict(list)
for file in input_dir.glob("*.json"):
    match = pattern.match(file.name)
    if match:
        grouped_files[match.group(1)].append(file)

def get_convergence_index(series, mode="max", tol=0.05):
    start, end = series[0], series[-1]
    if mode == "max":
        thresh = start + (end - start) * (1 - tol)
        idx = np.argmax(series >= thresh)
    else:
        thresh = end + (start - end) * tol
        idx = np.argmax(series <= thresh)
    return idx

hv_diffs, igd_diffs = [], []
hv_names, igd_names = [], []
threshold = 10

for group, files in grouped_files.items():
    data = {k: [] for k in ["hv_nsga2", "hv_ip2", "igd_nsga2", "igd_ip2"]}
    for file in files:
        with open(file) as f:
            js = json.load(f)
            for k in data:
                data[k].append(js[k])

    for metric, mode in [("hv", "max"), ("igd", "min")]:
        nsga2 = [np.array(x) for x in data[f"{metric}_nsga2"]]
        ip2   = [np.array(x) for x in data[f"{metric}_ip2"]]
        max_len = max(max(len(x) for x in nsga2), max(len(x) for x in ip2))
        nsga2 = np.array([np.pad(x, (0, max_len - len(x)), 'edge') for x in nsga2])
        ip2   = np.array([np.pad(x, (0, max_len - len(x)), 'edge') for x in ip2])
        mean_nsga2 = nsga2.mean(0)
        mean_ip2   = ip2.mean(0)
        idx_nsga2 = get_convergence_index(mean_nsga2, mode=mode)
        idx_ip2   = get_convergence_index(mean_ip2, mode=mode)
        diff = idx_nsga2 - idx_ip2

        if metric == "hv":
            hv_diffs.append(diff)
            hv_names.append(group)
        else:
            igd_diffs.append(diff)
            igd_names.append(group)

        plt.figure()
        for run in nsga2: plt.plot(run, color="blue", alpha=0.15, lw=1)
        for run in ip2:   plt.plot(run, color="orange", alpha=0.15, lw=1)
        plt.plot(mean_nsga2, color="blue", lw=2, label="NSGA-II mean")
        plt.plot(mean_ip2,   color="orange", lw=2, label="NSGA-II+IP2 mean")
        plt.axvline(idx_nsga2, color="blue", ls="--", alpha=0.5, label="NSGA-II conv")
        plt.axvline(idx_ip2,   color="orange", ls="--", alpha=0.5, label="NSGA-II+IP2 conv")
        plt.title(f"{group} - {metric.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{group}_{metric}.png")
        plt.close()

# Analyze and print
hv_diffs = np.array(hv_diffs)
igd_diffs = np.array(igd_diffs)

print("\nSignificant convergence speed differences for HV:")
for d, name in zip(hv_diffs, hv_names):
    if abs(d) >= threshold:
        print(f"{name}: {'NSGA-II' if d<0 else 'NSGA-II+IP2'} converged {abs(d)} iterations faster in HV")
print("\nSignificant convergence speed differences for IGD:")
for d, name in zip(igd_diffs, igd_names):
    if abs(d) >= threshold:
        print(f"{name}: {'NSGA-II' if d<0 else 'NSGA-II+IP2'} converged {abs(d)} iterations faster in IGD")

# Statistical test
for diffs, names, label in [(hv_diffs, hv_names, "HV"), (igd_diffs, igd_names, "IGD")]:
    t_stat, p_val = ttest_rel(diffs, np.zeros_like(diffs))
    print(f"\n{label}: Paired t-test vs zero difference: p={p_val:.4f}")
    print(p_val)
    if p_val < 0.05:
        print(f"  Statistically significant: {'NSGA-II' if diffs.mean()<0 else 'NSGA-II+IP2'} converges faster on average in {label}.")
    else:
        print(f"  No statistically significant difference in {label} convergence speed.")


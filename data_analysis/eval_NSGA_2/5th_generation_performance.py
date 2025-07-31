import json, re
from pathlib import Path
from scipy.stats import wilcoxon
import numpy as np
import pandas as pd

"""
This averages the results from the json over all seeds
and prints a LaTeX table: mean ± std, bold winner (better value),
(h) next to the worse value to indicate significance: (1) if the difference
is significant, else (0). Higher HV is better; lower IGD is better.
Columns: IP2 first, then NSGA2, for both HV and IGD.
"""

# Adjust run_dir to where your JSON files are located
run_dir = Path('/local/joeldag/ml4moo_topic2/src/runs/NSGA_2_paper_setup_run_20250728_143908')
out_dir = Path('5th_generation_NSGA2')
out_dir.mkdir(exist_ok=True)

pattern = re.compile(
    r'(?P<problem>.+)_tp(?P<tp>\d+)_tf(?P<tf>\d+)_jut(?P<jut>[\d.]+)_seed\d+.*\.json$'
)

# Group files by problem/tp/tf/jut
groups = {}
for f in run_dir.glob('*.json'):
    m = pattern.match(f.name)
    if not m:
        continue
    key = (m['problem'], int(m['tp']), int(m['tf']), float(m['jut']))
    with f.open() as fp:
        groups.setdefault(key, []).append(json.load(fp))

for key, items in groups.items():
    problem, tp, tf, jut = key
    print(f"Problem: {problem}, tp: {tp}, tf: {tf}, jut: {jut}: {len(items)} runs")

rows = []
GEN_IDX = 99 # 4 for 5th generation, 10 for 11th

# Aggregate statistics and perform Wilcoxon tests
for key, runs in groups.items():
    hv_nsga2, hv_ip2, igd_nsga2, igd_ip2 = [], [], [], []
    for run in runs:
        try:
            hv_nsga2.append(run['hv_nsga2'][GEN_IDX+1])
            hv_ip2.append(run['hv_ip2'][GEN_IDX+1])
            igd_nsga2.append(run['igd_nsga2'][GEN_IDX])
            igd_ip2.append(run['igd_ip2'][GEN_IDX])
        except (KeyError, IndexError):
            continue

    # Compute means and standard deviations
    hv_nsga2_mean, hv_nsga2_std = np.mean(hv_nsga2), np.std(hv_nsga2)
    hv_ip2_mean, hv_ip2_std = np.mean(hv_ip2), np.std(hv_ip2)
    igd_nsga2_mean, igd_nsga2_std = np.mean(igd_nsga2), np.std(igd_nsga2)
    igd_ip2_mean, igd_ip2_std = np.mean(igd_ip2), np.std(igd_ip2)

    # Wilcoxon test for HV (higher is better)
    hv_h1 = hv_h2 = 0
    if len(hv_nsga2) == len(hv_ip2) and len(hv_nsga2) > 0:
        try:
            stat, pval = wilcoxon(hv_nsga2, hv_ip2, alternative='two-sided')
            if pval < 0.05:
                if hv_nsga2_mean > hv_ip2_mean:
                    hv_h1 = 1  # NSGA-II better
                else:
                    hv_h2 = 1  # IP2 better
        except ValueError:
            pass

    # Wilcoxon test for IGD (lower is better)
    igd_h1 = igd_h2 = 0
    if len(igd_nsga2) == len(igd_ip2) and len(igd_nsga2) > 0:
        try:
            stat, pval = wilcoxon(igd_nsga2, igd_ip2, alternative='two-sided')
            if pval < 0.05:
                if igd_nsga2_mean < igd_ip2_mean:
                    igd_h1 = 1  # NSGA-II better
                else:
                    igd_h2 = 1  # IP2 better
        except ValueError:
            pass

    rows.append({
        'problem': key[0], 'tp': key[1], 'tf': key[2], 'jut': key[3],
        'hv_nsga2_mean': hv_nsga2_mean, 'hv_nsga2_std': hv_nsga2_std, 'hv_nsga2_h': hv_h1,
        'hv_ip2_mean': hv_ip2_mean, 'hv_ip2_std': hv_ip2_std, 'hv_ip2_h': hv_h2,
        'igd_nsga2_mean': igd_nsga2_mean, 'igd_nsga2_std': igd_nsga2_std, 'igd_nsga2_h': igd_h1,
        'igd_ip2_mean': igd_ip2_mean, 'igd_ip2_std': igd_ip2_std, 'igd_ip2_h': igd_h2,
    })

# Create DataFrame and sort
result_df = pd.DataFrame(rows)
result_df = result_df.sort_values(by="problem", key=lambda col: col.str.lower())
print(result_df)

# Formatter: add (h) only to the worse value; bold the better value
def fmt(mean, std, h, is_worse, bold=False):
    s = f"{mean:.6f} $\\pm$ {std:.6f}"
    if is_worse:
        s += f" ({int(h)})"
    if bold:
        return f"\\textbf{{{s}}}"
    return s

# Build LaTeX table
lines = []
lines.append(r"\begin{table}[H]")
lines.append(r"\centering")
lines.append(r"\scriptsize")
lines.append(rf"\caption{{{GEN_IDX+1} generation Comparison of NSGA-II+IP$^2$ and NSGA-II (mean $\pm$ std (h)).}}")
lines.append(r"\makebox[\linewidth]{")
lines.append(r"\begin{tabular}{l|ll|ll}")
lines.append(r"\toprule")
lines.append(r"Problem & HV IP$^2$ & HV NSGA-II & IGD IP$^2$ & IGD NSGA-II \\")
lines.append(r"\midrule")

# Counters for the number of wins
hv_nsga2_better = 0
hv_ip2_better = 0
igd_nsga2_better = 0
igd_ip2_better = 0

for _, row in result_df.iterrows():
    prob = row['problem']
    # Determine winners: HV higher is better; IGD lower is better
    hv1_winner = row['hv_nsga2_mean'] >= row['hv_ip2_mean']  # True if NSGA-II wins or ties
    igd1_winner = row['igd_nsga2_mean'] <= row['igd_ip2_mean']  # True if NSGA-II wins or ties

    # Update win counters
    if hv1_winner:
        hv_nsga2_better += 1
    else:
        hv_ip2_better += 1
    if igd1_winner:
        igd_nsga2_better += 1
    else:
        igd_ip2_better += 1

    # Format HV column
    if hv1_winner:
        # NSGA-II wins – bold NSGA-II; annotate IP2 with NSGA-II’s significance flag
        hv_nsga2 = fmt(row['hv_nsga2_mean'], row['hv_nsga2_std'], h=0, is_worse=False, bold=True)
        hv_ip2   = fmt(row['hv_ip2_mean'], row['hv_ip2_std'], h=row['hv_nsga2_h'], is_worse=True, bold=False)
    else:
        # IP2 wins – bold IP2; annotate NSGA-II with IP2’s significance flag
        hv_ip2   = fmt(row['hv_ip2_mean'], row['hv_ip2_std'], h=0, is_worse=False, bold=True)
        hv_nsga2 = fmt(row['hv_nsga2_mean'], row['hv_nsga2_std'], h=row['hv_ip2_h'], is_worse=True, bold=False)

    # Format IGD column
    if igd1_winner:
        # NSGA-II wins – bold NSGA-II; annotate IP2 with NSGA-II’s significance flag
        igd_nsga2 = fmt(row['igd_nsga2_mean'], row['igd_nsga2_std'], h=0, is_worse=False, bold=True)
        igd_ip2   = fmt(row['igd_ip2_mean'], row['igd_ip2_std'], h=row['igd_nsga2_h'], is_worse=True, bold=False)
    else:
        # IP2 wins – bold IP2; annotate NSGA-II with IP2’s significance flag
        igd_ip2   = fmt(row['igd_ip2_mean'], row['igd_ip2_std'], h=0, is_worse=False, bold=True)
        igd_nsga2 = fmt(row['igd_nsga2_mean'], row['igd_nsga2_std'], h=row['igd_ip2_h'], is_worse=True, bold=False)

    lines.append(f"{prob} & {hv_ip2} & {hv_nsga2} & {igd_ip2} & {igd_nsga2} \\\\")

# Totals line
lines.append(r"\midrule")
lines.append(f"\\textbf{{Total}} & \\textbf{{{hv_ip2_better}}} & \\textbf{{{hv_nsga2_better}}} & "
             f"\\textbf{{{igd_ip2_better}}} & \\textbf{{{igd_nsga2_better}}} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}}")
lines.append(r"\end{table}")

# Create LaTeX table string and print
latex_table = "\n".join(lines)
print(latex_table)

# Optionally save to file
with open(out_dir / "summary_table.tex", "w") as fp:
    fp.write(latex_table)

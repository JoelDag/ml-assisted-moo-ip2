import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from glob import glob

result_dir = "/local/joeldag/ml4moo_topic2/data_analysis/aggregated_per_param"
json_files = glob(os.path.join(result_dir, "*.json"))

records = []

for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)

    basename = os.path.basename(file)
    parts = basename.replace(".json", "").split("_")

    test_problem = parts[0]
    t_past = int([p for p in parts if p.startswith("tp")][0][2:])
    t_freq = int([p for p in parts if p.startswith("tf")][0][2:])
    jutting = float([p for p in parts if p.startswith("jut")][0][3:])

    hv_ip2 = data.get("hv_ip2_final", None)

    if hv_ip2 is not None:
        records.append({
            "test_problem": test_problem,
            "t_past": t_past,
            "t_freq": t_freq,
            "jutting_param": jutting,
            "hv_ip2_final": hv_ip2
        })

df = pd.DataFrame(records)

selected_problems = ["makeMMF1Function", "makeMMF2Function", "makeMMF3Function", "makeMMF10Function", "makeMMF15Function"]
df = df[df["test_problem"].isin(selected_problems)]

plot_df = df[["jutting_param", "t_past", "t_freq", "hv_ip2_final"]].copy()
plot_df["test_problem"] = df["test_problem"]

plot_df = plot_df.round({"jutting_param": 2, "hv_ip2_final": 4})

import numpy as np
plot_df["log10_hv_ip2"] = np.log10(plot_df["hv_ip2_final"])
plot_df_vis = plot_df[["jutting_param", "t_past", "t_freq", "log10_hv_ip2", "test_problem"]]

# Plot
plt.figure(figsize=(12, 6))
parallel_coordinates(plot_df_vis, class_column="test_problem", colormap=plt.cm.Set2, linewidth=2.0)
plt.title("Parallel Coordinates Plot (with log-scaled HV)")
plt.ylabel("Raw or Log10 Value")
plt.xticks(rotation=30)
plt.grid(True)
plt.tight_layout()
plt.show()
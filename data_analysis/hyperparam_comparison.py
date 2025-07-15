import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

json_folder = "/local/joeldag/ml4moo_topic2/data_analysis/aggregated_per_param"
plot_save_folder = "hyperparam_tuning/plots"
os.makedirs(plot_save_folder, exist_ok=True)

def load_jsons(folder):
    data = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file)) as f:
                data.append(json.load(f))
    return pd.DataFrame(data)

df = load_jsons(json_folder)

df["hv_diff"] = df["hv_ip2_final"] - df["hv_nsga2_final"]
df["igd_diff"] = df["igd_nsga2_final"] - df["igd_ip2_final"]

metrics = ["hv_ip2_final", "igd_ip2_final"]
params = ["t_past", "t_freq", "jutting"]

for metric in metrics:
    for param in params:
        plt.figure(figsize=(6, 4))
        sns.barplot(x=param, y=metric, data=df, estimator="mean", ci="sd")
        plt.title(f"Mean {metric} by {param}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_folder, f"{metric}_by_{param}_bar.png"))
        plt.close()

pivot = df.pivot_table(index="t_past", columns="t_freq", values="hv_ip2_final", aggfunc="mean")
plt.figure(figsize=(6, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
plt.title("HV (IP2) by t_past vs t_freq")
plt.tight_layout()
plt.savefig(os.path.join(plot_save_folder, "heatmap_hv_ip2_tpast_tfreq.png"))
plt.close()

plt.figure(figsize=(6, 5))
sns.scatterplot(x="hv_diff", y="igd_diff", data=df, hue="t_past", palette="coolwarm")
plt.axvline(0, color="gray", linestyle="--")
plt.axhline(0, color="gray", linestyle="--")
plt.title("HV vs IGD Diff (IP2 - NSGA2)")
plt.xlabel("HV Diff (IP2 - NSGA2)")
plt.ylabel("IGD Diff (NSGA2 - IP2)")
plt.legend(title="t_past")
plt.tight_layout()
plt.savefig(os.path.join(plot_save_folder, "hv_vs_igd_diff_scatter.png"))
plt.close()

import json
import scipy.io
import matplotlib.pyplot as plt

"""
This compare pareo fronts, from test case with reference solution PF
"""
with open("/local/joeldag/ml4moo_topic2/src/runs/makeMMF10Function_tp5_tf5_jut1.1_seed0.json", "r") as f:
    data = json.load(f)

fronts = data["fronts_ip2"]  # cahgne with fronts_nsga2 also


mat_data = scipy.io.loadmat("/local/joeldag/ml4moo_topic2/Reference_PSPF_data/MMF1_Reference_PSPF_data.mat")
print(mat_data.keys())
ref_pf = mat_data["PF"]

gen_idx = -1 #select last generation
front = fronts[gen_idx]
gen_num = gen_idx % len(fronts)

x_vals = [obj[0] for obj in front]
y_vals = [obj[1] for obj in front]

plt.figure(figsize=(6, 5))
plt.scatter(x_vals, y_vals, s=30, alpha=0.7, label=f"Gen {gen_num} (IP2)")
plt.scatter(ref_pf[:, 0], ref_pf[:, 1], s=15, color='red', marker='x', label="Reference PF", alpha=0.4)
plt.title("Test Problem makeMMF10 Pareto Front vs. Reference")
plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"pareto_vs_ref_gen{gen_num}.png")
plt.show()

import json
import matplotlib.pyplot as plt
from pathlib import Path

agg_dir = Path(".")
agg_files = sorted(agg_dir.glob("*_agg.json"))

for agg_file in agg_files:
    with agg_file.open() as f:
        data = json.load(f)

    hv_nsga2 = data["hv_nsga2"]
    hv_ip2 = data["hv_ip2"]
    igd_nsga2 = data["igd_nsga2"]
    igd_ip2 = data["igd_ip2"]
    problem = data.get("problem", "unknown_problem")

    gen_hv = list(range(0, 100, 5)) + [99] if len(hv_nsga2) == 21 else list(range(0, len(hv_nsga2)))
    gen_igd = list(range(100)) if len(igd_nsga2) == 100 else list(range(len(igd_nsga2)))

    safe_name = problem.replace("/", "_")

    plt.figure()
    plt.plot(gen_hv, hv_nsga2, label="NSGA-III")
    plt.plot(gen_hv, hv_ip2, label="NSGA-III + IP2")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title(f"HV Anytime Performance\n{problem}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/hv_{safe_name}_anytime.png")
    print(f"Saved HV plot to plots/hv/{safe_name}_anytime.png")

    plt.figure()
    plt.plot(gen_igd, igd_nsga2, label="NSGA-III")
    plt.plot(gen_igd, igd_ip2, label="NSGA-III + IP2")
    plt.xlabel("Generation")
    plt.ylabel("IGD")
    plt.title(f"IGD Anytime Performance\n{problem}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/igd_{safe_name}_anytime.png")
    print(f"Saved IGD plot to plots/igd/{safe_name}_anytime.png")

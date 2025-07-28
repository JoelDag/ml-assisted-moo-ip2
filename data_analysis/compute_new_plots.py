import json
import matplotlib.pyplot as plt

from pathlib import Path

def plot(hv_nsga2, hv_ip2, igd_nsga2, igd_ip2, test_problem, algorithm='NSGA-II', job_id="default"):
    Path("plots/hv").mkdir(parents=True, exist_ok=True)
    Path("plots/igd").mkdir(parents=True, exist_ok=True)

    safe_name = test_problem.replace("/", "_")
    plt.figure()
    plt.plot(hv_nsga2, label=algorithm)
    plt.plot(hv_ip2, label=algorithm + " + IP2")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title(f"HV Performance on {job_id}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/hv/{safe_name}_{job_id}.png")
    print(f"plot saved to plots/hv/{safe_name}_{job_id}.png")

    plt.figure()
    plt.plot(igd_nsga2, label=algorithm)
    plt.plot(igd_ip2, label=algorithm + " + IP2")
    plt.xlabel("Generation")
    plt.ylabel("IGD")
    plt.title(f"IGD on {job_id}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/igd/{safe_name}_{job_id}.png")
    print(f"plot saved to plots/igd/{safe_name}_{job_id}.png")



agg_dir = Path("aggregated_per_param")

for file in agg_dir.glob("*.json"):
    with file.open() as f:
        data = json.load(f)

    hv_nsga2 = data["hv_nsga2"]
    hv_ip2 = data["hv_ip2"]
    igd_nsga2 = data["igd_nsga2"]
    igd_ip2 = data["igd_ip2"]
    test_problem = data["problem"]

    job_id = f"tp{data['t_past']}_tf{data['t_freq']}_jut{data['jutting']}"

    plot(hv_nsga2, hv_ip2, igd_nsga2, igd_ip2, test_problem, algorithm='NSGA-II', job_id=job_id)

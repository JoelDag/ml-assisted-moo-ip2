import os
import itertools
from pathlib import Path
from scipy.spatial.distance import cdist
import numpy as np
import scipy.io as sio
from deap import creator
import matplotlib.pyplot as plt

def replace_nan_with_column_mean(offspring):
    offspring = np.array(offspring, dtype=np.float64)
    col_means = np.nanmean(offspring, axis=0)  # mean ignoring NaNs
    # Replace NaNs with column means
    inds = np.where(np.isnan(offspring))
    offspring[inds] = np.take(col_means, inds[1])
    return [creator.Individual(ind.tolist()) for ind in offspring]


def generate_reference_vectors(m_obj, h_interval):
    ref_vectors = []
    for comb in itertools.combinations_with_replacement(range(h_interval + 1), m_obj):
        if sum(comb) == h_interval:
            vec = np.array(comb) / h_interval
            ref_vectors.append(vec)
    return np.array(ref_vectors)

# Metric Functions
def compute_igd(reference_set, approx_set):
    distances = cdist(reference_set, approx_set)
    return np.mean(np.min(distances, axis=1))

def compute_psp(true_ps_sets, approx_set, threshold=0.05):
    covered = 0
    for region in true_ps_sets:
        dists = cdist(region, approx_set)
        if np.any(np.min(dists, axis=1) < threshold):
            covered += 1
    return covered

def compute_hypervolume(front, ref_point):
    front = np.array(front)
    sorted_front = front[np.argsort(front[:, 0])]
    hv = 0.0
    prev_f2 = ref_point[1]
    for f1, f2 in sorted_front:
        hv += (ref_point[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return hv

def load_reference_pf(problem_name: str, evaluator) -> np.ndarray:
    DATA_DIR = Path(__file__).resolve().parent.parents[1]/"Reference_PSPF_data"
    problem_name = problem_name.replace("make", "").replace("Function", "")
    file_path = os.path.join(DATA_DIR, f"{problem_name}_Reference_PSPF_data.mat")
    mat = sio.loadmat(file_path)

    if "PF" in mat:
        return np.asarray(mat["PF"], dtype=float)
    if "PS" not in mat:
        raise KeyError(f"'PS' not found in reference file for {problem_name}")

    ps = np.asarray(mat["PS"], dtype=float)
    pf = np.vstack([evaluator(x) for x in ps])
    return pf

def plot(hv_nsga2, hv_ip2, igd_nsga2, igd_ip2, test_problem):
    if not os.path.exists("plots"):
        os.makedirs("plots", exist_ok=True)
        os.makedirs("plots/hv", exist_ok=True)
        os.makedirs("plots/igd", exist_ok=True)


    plt.plot(hv_nsga2, label="NSGA-II")
    plt.plot(hv_ip2, label="NSGA-II + IP2")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title(f"HV Performance on {test_problem}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/hv/{test_problem}.png")
    
    plt.figure()
    plt.plot(igd_nsga2, label="NSGA-II")
    plt.plot(igd_ip2, label="NSGA-II + IP2")
    plt.xlabel("Generation")
    plt.ylabel("IGD")
    plt.title(f"IGD on {test_problem}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/igd/{test_problem}.png")

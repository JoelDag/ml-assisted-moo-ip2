import os
import itertools
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pygmo as pg
import logging

from deap import creator
from pathlib import Path
from scipy.spatial.distance import cdist

def get_three_objectives_problems():
    return ["makeMMF14Function", "makeMMF14aFunction", "makeMMF15Function", "makeMMF15aFunction"]

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

def normalize_front(front, min, max):
    front = np.array(front)
    return (front - min) / (max - min)

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
    hv = pg.hypervolume(front)
    logging.debug("[HV DEBUG] Front shape: %s", front.shape)
    logging.debug("[HV DEBUG] Sample front (first 2 points): %s", front[:2])
    logging.debug("[HV DEBUG] Reference point: %s", ref_point)
    return hv.compute(ref_point)

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

def plot(hv_nsga2, hv_ip2, igd_nsga2, igd_ip2, test_problem, algorithm='NSGA-II', job_id="default"):
    os.makedirs("plots/hv", exist_ok=True)
    os.makedirs("plots/igd", exist_ok=True)
    # if not os.path.exists("plots_for_" + algorithm):
    #     os.makedirs("plots_for_" + algorithm, exist_ok=True)
    #     os.makedirs("plots_for_" + algorithm + "/hv", exist_ok=True)
    #     os.makedirs("plots_for_" + algorithm + "/igd", exist_ok=True)

    safe_name = test_problem.replace("/", "_")
    plt.figure()
    plt.plot(hv_nsga2, label=algorithm)
    plt.plot(hv_ip2, label=algorithm +" + IP2")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title(f"HV Performance on {test_problem}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/hv/{safe_name}_{job_id}.png")
    
    plt.figure()
    plt.plot(igd_nsga2, label=algorithm)
    plt.plot(igd_ip2, label=algorithm +" + IP2")
    plt.xlabel("Generation")
    plt.ylabel("IGD")
    plt.title(f"IGD on {test_problem}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/igd/{safe_name}_{job_id}.png")

def setup_logger(level=None):
    lvl = level or getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="LOG-%(levelname)s: %(message)s")
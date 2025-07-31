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
    # Generates reference vectors for multi-objective optimization based on the number of objectives and interval
    ref_vectors = []
    for comb in itertools.combinations_with_replacement(range(h_interval + 1), m_obj):
        if sum(comb) == h_interval:
            vec = np.array(comb) / h_interval  # Normalize the vector
            ref_vectors.append(vec)
    return np.array(ref_vectors)

def normalize_front(front, min, max):
    front = np.array(front)
    return (front - min) / (max - min)

# Metric Functions
def compute_igd(reference_set, approx_set):
    distances = cdist(reference_set, approx_set)
    return np.mean(np.min(distances, axis=1))

def compute_hypervolume(front, ref_point):
    # Computes the hypervolume of a Pareto front with respect to a reference point.
    hv = pg.hypervolume(front)
    logging.debug("[HV DEBUG] Front shape: %s", front.shape)
    logging.debug("[HV DEBUG] Sample front (first 2 points): %s", front[:2])
    logging.debug("[HV DEBUG] Reference point: %s", ref_point)
    return hv.compute(ref_point)

def load_reference_pf(problem_name: str, evaluator) -> np.ndarray:
    # Load the reference Pareto front for a given problem.
    DATA_DIR = Path(__file__).resolve().parent.parents[1]/"Reference_PSPF_data"
    problem_name = problem_name.replace("make", "").replace("Function", "")
    file_path = os.path.join(DATA_DIR, f"{problem_name}_Reference_PSPF_data.mat")
    mat = sio.loadmat(file_path) # Load MATLAB file

    if "PF" in mat:
        return np.asarray(mat["PF"], dtype=float)
    if "PS" not in mat:
        raise KeyError(f"'PS' not found in reference file for {problem_name}")

    ps = np.asarray(mat["PS"], dtype=float)
    pf = np.vstack([evaluator(x) for x in ps]) # Evaluate the Pareto set to get the Pareto front
    return pf

def plot(hv_nsga2, hv_ip2, igd_nsga2, igd_ip2, test_problem, mse, model_performance, algorithm='NSGA-II', job_id="default"):
    # Plots hypervolume, IGD, and MSE metrics for the evolutionary algorithm results.
    root = Path(os.getenv("RUN_OUTPUT_DIR", "."))
    (root / "plots/hv").mkdir(parents=True, exist_ok=True)
    (root / "plots/igd").mkdir(parents=True, exist_ok=True)

    if model_performance:
        # Plot Mean Squared Error (MSE) per feature over training steps.
        (root / "plots/mse").mkdir(parents=True, exist_ok=True)
        x_ticks = list(range(len(mse[0])))  # Number of training steps
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']  # Adjust if more features

        mse = np.array(mse)
        mse_norm = (mse - np.min(mse)) / (np.max(mse) - np.min(mse))  # Normalize MSE values
        plt.figure(figsize=(8, 5))
        for i in range(mse.shape[0]):
            plt.plot(x_ticks, mse_norm[i], label=f'Feature {i}', marker='o', color=colors[i % len(colors)])

        plt.xlabel('Training Step (every t_freq generations)')
        plt.ylabel("MSE")
        plt.title(f"MSE per Feature over Training Steps for {test_problem}")
        plt.legend()
        plt.grid(True)
        plt.savefig(root / f"plots/mse/{job_id}.png")

    # Plot hypervolume metrics
    safe_name = test_problem.replace("/", "_")
    plt.figure()
    plt.plot(hv_nsga2, label=algorithm)
    plt.plot(hv_ip2, label=algorithm +" + IP2")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title(f"HV Performance on {test_problem}")
    plt.legend()
    plt.grid(True)
    plt.savefig(root/f"plots/hv/{safe_name}_{job_id}.png")
    print(f"plot saved to {root}/plots/hv/{safe_name}_{job_id}.png")

    # Plot IGD metrics
    plt.figure()
    plt.plot(igd_nsga2, label=algorithm)
    plt.plot(igd_ip2, label=algorithm +" + IP2")
    plt.xlabel("Generation")
    plt.ylabel("IGD")
    plt.title(f"IGD on {test_problem}")
    plt.legend()
    plt.grid(True)
    plt.savefig(root/f"plots/igd/{safe_name}_{job_id}.png")
    print(f"plot saved to {root}/plots/igd/{safe_name}_{job_id}.png")

def setup_logger(level=None):
    lvl = level or getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="LOG-%(levelname)s: %(message)s")
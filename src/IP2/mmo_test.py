import os
import itertools

from scipy.spatial.distance import cdist
import numpy as np
import scipy.io as sio
from deap import tools
from integration import EvolutionaryAlgorithm, evaluate_population
import matplotlib.pyplot as plt


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
    file_path = os.path.join("Reference_PSPF_data", f"{problem_name}_Reference_PSPF_data.mat")
    mat = sio.loadmat(file_path)

    if "PF" in mat:
        return np.asarray(mat["PF"], dtype=float)
    if "PS" not in mat:
        raise KeyError(f"'PS' not found in reference file for {problem_name}")

    ps = np.asarray(mat["PS"], dtype=float)
    pf = np.vstack([evaluator(x) for x in ps])
    return pf

def main(pop_size, n_gen, n_var, m_obj, t_past, t_freq, test_problem, jutting_param, h_interval):
    ea = EvolutionaryAlgorithm(algo='NSGA2', n=n_var, m=m_obj, test_problem=test_problem)
    prob_token = test_problem.replace("make", "").replace("Function", "")
    ref_pf = load_reference_pf(prob_token, ea.problem.evaluate) 
    
    pop_nsga2 = ea.toolbox.population(n=pop_size)
    pop_ip2 = ea.toolbox.population(n=pop_size)

    pop_nsga2 = evaluate_population(ea.problem, pop_nsga2)
    pop_ip2 = evaluate_population(ea.problem, pop_ip2)
    A_t, T_t = [], None

    # Metric trackers
    hv_ip2, hv_nsga2 = [], []
    igd_ip2, igd_nsga2 = [], []
    ref_point = (1.1, 1.1)  # For HV
    ref_vectors = generate_reference_vectors(m_obj, h_interval)

    for t in range(n_gen):
        # NSGA-II
        pop_nsga2 = ea.nsga2_without_IP(pop_nsga2, n_var)
        front_nsga2 = tools.sortNondominated(pop_nsga2, pop_size, True)[0]
        hv_nsga2.append(compute_hypervolume([ind.fitness.values for ind in front_nsga2], ref_point))

        # NSGA-II with IP2
        pop_ip2, A_t, T_t = ea.NSGA2(ref_vectors,
                                     pop_ip2, A_t, T_t,
                                     t_past,
                                     t_freq, t, n_var, jutting_param)
        front_ip2 = tools.sortNondominated(pop_ip2, pop_size, True)[0]
        hv_ip2.append(compute_hypervolume([ind.fitness.values for ind in front_ip2], ref_point))
        
        igd_nsga2.append(compute_igd(ref_pf, [ind.fitness.values for ind in front_nsga2]))
        igd_ip2.append(compute_igd(ref_pf, [ind.fitness.values for ind in front_ip2]))

    plt.plot(hv_nsga2, label="NSGA-II")
    plt.plot(hv_ip2, label="NSGA-II + IP2")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title(f"HV Performance on {prob_token}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plt.plot(igd_nsga2, label="NSGA-II")
    plt.plot(igd_ip2, label="NSGA-II + IP2")
    plt.xlabel("Generation")
    plt.ylabel("IGD")
    plt.title(f"IGD on {prob_token}")
    plt.legend(); plt.grid(True); plt.show()

if __name__ == "__main__":
    main(pop_size=100,
         n_gen=200,
         n_var=2, m_obj=2, t_past=10,
         t_freq=5, test_problem="makeMMF1Function", jutting_param=1.1, h_interval=3)

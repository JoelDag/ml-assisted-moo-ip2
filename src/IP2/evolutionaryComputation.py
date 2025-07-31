import numpy as np
import random
import copy

from deap import tools
from .integration import EvolutionaryAlgorithm
from .utils import load_reference_pf, generate_reference_vectors, compute_hypervolume, compute_igd, plot, \
    get_three_objectives_problems, normalize_front


class evolutionaryRunner:
    def __init__(self, pop_size, n_gen, n_var, m_obj, t_past, t_freq, test_problem, jutting_param, h_interval, algorithm='NSGA3', seed=None, rf_params=None, model_performance=None):
        self.pop_size = pop_size
        self.n_gen = n_gen  # Number of generations
        self.n_var = n_var  # Number of variables
        self.m_obj = m_obj  # Number of objectives
        self.t_past = t_past # Number of past generations to consider
        self.t_freq = t_freq # Frequency of training
        self.test_problem = test_problem
        self.jutting_param = jutting_param # Jutting parameter for IP2
        self.h_interval = h_interval  # Interval for reference vector generation
        self.algorithm = algorithm
        self.seed = seed
        self.model_performance = model_performance # Flag for tracking model performance
        self.ea = EvolutionaryAlgorithm(algo=algorithm, n=n_var, m=m_obj, test_problem=test_problem) # Initialize the evolutionary algorithm
        self.ref_pf = load_reference_pf(self.test_problem, self.ea.problem.evaluate)  # Load reference Pareto front
        self.rf_params = rf_params or {}
        if test_problem in get_three_objectives_problems():
             self.ref_point = (1.1, 1.1, 1.1)  # For HV
        else:
            self.ref_point = (1.1, 1.1)
        self.ref_vectors = generate_reference_vectors(self.m_obj, self.h_interval)

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.init_pop = self.ea.toolbox.population(n=self.pop_size)
        for ind in self.init_pop:
            ind.fitness.values = self.ea.problem.evaluate(np.array(ind))

        self.job_id = f"{self.test_problem}_tp{self.t_past}_tf{self.t_freq}_jut{self.jutting_param}_seed{self.seed}"

    def run(self):
        A_t, T_t = [], None
        hv_ip2, hv_nsga2 = [], []
        igd_ip2, igd_nsga2 = [], []
        print(f"[{self.job_id}] Starting evolutionary run...")

        if self.algorithm == 'NSGA2' or self.algorithm == 'NSGA3':
            hv_nsga2, hv_ip2, igd_nsga2, igd_ip2, front_ip2, front_nsga2, history_fronts_ip2, history_fronts_nsga, mse = self.run_NSGA(hv_ip2, hv_nsga2, igd_ip2, igd_nsga2, A_t, T_t)
            history_fronts_ip2 = [[list(p) for p in gen] for gen in history_fronts_ip2]
            history_fronts_nsga = [[list(p) for p in gen] for gen in history_fronts_nsga]

        # Plot results
        plot(hv_nsga2, hv_ip2, igd_nsga2, igd_ip2, self.test_problem, mse, self.model_performance, algorithm=self.algorithm, job_id=self.job_id)
        final = {
            "hv_ip2_final": hv_ip2[-1] if hv_ip2 else None,
            "hv_nsga2_final": hv_nsga2[-1] if hv_nsga2 else None,
            "igd_ip2_final": igd_ip2[-1] if igd_ip2 else None,
            "igd_nsga2_final": igd_nsga2[-1] if igd_nsga2 else None,
            "hv_ip2": hv_ip2,
            "hv_nsga2": hv_nsga2,
            "igd_ip2": igd_ip2,
            "igd_nsga2": igd_nsga2,
            "fronts_ip2": history_fronts_ip2,
            "fronts_nsga2": history_fronts_nsga
        }
        print(f"[{self.job_id}] Finished plotting results.")
        return final

    def run_NSGA(self, hv_with_IP2, hv_without_IP2, igd_with_IP2, igd_without_IP2, A_t, T_t):
        pop_ip2 = copy.deepcopy(self.init_pop)
        pop_nsga2 = copy.deepcopy(self.init_pop)

        history_fronts_nsga = [] # History of fronts for NSGA-II
        history_fronts_ip2 = [] # History of fronts for NSGA-II IP2
        history_fronts_ip2.append([ind.fitness.values for ind in pop_ip2])
        history_fronts_nsga.append([ind.fitness.values for ind in pop_nsga2])
        mse_values = [[] for _ in range(self.n_var)] # MSE values for model performance

        for t in range(self.n_gen):
            # NSGA-II
            pop_nsga2 = self.ea.NSGA_without_IP(pop_nsga2, self.n_var, self.ref_vectors)
            front_nsga2 = tools.sortNondominated(pop_nsga2, self.pop_size, True)[0]
            # hv_without_IP2.append(compute_hypervolume([ind.fitness.values for ind in front_nsga2], self.ref_point))

            # NSGA-II with IP2
            pop_ip2, A_t, T_t = self.ea.NSGA(self.ref_vectors,
                                        pop_ip2, A_t, T_t,
                                        self.t_past,
                                        self.t_freq, t, self.n_var, self.jutting_param, mse_values, rf_params=self.rf_params, model_performance=self.model_performance)
            front_ip2 = tools.sortNondominated(pop_ip2, self.pop_size, True)[0]
            # hv_with_IP2.append(compute_hypervolume([ind.fitness.values for ind in front_ip2], self.ref_point))

            # Save fronts every 5 generations only in float32 to save memory
            history_fronts_nsga.append(np.array([ind.fitness.values for ind in front_nsga2], dtype=np.float32).tolist())
            history_fronts_ip2.append(np.array([ind.fitness.values for ind in front_ip2], dtype=np.float32).tolist())

            # Compute IGD metrics for both methods
            igd_without_IP2.append(compute_igd(self.ref_pf, [ind.fitness.values for ind in front_nsga2]))
            igd_with_IP2.append(compute_igd(self.ref_pf, [ind.fitness.values for ind in front_ip2]))
            print(f"[{self.job_id}] Generation {t + 1}/{self.n_gen} complete.")

        # Normalize fronts and compute hypervolume metrics
        all_points = np.array([
            obj for gen_front in history_fronts_nsga + history_fronts_ip2 for obj in gen_front
        ])
        ideal = np.min(all_points, axis=0)
        nadir = np.max(all_points, axis=0)

        for front in history_fronts_nsga:
            norm = normalize_front(front, ideal, nadir)
            hv = compute_hypervolume(norm, self.ref_point)
            hv_without_IP2.append(hv)

        for front in history_fronts_ip2:
            norm = normalize_front(front, ideal, nadir)
            hv = compute_hypervolume(norm, self.ref_point)
            hv_with_IP2.append(hv)

        return hv_with_IP2, hv_without_IP2, igd_with_IP2, igd_without_IP2, front_ip2, front_nsga2, history_fronts_ip2, history_fronts_nsga, mse_values
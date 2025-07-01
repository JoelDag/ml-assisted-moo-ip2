from .integration import EvolutionaryAlgorithm
from .utils import load_reference_pf, generate_reference_vectors, compute_hypervolume, compute_igd, plot
from deap import tools
import numpy as np
import itertools


class evolutionaryRunner:
    def __init__(self, pop_size, n_gen, n_var, m_obj, t_past, t_freq, test_problem, jutting_param, h_interval, algorithm='NSGA2'):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_var = n_var
        self.m_obj = m_obj
        self.t_past = t_past
        self.t_freq = t_freq
        self.test_problem = test_problem
        self.jutting_param = jutting_param
        self.h_interval = h_interval
        self.algorithm = algorithm
        self.ea = EvolutionaryAlgorithm(algo=algorithm, n=n_var, m=m_obj, test_problem=test_problem)
        self.ref_pf = load_reference_pf(self.test_problem, self.ea.problem.evaluate) 
        self.ref_point = (1.1, 1.1)  # For HV
        self.ref_vectors = generate_reference_vectors(self.m_obj, self.h_interval)
        self.init_pop = self.ea.toolbox.population(n=self.pop_size)
        for ind in self.init_pop:
            ind.fitness.values = self.ea.problem.evaluate(np.array(ind))
#        self.init_pop = self.ea.problem.evaluate(np.array(self.init_pop))

    def run(self):
        A_t, T_t = [], None
        hv_ip2, hv_nsga2 = [], []
        igd_ip2, igd_nsga2 = [], []

        if self.algorithm == 'NSGA2':
            hv_nsga2, hv_ip2, igd_nsga2, igd_ip2, front_ip2, front_nsga2 = self.run_NSGA2(hv_ip2, hv_nsga2, igd_ip2, igd_nsga2, A_t, T_t)


        plot(hv_nsga2, hv_ip2, igd_nsga2, igd_ip2, self.test_problem)
        return [front_ip2, front_nsga2]

    def run_NSGA2(self, hv_with_IP2, hv_without_IP2, igd_with_IP2, igd_without_IP2, A_t, T_t):
        pop_ip2, pop_nsga2 = self.init_pop, self.init_pop
        for t in range(self.n_gen):
            # NSGA-II
            pop_nsga2 = self.ea.NSGA2_without_IP(pop_nsga2, self.n_var)
            front_nsga2 = tools.sortNondominated(pop_nsga2, self.pop_size, True)[0]
            hv_without_IP2.append(compute_hypervolume([ind.fitness.values for ind in front_nsga2], self.ref_point))

            # NSGA-II with IP2
            pop_ip2, A_t, T_t = self.ea.NSGA2(self.ref_vectors,
                                        pop_ip2, A_t, T_t,
                                        self.t_past,
                                        self.t_freq, t, self.n_var, self.jutting_param)
            front_ip2 = tools.sortNondominated(pop_ip2, self.pop_size, True)[0]
            hv_with_IP2.append(compute_hypervolume([ind.fitness.values for ind in front_ip2], self.ref_point))

            igd_with_IP2.append(compute_igd(self.ref_pf, [ind.fitness.values for ind in front_nsga2]))
            igd_without_IP2.append(compute_igd(self.ref_pf, [ind.fitness.values for ind in front_ip2]))

        return hv_with_IP2, hv_without_IP2, igd_with_IP2, igd_without_IP2, front_ip2, front_nsga2
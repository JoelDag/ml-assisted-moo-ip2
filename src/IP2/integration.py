from functools import partial
from rpy2 import robjects
import numpy as np
from input_archive import update_target_archive, archive_mapping
from ml_training_module import training, progress
from src.MMFProblem.mmf import MMFfunction
from deap import base, creator, tools, algorithms
import random
from rpy2.robjects.packages import importr
from src.IP2.utils import replace_nan_with_column_mean

smoof = importr('smoof')

class EvolutionaryAlgorithm:
    def __init__(self, algo, n, m, test_problem):
        self.algo = algo
        self.toolbox = base.Toolbox()
        self.m = m # Number of objectives
        self.n = n  # Number of decision variables
        robjects.r[test_problem]()
        self.problem = MMFfunction(test_problem)
        self.history_P = []
        self.history_Q = []


        if algo not in ['NSGA2', 'NSGA3']:
            raise ValueError("Unsupported algorithm. Choose 'NSGA2' or 'NSGA3'.")

        self._setup_deap()

    def eval_pymoo(self,individual):
        X = np.asarray(individual, dtype=float).reshape(1, -1)
        F = self.problem.evaluate(X)
        return tuple(F[0])

    def bounded_polynomial_mutation_safe(self, ind, eta, xl, xu, indpb, **kwargs):
        for i in range(len(ind)):
            if np.random.rand() < indpb:
                ind[i], = tools.mutPolynomialBounded(
                    [ind[i]],
                    eta=eta,
                    low=xl[i],
                    up=xu[i],
                    indpb=indpb
                )
        return ind,

    def bounded_sbx(self, ind1, ind2, eta, xl, xu, **kwargs):
        for i in range(len(ind1)):
            tools.cxSimulatedBinaryBounded(ind1, ind2, eta=eta, low=xl[i], up=xu[i])
        return ind1, ind2


    def _setup_deap(self):
        # Erzeuge DEAP Typen
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * self.m)
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Register toolbox components
        self.toolbox.register("attr_float", random.uniform, float(self.problem.xl[0]), float(self.problem.xu[0]))
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.n)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", partial(self.bounded_sbx, eta=10, xl=self.problem.xl, xu=self.problem.xu))
        self.toolbox.register("mutate", partial(self.bounded_polynomial_mutation_safe, eta=20, xl=self.problem.xl, xu=self.problem.xu, indpb=1.0 / self.n))

        if self.algo == 'NSGA2':
            self.toolbox.register("select", tools.selNSGA2)
        elif self.algo == 'NSGA3':
            self.toolbox.register("select", tools.selNSGA3)

        self.toolbox.register("evaluate", self.eval_pymoo)

    def NSGA2(self, R, P_t, A_t, T_t1, t_past, t_freq, t, n, jutting_param):
        self.history_P.append(P_t)
        T_t = update_target_archive(P_t, T_t1, R)
        count = t % t_freq
        if count == 0:
            D_t = archive_mapping(A_t, T_t, R)
            predict, x_min, x_max = training(D_t, self.problem.xl, self.problem.xu)
        Q_t = algorithms.varAnd(P_t, self.toolbox, cxpb=0.9, mutpb=1.0 / n)
        for item in Q_t:
            for j, i in enumerate(item):
                if isinstance(i, list):
                    item[j] = i[0]
        Q_t = replace_nan_with_column_mean(Q_t)  # Ensure no NaNs in offspring
        if count == 0:
            Q_t = progress(Q_t, jutting_param, x_min, x_max, self.problem.xl, self.problem.xu, predict)
        self.history_Q.append(Q_t)
        Q_t = self.toolbox.evaluate(np.array(Q_t))

        A_t1 = A_t + P_t + Q_t
        if len(A_t1) > t_past * len(P_t):
            A_t1 = A_t1[-(t_past * len(P_t)):]

        P_t1 = self.toolbox.select(P_t + Q_t, len(P_t))

        return P_t1, A_t1, T_t


    def NSGA3(self, R, P_t, A_t, T_t1, x_l, x_u, t_past, t_freq, t, n):
        T_t = update_target_archive(P_t, T_t1, R)
        count = t % t_freq
        if count == 0:
            D_t = archive_mapping(A_t, T_t, R)
            predict, x_min, x_max = training(D_t, x_l, x_u)
        offspring = algorithms.varAnd(P_t, self.toolbox, cxpb=1.0, mutpb=1.0)
        if count == 0:
            Q_t = progress(offspring, 1.1, x_min, x_max, x_l, x_u, predict)
        Q_t = self.toolbox.evaluate(np.array(Q_t))
        try:
            A_t1 = list(set(A_t).union(Q_t, self.history_P[t+1-t_past]) - set(self.history_P[t-t_past] - (set(self.history_Q[t-t_past]))))
        except ValueError:
            raise ValueError("Not enough history available for past t_past iterations.")
        P_t1 = self.toolbox.select(P_t + offspring, len(P_t), ref_points=R)
        # return P_t1, A_t1, T_t

    def NSGA2_without_IP(self, pop, n):
        offspring = algorithms.varAnd(pop, self.toolbox, cxpb=0.9, mutpb=1.0 / n)
        for item in offspring:
            for j, i in enumerate(item):
                if isinstance(i, list):
                    item[j] = i[0]
        offspring = replace_nan_with_column_mean(offspring)  # Ensure no NaNs in offspring
        for ind in offspring:
            ind.fitness.values = self.toolbox.evaluate(np.array(ind))
        pop = self.toolbox.select(pop + offspring, len(pop))
        return pop
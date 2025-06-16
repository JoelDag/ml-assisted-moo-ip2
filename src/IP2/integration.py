import numpy as np
from deap.benchmarks import zdt2, zdt1

from input_archive import update_target_archive, archive_mapping
from ml_training_module import training, progress
from deap import base, creator, tools, algorithms
import random

def evaluate_population(Qt):
    for ind in Qt:
        ind.fitness.values = zdt2(ind)
        if any(np.isnan(ind.fitness.values)):
            ind.fitness.values = (1e6, 1e6)
    return Qt

class EvolutionaryAlgorithm:
    def __init__(self, algo, n, m):
        self.algo = algo
        self.toolbox = base.Toolbox()
        self.m = m # Number of objectives
        self.n = n  # Number of decision variables
        self.history_P = []
        self.history_Q = []


        if algo not in ['NSGA2', 'NSGA3']:
            raise ValueError("Unsupported algorithm. Choose 'NSGA2' or 'NSGA3'.")

        self._setup_deap()

    def _setup_deap(self):
        # Erzeuge DEAP Typen
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * self.m)
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Register toolbox components
        self.toolbox.register("attr_float", random.random)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.n)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=10, low=0.0, up=1.0)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, eta=20, low=0.0, up=1.0, indpb=1 / self.n)
        if self.algo == 'NSGA2':
            self.toolbox.register("select", tools.selNSGA2)
        elif self.algo == 'NSGA3':
            self.toolbox.register("select", tools.selNSGA3)
        # self.toolbox.register("evaluate", self.evaluate)

    def NSGA2(self, R, P_t, A_t, T_t1, x_l, x_u, t_past, t_freq, t, n):
        self.history_P.append(P_t)
        T_t = update_target_archive(P_t, T_t1, R)
        count = t % t_freq
        if count == 0:
            D_t = archive_mapping(A_t, T_t, R)
            predict, x_min, x_max = training(D_t, x_l, x_u)
        Q_t = algorithms.varAnd(P_t, self.toolbox, cxpb=1.0, mutpb=1.0)
        if count == 0:
            Q_t = progress(Q_t, n, x_min, x_max, x_l, x_u, predict)
            self.history_Q.append(Q_t)
        Q_t = evaluate_population(Q_t)
        try:
            A_t1 = list(set(A_t).union(Q_t, self.history_P[t+1-t_past]) - set(self.history_P[t-t_past] - (set(self.history_Q[t-t_past]))))
        except ValueError:
            raise ValueError("Not enough history available for past t_past iterations.")

        # A_t1 = A_t + P_t + Q_t
        # if len(A_t1) > t_past * len(P_t):
        #     A_t1 = A_t1[-(t_past * len(P_t)):]

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
            Q_t = progress(offspring, n, x_min, x_max, x_l, x_u, predict)
        self.toolbox.evaluate(offspring)
    #    Evaluate(Q_t)    TODO
        try:
            A_t1 = list(set(A_t).union(Q_t, self.history_P[t+1-t_past]) - set(self.history_P[t-t_past] - (set(self.history_Q[t-t_past]))))
        except ValueError:
            raise ValueError("Not enough history available for past t_past iterations.")
        P_t1 = self.toolbox.select(P_t + offspring, len(P_t), ref_points=R)
        # return P_t1, A_t1, T_t

    def nsga2_without_IP(self, pop):
        offspring = algorithms.varAnd(pop, self.toolbox, cxpb=1.0, mutpb=1.0)
        for ind in offspring:
            ind.fitness.values = zdt2(ind)
        pop = self.toolbox.select(pop + offspring, len(pop))
        return pop
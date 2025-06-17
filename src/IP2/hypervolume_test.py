import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from integration import EvolutionaryAlgorithm
from deap.benchmarks import zdt2, zdt1

def hypervolume(front, ref_point):
    front = np.array(front)
    sorted_front = front[np.argsort(front[:, 0])]
    hv = 0.0
    prev_f2 = ref_point[1]
    for f1, f2 in sorted_front:
        hv += (ref_point[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return hv

pop_size = 100
n_gen = 100
n_var = 10
xl = np.zeros(n_var)
xu = np.ones(n_var)
ref_point = (1.1, 1.1) # For HV
ref_vectors = [np.array([i/99, 1 - i/99]) for i in range(100)]

ea = EvolutionaryAlgorithm(algo='NSGA2', n=n_var, m=2)

# Initialize both populations
pop_nsga2 = ea.toolbox.population(n=pop_size)
pop_ip2 = ea.toolbox.population(n=pop_size)

# Evaluate both populations initially
for ind in pop_nsga2 + pop_ip2:
    ind.fitness.values = zdt2(ind)

# Prepare logging variables
hv_nsga2 = []
hv_ip2 = []
A_t, T_t = [], None

# Evolution loop
for t in range(n_gen):
    # --- NSGA-II
    pop_nsga2 = ea.nsga2_without_IP(pop_nsga2)
    front_nsga2 = tools.sortNondominated(pop_nsga2, pop_size, True)[0]
    hv_nsga2.append(hypervolume([ind.fitness.values for ind in front_nsga2], ref_point))

    # --- NSGA-II + IP2 version ---
    pop_ip2, A_t, T_t = ea.NSGA2(ref_vectors,
                                 pop_ip2, A_t, T_t, xl, xu,
                                 t_past=5,
                                 t_freq=5, t = t, n=n_var
                                 )
    front_ip2 = tools.sortNondominated(pop_ip2, pop_size, True)[0]
    hv_ip2.append(hypervolume([ind.fitness.values for ind in front_ip2], ref_point))


plt.plot(hv_nsga2, label="NSGA-II")
plt.plot(hv_ip2, label="NSGA-II + IP2")
plt.xlabel("Generation")
plt.ylabel("Hypervolume")
plt.title("Performance on ZDT2")
plt.legend()
plt.grid(True)
plt.show()


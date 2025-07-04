import numpy as np

def initialise_target_archive(P_t, R):
    T_t = []
    for rv in R:
        best = None
        best_val = float('inf')
        for ind in P_t:
            val = asf(np.array(ind.fitness.values), rv)
            if val < best_val:
                best = ind
                best_val = val
        T_t.append(best)
    return T_t

def update_target_archive(P_t, T_prev, R):
    if T_prev is None or len(T_prev) == 0:
        # First generation → create targets directly from current parents
        return initialise_target_archive(P_t, R)
    N = len(R)
    
    F_P  = np.array([sol.fitness.values for sol in P_t])
    F_T = np.array([sol.fitness.values for sol in T_prev])

    z_ideal = F_P.min(axis=0)
    z_nadir = F_P.max(axis=0)

    F_P_norm = normalize_objectives(F_P, z_ideal, z_nadir)
    F_T_norm = normalize_objectives(F_T, z_ideal, z_nadir)

    T_new = T_prev.copy()
    for i, fp in enumerate(F_P_norm):   
        vals = [asf(fp, rv) for rv in R] 
        min_j = int(np.argmin(vals))
        if vals[min_j] < asf(F_T_norm[min_j], R[min_j]):   # is parent better for that RV?
            T_new[min_j] = P_t[i]
        
    # debug
    #repl = sum(id(a) != id(b) for a, b in zip(T_prev, T_new))
    #print(f"[Archive] replaced={repl} / {len(R)}")
    return T_new


def archive_mapping(A_t, T_t, R):
    if len(A_t) == 0 or len(T_t) == 0:
        return []

    F_A = np.array([sol.fitness.values for sol in A_t])
    
    z_ideal = F_A.min(0)
    z_nadir = F_A.max(0)

    F_A_norm = normalize_objectives(F_A, z_ideal, z_nadir)
    D_t = []

    for i, fa in enumerate(F_A_norm):
        values = [asf(fa, rv) for rv in R]
        min_j = int(np.argmin(values))
        D_t.append((A_t[i], T_t[min_j]))   

    return D_t

def normalize_objectives(F_vals, z_ideal, z_nadir):
    return (F_vals - z_ideal) / (z_nadir - z_ideal + 1e-12)


def asf(solution_f, ref_vector):
    ref_vector = np.where(ref_vector == 0, 1e-12, ref_vector)
    return np.max(solution_f / ref_vector)
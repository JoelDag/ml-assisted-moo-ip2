import numpy as np

def update_target_archive(P_t, T_t1, R):
    N = len(R)
    
    F_P = np.array([sol.f for sol in P_t])
    F_T = np.array([sol.f for sol in T_t_minus_1])

    z_ideal = np.min(F_P, axis=0)   # ideal and nadir points
    z_nadir = np.max(F_P, axis=0)

    F_P_norm = normalize_objectives(F_P, z_ideal, z_nadir)
    F_T_norm = normalize_objectives(F_T, z_ideal, z_nadir)

    T_t = T_t1.copy()
    for i in range(N):
        metrics = np.zeros(N)
        for j in range(N):
            metrics[j] = asf(F_P_norm[i], R[j])
            
        min_j = np.argmin(metrics)
        V_P = metrics[min_j]
        V_T = asf(F_T_norm[min_j], R[min_j])

        if V_P < V_T:
            T_t[min_j] = P_t[i]
        
    return T_t


def archive_mapping(A_t, T_t, R):
    N = len(R)
    N_A = len(A_t)
    F_A = np.array([sol.f for sol in A_t])
    
    z_ideal = np.min(F_A, axis=0)   # ideal and nadir points
    z_nadir = np.max(F_A, axis=0)

    F_A_norm = normalize_objectives(F_A, z_ideal, z_nadir)
    D_t = []

    for i in range(N_A):
        metrics = np.zeros(N)
        for j in range(N):
            metrics[j] = asf(F_A_norm[i], R[j])
    
        min_j = np.argmin(metrics)
        x_input = A_t[i].x
        x_target = T_t[min_j].x
        
        D_t.append((x_input, x_target))
    
    return D_t


def normalize_objectives(F, z_ideal, z_nadir):
    return (F - z_ideal) / (z_nadir - z_ideal + 1e-12)

def asf(solution_f, ref_vector):
    ref_vector = np.where(ref_vector == 0, 1e-12, ref_vector)
    return np.max(solution_f / ref_vector)
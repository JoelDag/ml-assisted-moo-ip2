from input_archive import update_target_archive, archive_mapping
from ml_training_module import training, progress


def NSGA2(R, P_t, A_t, T_t1, x_l, x_u, t_past, t_freq, t, n):   
    T_t = update_target_archive(P_t, T_t1, R)
    count = t% t_freq
    if count == 0:
        D_t = archive_mapping(A_t, T_t, R)
        predict, x_min, x_max = training(D_t, x_l, x_u)
#    Q_t = Crossover+ Mutation(P_t)    TODO
    if count == 0:
        Q_t = progress(Q_t, n, x_min, x_max, x_l, x_u, predict)
#    Evaluate(Q_t, )    TODO
#    A_t1 = (A_t + Q_t + P_t1tpast) - (P_ttpast + Q_ttpast)
#    P_t1 = SurvivalSelection of NSGA2 with P_t + Q_t
    return P_t1, A_t1, T_t


def NSGA3(R, P_t, A_t, T_t1, x_l, x_u, t_past, t_freq, t, n):   
    T_t = update_target_archive(P_t, T_t1, R)
    count = t% t_freq
    if count == 0:
        D_t = archive_mapping(A_t, T_t, R)
        predict, x_min, x_max = training(D_t, x_l, x_u)
#    Q_t = Crossover+ Mutation(P_t)    TODO
    if count == 0:
        Q_t = progress(Q_t, n, x_min, x_max, x_l, x_u, predict)
#    Evaluate(Q_t, )    TODO
#    A_t1 = (A_t + Q_t + P_t1tpast) - (P_ttpast + Q_ttpast)
#    P_t1 = SurvivalSelection of NSGA3 with P_t + Q_t
    return P_t1, A_t1, T_t
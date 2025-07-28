from sklearn.ensemble import RandomForestRegressor
import random
import numpy as np
from deap import creator

def training(d_t, x_l, x_u, rf_params=None):
    if len(d_t) == 0:
        return None, None, None
    input_vec, target_vec = zip(*d_t)
    n_samples, n_features = np.array(input_vec).shape
    
    if rf_params is None:
        rf_params = {}
    if rf_params.get("n_estimators") is None:
        rf_params["n_estimators"] = n_samples
    rf_params.setdefault("max_depth", None)
    rf_params.setdefault("max_features", n_features)
    rf_params.setdefault("criterion", "squared_error")
    rf_params.setdefault("random_state", 42)

    print(f"[RF-TRAINING] Creating RandomForestRegressor with params: {rf_params}")

    input_vec = np.array(input_vec)
    target_vec = np.array(target_vec)

    x_lt = np.minimum(np.nanmin(input_vec,axis=0), np.nanmin(target_vec,axis=0))
    x_ut = np.maximum(np.nanmax(input_vec,axis=0), np.nanmax(target_vec,axis=0))

    x_min = []
    x_max = []

    for k in range(n_features):
        x_min.append(0.5 * (x_lt[k] + x_l[k]))
        x_max.append(0.5 * (x_ut[k] + x_u[k]))

    # normalise d_t using x_min and x_max
    input_vec_normalized = np.array(
        [[(d[i] - x_min[i]) / (x_max[i] - x_min[i]) for i in range(n_features)] for d in input_vec])
    target_vec_normalized = np.array(
        [[(d[i] - x_min[i]) / (x_max[i] - x_min[i]) for i in range(n_features)] for d in target_vec])
            
    models = []
    for i in range(n_features):
        model = RandomForestRegressor(**rf_params)
        try:
            model.fit(input_vec_normalized, target_vec_normalized[:, i])

        except ValueError as e:
            print(f"Error fitting model for feature {target_vec_normalized}: {e}")
            continue
        models.append(model)

    def predict(x_normalized):
        return [m.predict([x_normalized])[0] for m in models]

    return predict, x_min, x_max


def progress(Q_t, n, x_min, x_max, x_l, x_u, predict):
    if predict is None:
        return Q_t
    indices = random.sample(range(len(Q_t)), len(Q_t) // 2)   # select 50% of offspring
    repaired_offspring = []

    for idx in indices:
        initial_offspring = Q_t[idx]
        # Normalize selected offspring using x_min and x_max
        normalized_offspring = [(initial_offspring[i] - x_min[i]) / (x_max[i] - x_min[i]) for i in
                                range(len(initial_offspring))]

        predicted_offspring = predict(normalized_offspring)  # Predict using the trained model

        denormalized_predictions = [predicted_offspring[i] * (x_max[i] - x_min[i]) + x_min[i] for i in
                                    range(len(predicted_offspring))]

        for k in range(len(denormalized_predictions)):
            tolerance = 0.01 * (x_u[k] - x_l[k])
            if initial_offspring[k] <= x_l[k] + tolerance or initial_offspring[k] >= x_u[k] - tolerance:
                denormalized_predictions[k] = initial_offspring[k]


        jutted_offspring = np.array(initial_offspring) + n * (
                np.array(denormalized_predictions) - np.array(initial_offspring))

        repaired_offspring.append(creator.Individual(parabolic_repair(initial_offspring, jutted_offspring, x_l, x_u)))

    # clip the repaired offspring to the bounds
    repaired_offspring = [creator.Individual(ind) for ind in np.clip(repaired_offspring, x_l, x_u)]

    for i, idx in enumerate(indices):
        Q_t[idx] = repaired_offspring[i]

    return Q_t

def parabolic_repair(initial_offspring, jutted_offspring, x_l, x_u, alpha=1.2):
    direction_vec = jutted_offspring - initial_offspring
    repaired = False

    for i in range(len(jutted_offspring)):
        if direction_vec[i] == 0:
            continue  # no movement in this dimension, skip

        # Check lower bound violation
        if jutted_offspring[i] < x_l[i]:
            t_l = (x_l[i] - initial_offspring[i]) / direction_vec[i]
            t_u = (x_u[i] - initial_offspring[i]) / direction_vec[i]
            repaired = True
            break

        # Check upper bound violation
        elif jutted_offspring[i] > x_u[i]:
            t_l = (x_u[i] - initial_offspring[i]) / direction_vec[i]
            t_u = (x_l[i] - initial_offspring[i]) / direction_vec[i]
            repaired = True
            break

    if not repaired:
        return jutted_offspring  # no repair needed

    # intersection point calculation
    X1 = initial_offspring + t_l * direction_vec
    X2 = initial_offspring + t_u * direction_vec

    # Compute distances
    d = np.linalg.norm(jutted_offspring - X1)

    r = np.random.uniform(0, 1)
    tan_arg = (np.linalg.norm(X2)) / (alpha * d)
    X_prime = d + alpha * d * np.tan(r * np.arctan(tan_arg))

    unit_vec = direction_vec / np.linalg.norm(direction_vec)  # unit vector in the direction of movement

    return X1 + X_prime * unit_vec  # new vector at dist X_prime from X1 in the direction of movement

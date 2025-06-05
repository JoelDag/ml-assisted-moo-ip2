from sklearn.ensemble import RandomForestRegressor
import random
import numpy as np


def training(d_t, x_l, x_u):
    input_vec, target_vec = zip(*d_t)
    n_samples, n_features = np.array(input_vec).shape

    input_vec = np.array(input_vec)
    target_vec = np.array(target_vec)

    x_lt = np.minimum(input_vec.min(axis=0), target_vec.min(axis=0))
    x_ut = np.maximum(input_vec.max(axis=0), target_vec.max(axis=0))

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
        model = RandomForestRegressor(n_estimators=n_samples,
                                      max_features=n_features,
                                      criterion="squared_error")
        model.fit(input_vec_normalized, target_vec_normalized[:, i])
        models.append(model)

    def predict(x_normalized):
        return [m.predict([x_normalized])[0] for m in models]

    return predict, x_min, x_max


def progress(Q_t, n, x_min, x_max, x_l, x_u, predict):
    selected_offspring = random.sample(Q_t, len(Q_t) // 2)  # Randomly selected 50% of offspring
    repaired_offspring = []

    for initial_offspring in selected_offspring:
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

        repaired_offspring.append(parabolic_repair(initial_offspring, jutted_offspring, x_l, x_u))

    return repaired_offspring


def parabolic_repair(initial_offspring, jutted_offspring, x_l, x_u, alpha=1.2):
    return jutted_offspring  # Placeholder for the actual repair logic

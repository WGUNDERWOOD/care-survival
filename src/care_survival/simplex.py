import numpy as np
import itertools

from care_survival import metrics as care_metrics


class ConvexEstimator:
    def __init__(self, kernel_estimator, theta):
        self.kernel_estimator = kernel_estimator
        self.theta = theta
        self.simplex_dimension = len(theta)
        self.f_check = {}
        for split in care_metrics.get_splits():
            self.f_check[split] = self.get_f_check_split(split)
        self.score = self.get_score()

    def get_f_check_split(self, split):
        theta = self.theta
        theta_0 = 1.0 - np.sum(theta)
        f_check = theta_0 * self.kernel_estimator.f_hat[split]
        embedding_data = self.kernel_estimator.embedding.data
        for i in range(self.simplex_dimension):
            f_check = f_check + theta[i] * embedding_data[split].f_tilde[:, i]
        return f_check

    def get_score(self):
        embedding = self.kernel_estimator.embedding
        f = {}
        for split in care_metrics.get_splits():
            f[split] = self.get_f_check_split(split)

        score = {}
        for metric in care_metrics.get_metrics():
            score[metric] = {}
            for split in care_metrics.get_splits():
                score[metric][split] = care_metrics.get_metric_split(
                    f[split], embedding, metric, split
                )
        return score


class SimplexSelection:
    def __init__(self, kernel_estimator, simplex_resolution):
        embedding_data = kernel_estimator.embedding.data
        self.kernel_estimator = kernel_estimator
        self.simplex_dimension = np.shape(embedding_data["train"].f_tilde)[1]
        self.simplex_resolution = simplex_resolution
        self.thetas = get_simplex(self.simplex_dimension, simplex_resolution)
        self.n_thetas = len(self.thetas)

    def fit(self):
        self.convex_estimators = [None for _ in range(self.n_thetas)]
        for i in range(self.n_thetas):
            theta = self.thetas[i]
            self.convex_estimators[i] = ConvexEstimator(
                self.kernel_estimator, theta
            )


def get_simplex(simplex_dimension, simplex_resolution):
    n_values = int(np.ceil(1 / simplex_resolution))
    values = [i * simplex_resolution for i in range(n_values)]
    values.append(1)
    values = list(set(values))
    values_rep = [values for _ in range(simplex_dimension)]
    simplex = list(itertools.product(*values_rep))
    simplex = [np.array(s) for s in simplex if np.sum(s) <= 1]
    return simplex

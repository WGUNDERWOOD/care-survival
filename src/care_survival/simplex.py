import numpy as np
import itertools

from care_survival import metrics as care_metrics


class Combination:
    def __init__(self, estimator, theta):
        self.estimator = estimator
        self.theta = theta
        self.simplex_dimension = len(theta)
        self.f_check = {}
        for split in care_metrics.get_splits():
            self.f_check[split] = self.get_f_check_split(split)
        self.score = self.get_score()

    def get_f_check_split(self, split):
        theta = self.theta
        theta_0 = 1.0 - np.sum(theta)
        f_check = theta_0 * self.estimator.f_hat[split]
        embedding_data = self.estimator.embedding.data
        for i in range(self.simplex_dimension):
            f_check = f_check + theta[i] * embedding_data[split].f_tilde[:, i]
        return f_check

    def get_score(self):
        embedding = self.estimator.embedding
        f = {}
        for split in care_metrics.get_splits():
            f[split] = self.get_f_check(split)

        score = {}
        for metric in care_metrics.get_metrics():
            score[metric] = {}
            for split in care_metrics.get_splits():
                score[metric][split] = care_metrics.get_metric_split(
                    f[split], embedding, metric, split
                )
        return score


class SimplexSelection:
    def __init__(self, estimator, simplex_dimension, simplex_resolution):
        self.estimator = estimator
        self.simplex_dimension = simplex_dimension
        self.simplex_resolution = simplex_resolution
        self.thetas = get_simplex(simplex_dimension, simplex_resolution)
        self.n_thetas = len(self.thetas)

    def select(self):
        self.combinations = [None for _ in range(self.n_thetas)]

        for i in range(self.n_thetas):
            theta = self.thetas[i]
            self.combinations[i] = Combination(self.estimator, theta)


def get_simplex(simplex_dimension, simplex_resolution):
    n_values = np.ceil(1 / simplex_resolution)
    values = [i * simplex_resolution for i in range(n_values)]
    values.append(1)
    values = list(set(values))
    values_rep = [values for _ in range(simplex_dimension)]
    simplex = list(itertools.product(*values_rep))
    simplex = [np.array(s) for s in simplex if np.sum(s) <= 1]

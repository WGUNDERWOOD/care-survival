import numpy as np

from care_survival import metrics as care_metrics


class ConvexEstimator:
    def __init__(self, kernel_estimator, theta):
        self.kernel_estimator = kernel_estimator
        self.gamma = kernel_estimator.gamma
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
        f_check = f_check - np.sum(f_check) / max(len(f_check), 1)
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
                    f[split],
                    embedding,
                    metric,
                    split,
                    self.kernel_estimator.with_concordance,
                )
        return score

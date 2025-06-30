import numpy as np
import itertools

from care_survival import convex as care_convex
from care_survival import kernel_estimator as care_kernel_estimator
from care_survival import metrics as care_metrics


class CARE:
    def __init__(
        self,
        embedding,
        gamma_min,
        gamma_max,
        n_gammas,
        simplex_resolution,
        with_concordance=care_metrics.get_splits(),
        verbose=False,
    ):
        self.embedding = embedding
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.n_gammas = n_gammas
        self.gammas = get_gammas(gamma_min, gamma_max, n_gammas)
        self.simplex_resolution = simplex_resolution
        self.simplex_dimension = np.shape(embedding.data["train"].f_tilde)[1]
        self.thetas = get_simplex(self.simplex_dimension, simplex_resolution)
        self.n_thetas = len(self.thetas)
        self.with_concordance = with_concordance
        self.verbose = verbose

    def fit(self):
        beta_hat = self.embedding.data["train"].get_default_beta()
        inv_hessian_hat = self.embedding.data["train"].get_default_inv_hessian()
        self.convex_estimators = []
        self.kernel_estimators = []

        for i in range(self.n_gammas):
            # fit kernel estimator at gamma
            gamma = self.gammas[i]
            if self.verbose:
                print(f"{i + 1} / {self.n_gammas}: gamma = {gamma}")
            kernel_estimator = care_kernel_estimator.KernelEstimator(
                self.embedding, gamma, self.with_concordance
            )
            kernel_estimator.fit(beta_hat, inv_hessian_hat)
            inv_hessian_hat = kernel_estimator.inv_hessian_hat
            beta_hat = kernel_estimator.beta_hat

            for j in range(self.n_thetas):
                # fit convex estimator at theta
                theta = self.thetas[j]
                convex_estimator = care_convex.ConvexEstimator(kernel_estimator, theta)
                self.convex_estimators.append(convex_estimator)
                if np.sum(theta) == 0:
                    self.kernel_estimators.append(convex_estimator)

        self.best = {}
        for model in care_metrics.get_models():
            self.best[model] = {}
            for metric in care_metrics.get_metrics():
                self.best[model][metric] = {}
                for split in care_metrics.get_splits():
                    self.best[model][metric][split] = self.best_by(model, metric, split)

        self.summarise()

    def best_by(self, model, metric, split):
        cs = [c for c in self.convex_estimators if c.score[metric][split] is not None]

        if model == "kernel":
            cs = [c for c in cs if np.sum(c.theta) == 0]
        elif model == "external":
            cs = [c for c in cs if np.any(c.theta == 1)]

        def key(c):
            return c.score[metric][split]

        return min(cs, key=key)

    def summarise(self):
        star = self.best["kernel"]["l2"]["test"]
        hat = self.best["kernel"]["ln"]["valid"]
        dagger = self.best["aggregated"]["l2"]["test"]
        check = self.best["aggregated"]["ln"]["valid"]
        tilde = self.best["external"]["ln"]["valid"]

        self.summary = {
            "n_train": self.embedding.data["train"].n,
            "n_valid": self.embedding.data["valid"].n,
            "n_test": self.embedding.data["test"].n,
            "gamma_star": star.kernel_estimator.gamma,
            "gamma_hat": hat.kernel_estimator.gamma,
            "gamma_dagger": dagger.kernel_estimator.gamma,
            "gamma_check": check.kernel_estimator.gamma,
            "theta_dagger": dagger.theta,
            "theta_check": check.theta,
            "l2_star": star.score["l2"]["test"],
            "l2_hat": hat.score["l2"]["test"],
            "l2_dagger": dagger.score["l2"]["test"],
            "l2_check": check.score["l2"]["test"],
            "l2_tilde": tilde.score["l2"]["test"],
            "concordance_star": star.score["concordance"]["test"],
            "concordance_hat": hat.score["concordance"]["test"],
            "concordance_dagger": dagger.score["concordance"]["test"],
            "concordance_check": check.score["concordance"]["test"],
            "concordance_tilde": tilde.score["concordance"]["test"],
        }


def get_gammas(gamma_min, gamma_max, n_gammas):
    if n_gammas == 1:
        assert gamma_min == gamma_max, "if n_gammas = 1, gamma_min must equal gamma_max"
        return [gamma_min]
    else:
        ratio = (gamma_max / gamma_min) ** (1 / (n_gammas - 1))
        return [gamma_min * ratio**i for i in reversed(range(n_gammas))]


def get_simplex(simplex_dimension, simplex_resolution):
    n_values = int(np.ceil(1 / simplex_resolution))
    values = [i * simplex_resolution for i in range(n_values)]
    values.append(1)
    values = list(set(values))
    values_rep = [values for _ in range(simplex_dimension)]
    simplex = list(itertools.product(*values_rep))
    simplex = [np.array(s) for s in simplex if np.sum(s) <= 1]
    return simplex

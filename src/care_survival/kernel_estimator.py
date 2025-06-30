import numpy as np
from scipy.optimize import minimize

from care_survival import metrics as care_metrics


class KernelEstimator:
    def __init__(self, embedding, gamma, with_concordance):
        self.embedding = embedding
        self.gamma = gamma
        self.method = embedding.data["train"].method
        self.with_concordance = with_concordance

        if self.method == "feature_map":
            self.feature_dim = embedding.data["train"].feature_dim

    def get_f(self, beta, split):
        if self.method == "kernel":
            if split == "valid":
                matrix = self.embedding.K_tilde_valid_train
            elif split == "test":
                matrix = self.embedding.K_tilde_test_train
            else:
                matrix = self.embedding.data[split].K_tilde

        elif self.method == "feature_map":
            matrix = self.embedding.data[split].Phi_tilde

        return matrix @ beta

    def get_ln_split(self, beta, split):
        f = self.get_f(beta, split)
        return care_metrics.get_ln_split(f, self.embedding, split)

    def get_lng_split(self, beta, split):
        ln = self.get_ln_split(beta, split)

        if self.method == "kernel":
            K_hat_train = self.embedding.data["train"].K_hat
            penalty = self.gamma * beta.T @ K_hat_train @ beta

        elif self.method == "feature_map":
            Phi_bar = self.embedding.data["train"].Phi_bar
            feature_const = self.embedding.data["train"].feature_const
            beta_0 = -beta @ Phi_bar / feature_const
            penalty = self.gamma * np.sum(beta**2) + beta_0**2

        lng = ln + penalty
        return lng

    def get_dlng_split(self, beta, split):
        embedding_data = self.embedding.data[split]
        f = self.get_f(beta, split)
        f_max = np.max(f)
        f_expt = expt(f, f_max)
        sn = get_sn(embedding_data, f_expt)
        Dsn = get_Dsn(embedding_data, f_expt)
        n = embedding_data.n
        N = embedding_data.N

        if self.method == "kernel":
            K_tilde = embedding_data.K_tilde
            K_hat = embedding_data.K_hat
            dlng = np.sum(
                (Dsn.T / sn - K_tilde.T) * N / n + 2 * self.gamma * K_hat.T * beta,
                axis=1,
            )

        elif self.method == "feature_map":
            Phi_tilde = embedding_data.Phi_tilde
            Phi_bar = embedding_data.Phi_bar
            feature_const = embedding_data.feature_const
            beta_0 = -beta @ Phi_bar / feature_const
            dlng = (
                np.sum((Dsn.T / sn - Phi_tilde.T) * N / n, axis=1)
                + 2 * self.gamma * beta
                - 2 * self.gamma * Phi_bar * beta_0 / feature_const
            )

        return dlng

    def fit(self, beta_init, inv_hessian_init):
        def cost(beta):
            return self.get_lng_split(beta, "train")

        def gradient(beta):
            return self.get_dlng_split(beta, "train")

        if beta_init is None:
            beta_init = self.embedding.data["train"].get_default_beta()
        if inv_hessian_init is None:
            inv_hessian_init = self.embedding.data["train"].get_default_inv_hessian()

        gtol = 1e-6
        res = minimize(
            cost,
            beta_init,
            method="BFGS",
            jac=gradient,
            options={"hess_inv0": inv_hessian_init, "gtol": gtol},
        )

        self.beta_hat = res.x
        self.inv_hessian_hat = (res.hess_inv + res.hess_inv.T) / 2

        self.f_hat = {}
        for split in care_metrics.get_splits():
            self.f_hat[split] = self.get_f(self.beta_hat, split)


def expt(f, f_max):
    return np.exp(f - f_max)


def get_sn(embedding_data, f_expt):
    n = embedding_data.n
    cumulative_mean = np.cumsum(f_expt[::-1]) / n
    R = embedding_data.R.astype(int)
    return cumulative_mean[n - R - 1]


def get_Dsn(embedding_data, f_expt):
    n = embedding_data.n
    R = embedding_data.R.astype(int)

    if embedding_data.method == "kernel":
        K_tilde = embedding_data.K_tilde
        counter = np.array(np.arange(n))
        A = (R.reshape(-1, 1) <= counter) * f_expt / n
        return A @ K_tilde

    elif embedding_data.method == "feature_map":
        Phi_tilde = embedding_data.Phi_tilde
        A = Phi_tilde * f_expt.reshape(-1, 1)
        B = np.cumsum(A[::-1, :], axis=0) / n
        return B[n - R - 1, :]

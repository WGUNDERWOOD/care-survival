import numpy as np
from scipy.optimize import minimize

from care_survival import metrics as care_metrics


class Estimator:
    def __init__(self, embedding, gamma):
        self.embedding = embedding
        self.gamma = gamma
        self.method = embedding.data["train"].method

        if self.method == "feature_map":
            self.feature_dim = embedding.data["train"].feature_dim

    def get_f(self, beta, split):
        if self.method == "kernel":
            if split == "valid":
                matrix = self.embedding.K_cent_valid_train
            elif split == "test":
                matrix = self.embedding.K_cent_test_train
            else:
                matrix = self.embedding.data[split].K_cent

        elif self.method == "feature_map":
            matrix = self.embedding.data[split].Phi_cent

        return matrix @ beta

    def get_ln_split(self, beta, split):
        f = self.get_f(beta, split)
        return care_metrics.get_ln_split(f, self.embedding, split)

    def get_lng_split(self, beta, split):
        ln = self.get_ln_split(beta, split)

        if self.method == "kernel":
            K_hat_train = self.embedding.data["train"].K_hat
            Kb = K_hat_train @ beta
            penalty = self.gamma * Kb @ beta

        elif self.method == "feature_map":
            Phi_bar = self.embedding.data["train"].Phi_bar
            feature_const = self.embedding.data["train"].feature_const
            beta_0 = -beta @ Phi_bar / feature_const
            penalty = self.gamma * np.sum(beta**2) + beta_0**2

        return ln + penalty

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
            K_cent = embedding_data.K_cent
            K_hat = embedding_data.K_hat
            return np.sum(
                (Dsn.T / sn - K_cent.T) * N / n + 2 * self.gamma * K_hat.T * beta,
                axis=1,
            )

        elif self.method == "feature_map":
            Phi_cent = embedding_data.Phi_cent
            Phi_bar = embedding_data.Phi_bar
            feature_const = embedding_data.feature_const
            beta_0 = -beta @ Phi_bar / feature_const
            return (
                np.sum((Dsn.T / sn - Phi_cent.T) * N / n, axis=1)
                + 2 * self.gamma * beta
                - 2 * self.gamma * Phi_bar * beta_0 / feature_const
            )

    def optimise(self, beta_init, inv_hessian_init):
        def cost(beta):
            return self.get_lng_split(beta, "train")

        def gradient(beta):
            return self.get_dlng_split(beta, "train")

        if beta_init is None:
            beta_init = self.embedding.data["train"].get_default_beta()
        if inv_hessian_init is None:
            inv_hessian_init = self.embedding.data["train"].get_default_inv_hessian()

        gtol = 1e-6
        #print("Starting BFGS")
        res = minimize(
            cost,
            beta_init,
            method="BFGS",
            jac=gradient,
            options={"hess_inv0": inv_hessian_init, "gtol": gtol},
        )
        #print("Finished BFGS")

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
        K_cent = embedding_data.K_cent
        counter = np.array(np.arange(n))
        A = (R.reshape(-1, 1) <= counter) * f_expt / n
        return A @ K_cent

    elif embedding_data.method == "feature_map":
        Phi_cent = embedding_data.Phi_cent
        A = Phi_cent * f_expt.reshape(-1, 1)
        B = np.cumsum(A[::-1, :], axis=0) / n
        return B[n - R - 1, :]

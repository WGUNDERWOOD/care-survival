import numpy as np
from scipy.optimize import minimize

from care_survival import score as care_score
from care_survival import concordance as care_concordance


class Estimator:
    def __init__(self, embedding, gamma):
        self.embedding = embedding
        self.gamma = gamma
        self.n_train = embedding.data["train"].n
        self.n_valid = embedding.data["valid"].n
        self.n_test = embedding.data["test"].n
        self.method = embedding.data["train"].method

        if self.method == "feature_map":
            self.feature_dim = embedding.data["train"].feature_dim

        self.beta_hat = self.embedding.data["train"].get_default_beta()
        self.inv_hessian_hat = self.embedding.data["train"].get_default_inv_hessian()

    def get_f(self, beta, split):
        embedding = self.embedding

        if self.method == "kernel":
            if split == "valid":
                matrix = embedding.K_cent_valid_train
            elif split == "test":
                matrix = embedding.K_cent_test_train
            else:
                matrix = embedding.data[split].K_cent

        elif self.method == "feature_map":
            matrix = embedding.data[split].Phi_cent

        return matrix @ beta

    # TODO rewrite these to just take f, not beta
    # TODO should be functions not methods
    # TODO also write a simplified method to take beta
    # TODO functions then can be used in CARE
    def get_ln_split(self, beta, split):
        embedding_data = self.embedding.data[split]
        n = embedding_data.n
        f = self.get_f(beta, split)
        f_max = np.max(f)
        f_expt = expt(f, f_max)
        sn = get_sn(embedding_data, f_expt)
        N = embedding_data.N
        ln_cent = embedding_data.ln_cent

        return np.sum((np.log(sn) + f_max - f) * N) / n - ln_cent

    def get_ln(self, beta):
        ln = {}
        for split in get_splits():
            ln[split] = self.get_ln_split(beta, split)
        return ln

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

    def get_lng(self, beta):
        lng = {}
        for split in get_splits():
            lng[split] = self.get_lng_split(beta, split)
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
            K_cent = embedding_data.K_cent
            K_hat = embedding_data.K_hat
            return np.sum(
                (Dsn.T / sn - K_cent.T) * N / n + 2 * self.gamma * K_hat.T * beta,
                axis=1,
            )

        elif self.method == "feature_map":
            Phi_cent = embedding_data.Phi_cent
            Phi_bar = embedding_data.Phi_bar
            feature_dim = embedding_data.feature_dim
            feature_const = embedding_data.feature_const
            beta_0 = -beta @ Phi_bar / feature_const
            return (
                np.sum((Dsn.T / sn - Phi_cent.T) * N / n, axis=1)
                + 2 * self.gamma * beta
                - 2 * self.gamma * Phi_bar * beta_0 / feature_const
            )

    def get_rmse(self, beta):
        rmse = {}
        for split in get_splits():
            f = self.get_f(beta, split)
            rmse[split] = get_rmse(f, self.embedding.data[split].f_0)
        return rmse

    def get_concordance(self, beta):
        concordance = {}
        for split in get_splits():
            f = self.get_f(beta, split)
            concordance[split] = care_concordance.get_concordance(
                f, self.embedding.data[split]
            )
        return concordance

    def get_score(self, beta):
        score = {}
        score["ln"] = self.get_ln(beta)
        score["rmse"] = self.get_rmse(beta)
        score["concordance"] = self.get_concordance(beta)
        return score

    def optimise(self, beta_init, inv_hessian_init):
        cost = lambda beta: self.get_lng_split(beta, "train")
        gradient = lambda beta: self.get_dlng_split(beta, "train")
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
        self.f_hat_train = self.get_f(self.beta_hat, "train")
        self.f_hat_valid = self.get_f(self.beta_hat, "valid")
        self.f_hat_test = self.get_f(self.beta_hat, "test")
        self.score = self.get_score(self.beta_hat)


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
        feature_dim = embedding_data.feature_dim
        Phi_cent = embedding_data.Phi_cent
        A = Phi_cent * f_expt.reshape(-1, 1)
        B = np.cumsum(A[::-1, :], axis=0) / n
        return B[n - R - 1, :]


def get_rmse(f, f_0):
    if f_0 is None:
        return None
    else:
        n = len(f)
        diffs = f - f_0
        mse = np.sum(diffs**2) / n
        return np.sqrt(mse)


def get_splits():
    return ["train", "valid", "test"]


def get_metrics():
    return ["ln", "rmse", "concordance"]

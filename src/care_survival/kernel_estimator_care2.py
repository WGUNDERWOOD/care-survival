import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag

from care_survival import metrics as care_metrics


class KernelEstimatorCARE2:
    def __init__(self, embedding, gamma, with_concordance):
        self.embedding = embedding
        self.gamma = gamma
        self.method = embedding.data["train"].method
        self.with_concordance = with_concordance

        if self.method == "feature_map":
            self.feature_dim = embedding.data["train"].feature_dim

    def get_f(self, beta, theta, split):
        if self.method == "kernel":
            if split == "valid":
                matrix = self.embedding.K_tilde_valid_train
            elif split == "test":
                matrix = self.embedding.K_tilde_test_train
            else:
                matrix = self.embedding.data[split].K_tilde

        elif self.method == "feature_map":
            matrix = self.embedding.data[split].Phi_tilde

        f_tilde = self.embedding.data[split].f_tilde
        return matrix @ beta + f_tilde @ theta

    def get_ln_split(self, beta, theta, split):
        f = self.get_f(beta, theta, split)
        return care_metrics.get_ln_split(f, self.embedding, split)

    def get_lng_split(self, beta, theta, split):
        ln = self.get_ln_split(beta, theta, split)

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

    def get_dlng_split(self, beta, theta, split):
        embedding_data = self.embedding.data[split]
        f = self.get_f(beta, theta, split)
        f_max = np.max(f)
        f_expt = expt(f, f_max)
        sn = get_sn(embedding_data, f_expt)
        Dsn = get_Dsn(embedding_data, f_expt)
        Dsn_beta = Dsn[0]
        Dsn_theta = Dsn[1]
        n = embedding_data.n
        N = embedding_data.N

        if self.method == "kernel":
            K_tilde = embedding_data.K_tilde
            K_hat = embedding_data.K_hat
            dlng_beta = np.sum(
                (Dsn_beta.T / sn - K_tilde.T) * N / n
                + 2 * self.gamma * K_hat.T * beta,
                axis=1,
            )

        elif self.method == "feature_map":
            Phi_tilde = embedding_data.Phi_tilde
            Phi_bar = embedding_data.Phi_bar
            feature_const = embedding_data.feature_const
            beta_0 = -beta @ Phi_bar / feature_const
            dlng_beta = (
                np.sum((Dsn_beta.T / sn - Phi_tilde.T) * N / n, axis=1)
                + 2 * self.gamma * beta
                - 2 * self.gamma * Phi_bar * beta_0 / feature_const
            )

        f_tilde = self.embedding.data["train"].f_tilde
        dlng_theta = np.sum(
            (Dsn_theta.T / sn - f_tilde.T) * N / n,
            axis=1,
        )

        return dlng_beta, dlng_theta

    def fit(self, beta_init, theta_init, inv_hessian_init):

        n = self.embedding.data["train"].n
        p = self.embedding.data["train"].f_tilde.shape[1]

        def cost(param):
            beta = param[0:n]
            theta = param[n:]
            return self.get_lng_split(beta, theta, "train")

        def gradient(param):
            beta = param[0:n]
            theta = param[n:]
            dlng = self.get_dlng_split(beta, theta, "train")
            dlng_beta = dlng[0]
            dlng_theta = dlng[1]
            return np.concatenate((dlng_beta, dlng_theta))

        if beta_init is None:
            beta_init = self.embedding.data["train"].get_default_beta()

        if theta_init is None:
            theta_init = np.zeros(p)

        init = np.concatenate((beta_init, theta_init))

        if inv_hessian_init is None:
            inv_hessian_beta_init = self.embedding.data["train"].get_default_inv_hessian()
            inv_hessian_theta_init = np.eye(p)
            inv_hessian_init = block_diag(inv_hessian_beta_init, inv_hessian_theta_init)

        gtol = 1e-6
        res = minimize(
            cost,
            init,
            method="BFGS",
            jac=gradient,
            options={"hess_inv0": inv_hessian_init, "gtol": gtol},
        )

        self.beta_hat = res.x[0:n]
        self.theta_hat = res.x[n:]
        self.inv_hessian_hat = (res.hess_inv + res.hess_inv.T) / 2

        self.f_hat = {}
        for split in care_metrics.get_splits():
            self.f_hat[split] = self.get_f(self.beta_hat, self.theta_hat, split)

        self.get_score()

    def get_score(self):
        embedding = self.embedding
        f = {}
        for split in care_metrics.get_splits():
            f[split] = self.f_hat[split]

        score = {}
        for metric in care_metrics.get_metrics():
            score[metric] = {}
            for split in care_metrics.get_splits():
                score[metric][split] = care_metrics.get_metric_split(
                    f[split],
                    embedding,
                    metric,
                    split,
                    self.with_concordance,
                )
        self.score = score


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
    counter = np.array(np.arange(n))
    A = (R.reshape(-1, 1) <= counter) * f_expt / n

    if embedding_data.method == "kernel":
        K_tilde = embedding_data.K_tilde
        Dsn_beta = A @ K_tilde

    elif embedding_data.method == "feature_map":
        Phi_tilde = embedding_data.Phi_tilde
        B = Phi_tilde * f_expt.reshape(-1, 1)
        C = np.cumsum(B[::-1, :], axis=0) / n
        Dsn_beta = C[n - R - 1, :]

    Dsn_theta = A @ embedding_data.f_tilde
    return Dsn_beta, Dsn_theta

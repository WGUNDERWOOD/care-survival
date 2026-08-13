import numpy as np

def get_R(T):
    # Ri = min{j: Rj(Ti) = 1} = min{j: Tj >= Ti}
    return np.searchsorted(T, T, side="left")

def get_Z(T):
    # Zi = max{j: Ri(Tj) = 1} = max{j: Ti >= Tj}
    return np.searchsorted(T, T, side="right") - 1

def get_R_bar(T):
    # R_bar_i = sum_j Rj(Ti) / n = sum_j 1{Tj >= Ti} / n
    return 1 - np.searchsorted(T, T, side="left") / len(T)

class EmbeddingData:
    def __init__(self, data, kernel, method, nystrom_m=None):
        data.sort()
        self.n = data.n
        self.d = data.d
        self.X = data.X
        self.T = data.T
        self.I = data.I
        self.f_tilde = data.f_tilde
        self.f_0 = data.f_0
        self.method = method
        self.N = 1 - self.I
        self.nystrom_m = nystrom_m

        self.R = get_R(self.T)
        self.Z = get_Z(self.T)
        self.R_bar = get_R_bar(self.T)
        self.ln_cent = np.sum(np.log(self.R_bar) * self.N) / max(self.n, 1)

        if method == "kernel":
            self.norm_one = kernel.norm_one()
            self.K = kernel.k(self.X, self.X)
            self.K_bar = np.sum(self.K, axis=0) / self.n
            self.K_tilde = self.K - self.K_bar
            self.K_hat = (
                self.K
                - self.K_bar
                - self.K_bar.reshape(-1, 1)
                + np.outer(self.K_bar, self.K_bar) * self.norm_one**2
            )

        elif method == "nystrom":
            self.nystrom_indices = np.random.choice(self.n, self.nystrom_m, replace=False)
            self.nystrom_indices = np.sort(self.nystrom_indices)
            self.norm_one = kernel.norm_one()
            self.K = kernel.k(self.X, self.X[self.nystrom_indices])
            self.K_bar = np.sum(self.K, axis=0) / self.n
            self.K_tilde = self.K - self.K_bar
            self.K_hat = (
                self.K[self.nystrom_indices]
                - self.K_bar
                - self.K_bar.reshape(-1, 1)
                + np.outer(self.K_bar, self.K_bar) * self.norm_one**2
            )

        elif method == "feature_map":
            self.feature_dim = kernel.feature_dim(self.d)
            self.feature_const = kernel.feature_const()
            self.Phi = kernel.phi(self.X)
            self.Phi_bar = np.sum(self.Phi, axis=0) / max(self.n, 1)
            self.Phi_tilde = self.Phi - self.Phi_bar

        self.breslow = self.get_breslow()

    def get_default_beta(self):
        if self.method == "kernel":
            return np.zeros(self.n)

        elif self.method == "nystrom":
            return np.zeros(self.nystrom_m)

        elif self.method == "feature_map":
            return np.zeros(self.feature_dim)

    def get_default_inv_hessian(self):
        if self.method == "kernel":
            return np.eye(self.n)

        if self.method == "nystrom":
            return np.eye(self.nystrom_m)

        elif self.method == "feature_map":
            return np.eye(self.feature_dim)

    def get_breslow(self):
        n = self.n
        N_over_R = self.N / (self.R_bar * n)
        cumulative_sum = np.cumsum(N_over_R)
        p = cumulative_sum[self.Z.astype(int)]
        return np.exp(-p)


class Embedding:
    def __init__(self, data_train, data_valid, data_test, kernel, method, nystrom_m=None):
        self.data = {
            "train": EmbeddingData(data_train, kernel, method, nystrom_m),
            "valid": EmbeddingData(data_valid, kernel, method, nystrom_m),
            "test": EmbeddingData(data_test, kernel, method, nystrom_m),
        }

        if method == "kernel":
            self.K_tilde_valid_train = (
                kernel.k(self.data["valid"].X, self.data["train"].X)
                - self.data["train"].K_bar
            )
            self.K_tilde_test_train = (
                kernel.k(self.data["test"].X, self.data["train"].X)
                - self.data["train"].K_bar
            )

        elif method == "nystrom":
            nystrom_indices = self.data["train"].nystrom_indices
            self.K_tilde_valid_train = (
                kernel.k(self.data["valid"].X, self.data["train"].X[nystrom_indices])
                - self.data["train"].K_bar
            )
            self.K_tilde_test_train = (
                kernel.k(self.data["test"].X, self.data["train"].X[nystrom_indices])
                - self.data["train"].K_bar
            )

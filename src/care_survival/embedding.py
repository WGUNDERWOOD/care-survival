import numpy as np


class EmbeddingData:
    def __init__(self, data, kernel, method):
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

        # R
        self.R = np.zeros(self.n)
        R_prev = 0
        for j in range(self.n):
            R_prev += np.argmax(self.T[R_prev : (j + 1)] >= self.T[j])
            self.R[j] = R_prev

        # Z
        self.Z = np.zeros(self.n)
        Z_prev = data.n - 1
        for i in reversed(range(data.n)):
            if i == 0:
                Z_prev -= np.argmax(self.T[Z_prev::-1] <= self.T[i])
            else:
                Z_prev -= np.argmax(self.T[Z_prev : i - 1 : -1] <= self.T[i])
            self.Z[i] = Z_prev

        self.R_bar = (self.n - self.R) / self.n
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

        elif self.method == "feature_map":
            return np.zeros(self.feature_dim)

    def get_default_inv_hessian(self):
        if self.method == "kernel":
            return np.eye(self.n)

        elif self.method == "feature_map":
            return np.eye(self.feature_dim)

    def get_breslow(self):
        n = self.n
        N_over_R = self.N / (self.R_bar * n)
        cumulative_sum = np.cumsum(N_over_R)
        p = cumulative_sum[self.Z.astype(int)]
        return np.exp(-p)


class Embedding:
    def __init__(self, data_train, data_valid, data_test, kernel, method):
        self.data = {
            "train": EmbeddingData(data_train, kernel, method),
            "valid": EmbeddingData(data_valid, kernel, method),
            "test": EmbeddingData(data_test, kernel, method),
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

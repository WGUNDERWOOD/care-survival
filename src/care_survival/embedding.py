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
        # TODO check this agrees with Rust
        self.R = np.zeros(data.n)
        R_prev = 0
        for j in range(data.n):
            candidates = np.arange(R_prev, j + 1)
            mask = data.T[candidates] >= data.T[j]
            R_val = candidates[np.argmax(mask)]
            self.R[j] = R_val
            R_prev = R_val

        # Z
        # TODO check this agrees with Rust
        self.Z = np.zeros(data.n)
        Z_prev = data.n
        for i in reversed(range(data.n)):
            candidates = np.arange(i, Z_prev)
            mask = data.T[i] >= data.T[candidates]
            Z_val = candidates[mask][-1]
            self.Z[i] = Z_val
            Z_prev = Z_val

        self.R_bar = (self.n - self.R) / self.n
        self.ln_cent = np.sum(np.log(self.R_bar) * self.N) / self.n

        if method == "kernel":
            self.norm_one = kernel.norm_one()
            self.K = kernel.k(self.X, self.X)
            self.K_bar = np.sum(self.K, axis=0) / self.n
            self.K_cent = self.K - self.K_bar
            self.K_hat = (
                self.K
                - self.K_bar
                - self.K_bar.reshape(-1, 1)
                - np.outer(self.K_bar, self.K_bar) * self.norm_one
            )

        elif method == "feature_map":
            self.feature_dim = kernel.feature_dim(self.d)
            self.feature_const = kernel.feature_const()
            self.Phi = kernel.phi(self.X)
            self.Phi_bar = np.sum(self.Phi, axis=0) / self.n
            self.Phi_cent = self.Phi - self.Phi_bar

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


class Embedding:
    def __init__(self, data_train, data_valid, data_test, kernel, method):
        self.train = EmbeddingData(data_train, kernel, method)
        self.valid = EmbeddingData(data_valid, kernel, method)
        self.test = EmbeddingData(data_test, kernel, method)

        if method == "kernel":
            self.K_cent_valid_train = (
                kernel.k(self.valid.X, self.train.X) - self.train.K_bar
            )
            self.K_cent_test_train = (
                kernel.k(self.test.X, self.train.X) - self.train.K_bar
            )

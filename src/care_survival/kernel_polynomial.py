import numpy as np

class PolynomialKernel:
    def __init__(self, a, p):
        self.a = a
        self.p = p

    def k(self, X1, X2):
        d = np.shape(X1)[1];
        return (X1 @ X2.T / d + self.a) ** self.p

    def norm_one(self):
        return (1 / self.a) ** (self.p / 2)

    def phi(self, X):
        n = np.shape(X)[0];
        d = np.shape(X)[1];
        feature_dim = self.feature_dim(d);
        if self.p == 1:
            return X
        elif self.p == 2:
            phi_X = np.zeros((n, feature_dim))
            r = 0
            for i in range(d):
                phi_X[:,r] = np.sqrt(2) * X[:,i] / np.sqrt(d);
                r += 1
            for i in range(d):
                phi_X[:,r] = X[:,i] * X[:,i] / np.sqrt(d);
                r += 1
            for i in range(d):
                for j in range(i):
                    phi_X[:,r] = np.sqrt(2) * X[:,i] * X[:,j] / np.sqrt(d);
                    r += 1;
            return phi_X

    def feature_dim(self, d):
        if self.p == 1:
            return d
        elif self.p == 2:
            return int(2 * d + d * (d - 1) / 2)

    def feature_const(self):
        return self.a ** (self.p / 2)

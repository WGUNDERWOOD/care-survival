import numpy as np


class Data:
    def __init__(self, X, T, I, f_tilde, f_0):
        self.n = np.shape(X)[0]
        self.d = np.shape(X)[1]
        self.m = np.shape(f_tilde)[1]
        self.X = X
        self.T = T
        self.I = I
        self.f_tilde = f_tilde
        self.f_0 = f_0

    def sort(self):
        indices = np.argsort(self.T)
        self.X = self.X[indices]
        self.T = self.T[indices]
        self.I = self.I[indices]
        self.f_tilde = self.f_tilde[indices]
        self.f_0 = self.f_0[indices]

from care_survival import data as care_data

import numpy as np


class Distribution:
    def __init__(self, d, TC_low, TC_high, f_tilde_funcs, f_0_func, Lambda_inv):
        self.TC_low = TC_low
        self.TC_high = TC_high
        self.f_tilde_funcs = f_tilde_funcs
        self.f_0_func = f_0_func
        self.Lambda_inv = Lambda_inv

    def sample(self, n):
        X = np.random.random((n, self.d))
        U_TC = np.random.random(n)
        TC = np.minimum(U_TC * (self.TC_high - self.TC_low) + self.TC_low, 1)
        f_0_X = np.apply_along_axis(self.f_0_func, 1, X)
        U_TS = np.random.random(n)
        TS = self.Lambda_inv(-np.exp(-f_0_X) * np.log(U_TS))
        I = TC < TS
        T = np.minimum(TC, TS)
        f_tilde = np.array([[f(X[i, :]) for f in self.f_tilde_funcs] for i in range(n)])
        f_0_mean = np.sum(f_0_X) / n
        f_0 = f_0_X - f_0_mean
        return care_data.data(X, T, I, f_tilde, f_0)

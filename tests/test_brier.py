import pytest
import numpy as np
from care_survival.embedding import get_R, get_Z, get_R_bar
from care_survival.metrics import get_adjusted_breslow, get_censoring_breslow, get_censoring_breslow_T

def test_get_adjusted_breslow():
    for rep in range(100):
        n = 7
        T = np.array([0, 0, 0.5, 0.5, 0.6, 0.7, 1])
        f = np.random.random(n)
        N = np.random.random(n) > 0.5
        R = get_R(T)
        R = get_Z(T)
        brier_ts = np.sort(np.random.random(2 * n))

        # compute sn
        cumulative_mean = np.cumsum(np.exp(f)[::-1]) / n
        sn = cumulative_mean[n - R - 1]

        # compute adjusted Breslow
        T_leq_t = T[:, None] <= brier_ts[None, :]
        numerator = T_leq_t * N[:, None]
        p = - np.sum(numerator / sn[:, None], axis=0) / n
        adjusted_breslow = np.exp(p)

        assert np.allclose(get_adjusted_breslow(T, N, sn, brier_ts, T_leq_t),
                           adjusted_breslow)

def test_get_censoring_breslow():
    for rep in range(100):
        n = 7
        T = np.array([0, 0, 0.5, 0.5, 0.6, 0.7, 1])
        f = np.random.random(n)
        I = np.random.random(n) > 0.5
        R = get_R(T)
        Z = get_Z(T)
        R_bar = get_R_bar(T)
        brier_ts = np.sort(np.random.random(2 * n))

        # compute denominator
        cumulative_mean = np.cumsum(np.exp(0 * f)[::-1]) / n
        denominator = cumulative_mean[n - R - 1]

        # compute censoring Breslow
        T_leq_t = T[:, None] <= brier_ts[None, :]
        numerator = T_leq_t * I[:, None]
        p = - np.sum(numerator / denominator[:, None], axis=0) / n
        censoring_breslow = np.exp(p)

        assert np.allclose(get_censoring_breslow(T, I, R_bar, f, brier_ts, T_leq_t),
                           censoring_breslow)

        T_leq_T = T[:, None] <= T[None, :]

        assert np.allclose(get_censoring_breslow(T, I, R_bar, f, T, T_leq_T),
                           get_censoring_breslow_T(I, Z, R_bar, f))

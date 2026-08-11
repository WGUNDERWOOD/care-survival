import pytest
import numpy as np
from care_survival.embedding import get_R, get_Z, get_R_bar
from care_survival.metrics import get_pointwise_brier

def get_adjusted_breslow(T, N, sn, brier_ts):
    n = len(T)
    cs = np.r_[0, np.cumsum(N / sn)]
    k = np.searchsorted(T, brier_ts, side="right")
    return np.exp(-cs[k] / n)


def get_survival_probability(T, N, sn, f, brier_ts, T_leq_t):
    adjusted_breslow = get_adjusted_breslow(T, N, sn, brier_ts)
    survival_probability = adjusted_breslow[None, :] ** np.exp(f)[:, None]
    return survival_probability


def get_censoring_breslow(T, I, R_bar, f, brier_ts, T_leq_t):
    n = len(T)
    cs = np.r_[0, np.cumsum(I / R_bar)]
    k = np.searchsorted(T, brier_ts, side="right")
    return np.exp(-cs[k] / n)


def get_censoring_breslow_T(I, Z, R_bar, f):
    n = len(I)
    I_over_R = I / (R_bar * n)
    cumulative_sum = np.cumsum(I_over_R)
    p = cumulative_sum[Z.astype(int)]
    return np.exp(-p)


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

        assert np.allclose(get_adjusted_breslow(T, N, sn, brier_ts),
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


def get_pointwise_brier_old(T, N, I, Z, R_bar, sn, f, brier_ts, T_leq_t):
    n = len(T)
    survival_probability = get_survival_probability(T, N, sn, f, brier_ts, T_leq_t)
    censoring_breslow_T = get_censoring_breslow_T(I, Z, R_bar, f)
    censoring_breslow_ts = get_censoring_breslow(T, I, R_bar, f, brier_ts, T_leq_t)
    numer1 = survival_probability**2 * T_leq_t * N[:, None]
    term1 = np.sum(numer1 / censoring_breslow_T[:, None], axis=0) / n
    numer2 = (1 - survival_probability)**2 * (1 - T_leq_t)
    term2 = np.sum(numer2 / censoring_breslow_ts[None, :], axis=0) / n
    return term1 + term2

def test_get_brier():
    for rep in range(100):
        n = 7
        T = np.array([0, 0, 0.5, 0.5, 0.6, 0.7, 1])
        f = np.random.random(n)
        I = np.random.random(n) > 0.5
        N = 1 - I
        R = get_R(T)
        Z = get_Z(T)
        R_bar = get_R_bar(T)
        cumulative_mean = np.cumsum(np.exp(f)[::-1]) / n
        sn = cumulative_mean[n - R - 1]
        brier_ts = np.sort(np.random.random(2 * n))
        T_leq_t = T[:, None] <= brier_ts[None, :]
        pointwise_brier = get_pointwise_brier_old(T, N, I, Z, R_bar, sn, f, brier_ts, T_leq_t)
        brier = np.sum(pointwise_brier) / len(brier_ts)
        pointwise_brier_calc = get_pointwise_brier(T, N, I, Z, R_bar, sn, f, brier_ts)
        #brier_calc = get_brier(T, N, I, Z, R_bar, sn, f, brier_ts)
        assert np.allclose(pointwise_brier_calc, pointwise_brier)
        #assert np.allclose(brier_calc, brier)

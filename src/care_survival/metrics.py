import numpy as np
from scipy.stats import kendalltau
import numba

from care_survival import kernel_estimator as care_kernel_estimator


def get_splits():
    return ["train", "valid", "test"]


def get_metrics():
    return ["ln", "l2", "concordance", "brier"]


def get_models():
    return ["kernel", "external", "aggregated"]


def get_ln_split(f, embedding, split):
    embedding_data = embedding.data[split]
    n = embedding_data.n
    if n > 0:
        f_max = np.max(f)
    else:
        f_max = 0
    f_expt = care_kernel_estimator.expt(f, f_max)
    sn = care_kernel_estimator.get_sn(embedding_data, f_expt)
    N = embedding_data.N
    ln_cent = embedding_data.ln_cent

    return np.sum((np.log(sn) + f_max - f) * N) / max(n, 1) - ln_cent


def get_l2_split(f, embedding, split):
    embedding_data = embedding.data[split]
    f_0 = embedding_data.f_0
    if f_0 is None:
        return None
    else:
        n = len(f)
        diffs = f - f_0
        mse = np.sum(diffs**2) / max(n, 1)
        return np.sqrt(mse)

@numba.njit(cache=True)
def fenwick_tree(f, N, Z):
    n = len(f)
    order = np.argsort(f)
    rank = np.empty(n, dtype=np.int64)
    r = 0
    rank[order[0]] = 0
    for k in range(1, n):
        if f[order[k]] != f[order[k - 1]]:
            r += 1
        rank[order[k]] = r
    bit = np.empty(r + 2, dtype=np.int64)
    bit[0] = 0
    for k in range(1, len(bit)):
        bit[k] = k & -k
    fenwick = 0
    i = 0
    for j in range(n):
        while i <= Z[j]:
            k = rank[i] + 1
            while k < len(bit):
                bit[k] -= 1
                k += k & -k
            i += 1
        if N[j]:
            k = rank[j]
            while k > 0:
                fenwick += bit[k]
                k -= k & -k
    return fenwick

def get_concordance(f, N, Z, R):
    n = len(f)
    denominator = np.sum((n - Z - 1) * N)
    i = np.arange(n)

    if denominator > 0:
        numerator = fenwick_tree(f, N, Z)

        return numerator / denominator
    else:
        return 0


def get_concordance_split(f, embedding, split):
    embedding_data = embedding.data[split]
    N = embedding_data.N
    Z = embedding_data.Z
    R = embedding_data.R
    return get_concordance(f, N, Z, R)


@numba.njit()
def brier_loop(T, N, ef, G_T, p, G_ts, k):
    n = len(T)
    m = len(k)
    out = np.empty(m)
    for j in range(m):
        log_s0 = -p[j]
        kj = k[j]
        term1 = 0.0
        term2 = 0.0
        for i in range(kj):
            S = np.exp(ef[i] * log_s0)
            term1 += N[i] * S * S / G_T[i]
        for i in range(kj, n):
            S = np.exp(ef[i] * log_s0)
            x = 1.0 - S
            term2 += x * x
        out[j] = (term1 + term2 / G_ts[j]) / n
    return out


def get_pointwise_brier(T, N, I, Z, R_bar, sn, f, brier_ts):
    n = len(T)
    k = np.searchsorted(T, brier_ts, side="right")
    p = np.r_[0.0, np.cumsum(N / sn)][k] / n
    G_ts = np.exp(-np.r_[0.0, np.cumsum(I / R_bar)][k] / n)
    G_T = np.exp(
        -np.cumsum(I / (R_bar * n))[Z.astype(np.int64)]
    )
    pointwise_brier = brier_loop(
            T, N, np.exp(f), G_T, p, G_ts, k)
    return pointwise_brier


def get_brier_split(f, embedding, split, brier_ts):
    embedding_data = embedding.data[split]
    T = embedding_data.T
    N = embedding_data.N
    I = embedding_data.I
    Z = embedding_data.Z
    R_bar = embedding_data.R_bar
    sn = care_kernel_estimator.get_sn(embedding_data, np.exp(f))
    pointwise_brier = get_pointwise_brier(T, N, I, Z, R_bar, sn, f, brier_ts)
    brier = np.sum(pointwise_brier) / len(brier_ts)
    return brier


def get_survival_prob(T, N, sn, f, ts):
    n = len(T)
    k = np.searchsorted(T, ts, side="right")
    p = np.r_[0.0, np.cumsum(N / sn)][k] / n
    survival_prob = np.exp(np.exp(f) * (-p))
    return survival_prob


def get_metric_split(f, embedding, metric, split, with_metrics, brier_ts):
    if metric in with_metrics.keys():
        if split in with_metrics[metric]:
            if metric == "ln":
                score = get_ln_split(f, embedding, split)
            elif metric == "l2":
                score = get_l2_split(f, embedding, split)
            elif metric == "concordance":
                score = get_concordance_split(f, embedding, split)
            elif metric == "brier":
                score = get_brier_split(f, embedding, split, brier_ts)
            return float(score)

    return np.inf


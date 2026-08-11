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

def get_concordance(f, N, Z, R, use_fenwick):
    n = len(f)
    denominator = np.sum((n - Z - 1) * N)
    i = np.arange(n)

    if denominator > 0:
        if use_fenwick:
            numerator = fenwick_tree(f, N, Z)
        else:
            numerator = sum(np.sum((np.arange(n) > Z[j]) * (f < f[j])) * N[j] for j in range(n))

        return numerator / denominator
    else:
        return 0


def get_concordance_split(f, embedding, split):
    embedding_data = embedding.data[split]
    N = embedding_data.N
    Z = embedding_data.Z
    R = embedding_data.R
    if embedding_data.n > 20:
        use_fenwick = True
    else:
        use_fenwick = False
    return get_concordance(f, N, Z, R, use_fenwick)


def get_adjusted_breslow(T, N, sn, brier_ts, T_leq_t):
    n = len(T)
    N_over_sn = T_leq_t * N[:, None] / (sn[:, None] * n)
    p = np.sum(N_over_sn, axis=0)
    return np.exp(-p)


def get_survival_probability(T, N, sn, f, brier_ts, T_leq_t):
    adjusted_breslow = get_adjusted_breslow(T, N, sn, brier_ts, T_leq_t)
    survival_probability = adjusted_breslow[None, :] ** np.exp(f)[:, None]
    return survival_probability


def get_censoring_breslow(T, I, R_bar, f, brier_ts, T_leq_t):
    n = len(T)
    NC_over_R = T_leq_t * I[:, None] / (R_bar[:, None] * n)
    p = np.sum(NC_over_R, axis=0)
    return np.exp(-p)


def get_censoring_breslow_T(I, Z, R_bar, f):
    n = len(I)
    I_over_R = I / (R_bar * n)
    cumulative_sum = np.cumsum(I_over_R)
    p = cumulative_sum[Z.astype(int)]
    return np.exp(-p)


def get_pointwise_brier(T, N, I, Z, R_bar, sn, f, brier_ts, T_leq_t):
    n = len(T)
    survival_probability = get_survival_probability(T, N, sn, f, brier_ts, T_leq_t)
    censoring_breslow_T = get_censoring_breslow_T(I, Z, R_bar, f)
    censoring_breslow_ts = get_censoring_breslow(T, I, R_bar, f, brier_ts, T_leq_t)
    numer1 = survival_probability**2 * T_leq_t * N[:, None]
    term1 = np.sum(numer1 / censoring_breslow_T[:, None], axis=0) / n
    numer2 = (1 - survival_probability)**2 * (1 - T_leq_t)
    term2 = np.sum(numer2 / censoring_breslow_ts[None, :], axis=0) / n
    return term1 + term2


def get_brier_split(f, embedding, split, brier_ts):
    embedding_data = embedding.data[split]
    T = embedding_data.T
    N = embedding_data.N
    I = embedding_data.I
    Z = embedding_data.Z
    R_bar = embedding_data.R_bar
    sn = care_kernel_estimator.get_sn(embedding_data, np.exp(f))
    T_leq_t = T[:, None] <= brier_ts[None, :]
    pointwise_brier = get_pointwise_brier(T, N, I, Z, R_bar, sn, f, brier_ts, T_leq_t)
    brier = np.sum(pointwise_brier) / len(brier_ts)
    return brier


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


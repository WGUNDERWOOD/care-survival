import numpy as np

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


def get_concordance_split(f, embedding, split):
    embedding_data = embedding.data[split]
    I = embedding_data.I
    n = embedding_data.n
    R = embedding_data.R
    valid = 1 - I

    numerator = 0
    for j in np.where(valid)[0]:
        i_range = np.arange(R[j], n).astype(int)
        numerator += np.sum((f[i_range] < f[j]) & (i_range != j))

    denominator = np.sum((n - R - 1) * valid)
    if denominator > 0:
        return numerator / denominator
    else:
        return 0


def get_adjusted_breslow(f, embedding, split, brier_ts):
    embedding_data = embedding.data[split]
    n = embedding_data.n
    T = embedding_data.T
    N = embedding_data.N
    sn = care_kernel_estimator.get_sn(embedding_data, np.exp(f))
    T_leq_t = T[:, None] <= brier_ts[None, :]
    N_over_sn = T_leq_t * N[:, None] / (sn[:, None] * n)
    p = np.sum(N_over_sn, axis=0)
    return np.exp(-p)


def get_survival_probability(f, embedding, split, brier_ts):
    adjusted_breslow = get_adjusted_breslow(f, embedding, split, brier_ts)
    return adjusted_breslow[None, :] ** np.exp(f[:, None])


def get_censoring_breslow_ts(f, embedding, split, brier_ts):
    embedding_data = embedding.data[split]
    n = embedding_data.n
    T = embedding_data.T
    I = embedding_data.I
    R_bar = embedding_data.R_bar
    T_leq_t = T[:, None] <= brier_ts[None, :]
    NC_over_R = T_leq_t * I[:, None] / (R_bar[:, None] * n)
    p = np.sum(NC_over_R, axis=0)
    return np.exp(-p)


def get_censoring_breslow_T(f, embedding, split):
    embedding_data = embedding.data[split]
    I = embedding_data.T
    Z = embedding_data.Z
    R_bar = embedding_data.R_bar
    n = embedding_data.n
    I_over_R = I / (R_bar * n)
    cumulative_sum = np.cumsum(I_over_R)
    p = cumulative_sum[Z.astype(int)]
    return np.exp(-p)


def get_pointwise_brier(f, embedding, split, brier_ts):
    embedding_data = embedding.data[split]
    n = embedding_data.n
    T = embedding_data.T
    N = embedding_data.N
    I = embedding_data.I
    survival_probability = get_survival_probability(f, embedding, "train", brier_ts)
    censoring_breslow_T = get_censoring_breslow_T(f, embedding, split)
    censoring_breslow_ts = get_censoring_breslow_ts(f, embedding, split, brier_ts)
    T_leq_t = T[:, None] <= brier_ts[None, :]
    numer1 = survival_probability**2 * T_leq_t * N[:, None]
    term1 = np.sum(numer1 / censoring_breslow_T[:, None], axis=0) / n
    numer2 = (1 - survival_probability)**2 * (1 - T_leq_t)
    term2 = np.sum(numer2 / censoring_breslow_ts[None, :], axis=0) / n
    return term1 + term2

def get_brier_split(f, embedding, split, brier_ts):
    pointwise_brier = get_pointwise_brier(f, embedding, split, brier_ts)
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


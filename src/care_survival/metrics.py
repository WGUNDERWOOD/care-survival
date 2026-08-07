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


def get_adjusted_breslow(f, embedding, split):
    # TODO need to be able to evaluate at arbitrary t values
    embedding_data = embedding.data[split]
    N = embedding_data.N
    Z = embedding_data.Z
    n = embedding_data.n
    if n > 0:
        f_max = np.max(f)
    else:
        f_max = 0
    f_expt = care_kernel_estimator.expt(f, f_max)
    sn = care_kernel_estimator.get_sn(embedding_data, f_expt)
    N_over_sn = N / sn
    cumulative_sum = np.cumsum(N_over_sn)
    p = cumulative_sum[Z.astype(int)]
    return np.exp(-p)


def get_censoring_breslow(f, embedding, split):
    embedding_data = embedding.data[split]
    I = embedding_data.I
    R_bar = embedding_data.R_bar
    Z = embedding_data.Z
    I_over_R = I / (R_bar * n)
    cumulative_sum = np.cumsum(I_over_R)
    p = cumulative_sum[self.Z.astype(int)]
    return np.exp(-p)


def get_survival_probability(f, embedding, split):
    adjusted_breslow = get_adjusted_breslow(f, embedding, split)
    return adjusted_breslow ** np.exp(f)


def get_brier_split(f, embedding, split):
    survival_probability = get_survival_probability(f, embedding, "train")
    censoring_breslow = get_censoring_breslow(f, embedding, split)
    return 0.0
    # TODO


def get_metric_split(f, embedding, metric, split, with_concordance, with_brier):
    if metric == "ln":
        score = get_ln_split(f, embedding, split)
    elif metric == "l2":
        score = get_l2_split(f, embedding, split)
    elif metric == "concordance":
        if split in with_concordance:
            score = get_concordance_split(f, embedding, split)
        else:
            return np.inf
    elif metric == "brier":
        if split in with_brier:
            score = get_brier_split(f, embedding, split)
        else:
            return np.inf
    return float(score)

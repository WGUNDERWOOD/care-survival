import numpy as np

from care_survival import estimator as care_estimator


def get_splits():
    return ["train", "valid", "test"]


def get_metrics():
    return ["ln", "rmse", "concordance"]


def get_ln_split(f, embedding, split):
    embedding_data = embedding.data[split]
    n = embedding_data.n
    f_max = np.max(f)
    f_expt = care_estimator.expt(f, f_max)
    sn = care_estimator.get_sn(embedding_data, f_expt)
    N = embedding_data.N
    ln_cent = embedding_data.ln_cent

    return np.sum((np.log(sn) + f_max - f) * N) / n - ln_cent


def get_rmse_split(f, embedding, split):
    embedding_data = embedding.data[split]
    f_0 = embedding_data.f_0
    if f_0 is None:
        return None
    else:
        n = len(f)
        diffs = f - f_0
        mse = np.sum(diffs**2) / n
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
        i_mask = (f[i_range] < f[j]) & (i_range != j)
        numerator += np.sum(i_mask)

    denominator = np.sum((n - R - 1) * valid)
    if denominator > 0:
        return numerator / denominator
    else:
        return 0


def get_metric_split(f, embedding, metric, split):
    if metric == "ln":
        score = get_ln_split(f, embedding, split)
    elif metric == "rmse":
        score = get_rmse_split(f, embedding, split)
    elif metric == "concordance":
        score = get_concordance_split(f, embedding, split)
    return float(score)

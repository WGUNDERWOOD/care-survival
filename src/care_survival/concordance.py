import numpy as np


def get_concordance(f, embedding_data):
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

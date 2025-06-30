import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd

from care_survival import data as care_data
from care_survival import score2 as care_score2
from care_survival import kernels as care_kernels
from care_survival import embedding as care_embedding


def main():
    sex = sys.argv[1]
    today = datetime.now().strftime("%Y-%m-%d")
    dry_run = False

    if dry_run:
        n_train = 60
    else:
        n_female = 162682
        n_male = 121333
        if sex == "female":
            n_train = n_female
        elif sex == "male":
            n_train = n_male

    n_valid = 0
    n_test = 0
    covs = ["imd"]
    method = "feature_map"

    rep = 0
    data_train, data_valid, data_test = care_score2.get_score2_data(
        n_train, n_valid, n_test, covs, sex, dry_run, rep
    )

    # kernel and embedding
    a = 1
    p = 1
    kernel = care_kernels.PolynomialKernel(a, p)
    embedding = care_embedding.Embedding(
        data_train, data_valid, data_test, kernel, method
    )
    X = data_train.X
    T = data_train.T
    I = data_train.I
    f_tilde = data_train.f_tilde
    f_0 = data_train.f_0

    # get high and low imd data sets
    imds = X[:, 0].copy()
    imds.sort()
    imd_med = np.median(imds)
    high_is = [i for i in range(n_train) if X[i, 0] >= imd_med]
    low_is = [i for i in range(n_train) if X[i, 0] < imd_med]
    data_train_high = care_data.Data(
        X[high_is], T[high_is], I[high_is], f_tilde[high_is], f_0[high_is]
    )
    data_train_low = care_data.Data(
        X[low_is], T[low_is], I[low_is], f_tilde[low_is], f_0[low_is]
    )
    embedding_high = care_embedding.Embedding(
        data_train_high, data_valid, data_test, kernel, method
    )
    embedding_low = care_embedding.Embedding(
        data_train_low, data_valid, data_test, kernel, method
    )

    # get breslow estimator
    breslow = embedding.data["train"].get_breslow()
    breslow_high = embedding_high.data["train"].get_breslow()
    breslow_low = embedding_low.data["train"].get_breslow()
    results = np.concat(
        [T.reshape(-1, 1), I.reshape(-1, 1), X, breslow.reshape(-1, 1)], axis=1
    )
    results = pd.DataFrame(results, columns=["T", "I", "imd", "breslow"])
    results["breslow_high"] = np.nan
    results.loc[high_is, "breslow_high"] = breslow_high
    results["breslow_low"] = np.nan
    results.loc[low_is, "breslow_low"] = breslow_low

    # write breslow results
    path = f"./data/{today}/score2/illustration_score2_{sex}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path)


if __name__ == "__main__":
    main()

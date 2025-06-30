import numpy as np
import pandas as pd

from care_survival import data as care_data


def get_score2_data(n_train, n_valid, n_test, covs, sex, dry_run, rep):
    # read data
    if dry_run:
        file = pd.read_csv("./data/score2_test.csv")

    else:
        path = "~/rds/rds-ceu-ukbiobank-RtePkTecWB4/projects/"
        path += f"P7439/lambertlab/wgu21/data/df_scaled_{sex}.csv"
        file = pd.read_csv(path)

    n_avail = len(file)

    # get a random ordering of all the samples
    np.random.seed(0)
    all_is = np.array(list(range(n_avail)))
    np.random.shuffle(all_is)

    # fix the first n_test for a predictable test set
    test_is = all_is[0:n_test]

    # shuffle the remaining samples by the value of rep
    np.random.seed(rep)
    other_is = all_is[n_test:n_avail]
    np.random.shuffle(other_is)

    # make sure no empty sets
    if n_train == 0:
        train_is = [0]
    else:
        train_is = other_is[0:n_train]
    if n_valid == 0:
        valid_is = [0]
    else:
        valid_is = other_is[n_train : n_train + n_valid]
    if n_test == 0:
        test_is = [0]

    # get the data
    X = np.array(file[covs])
    T = np.array(file["time"])
    I = np.array(file["censored"])
    score2_rel = np.array(file[["score2_rel"]])
    f_0 = np.full((len(I), 1), np.nan)

    data_train = care_data.Data(
        X[train_is], T[train_is], I[train_is], score2_rel[train_is], f_0
    )
    data_valid = care_data.Data(
        X[valid_is], T[valid_is], I[valid_is], score2_rel[valid_is], f_0
    )
    data_test = care_data.Data(
        X[test_is], T[test_is], I[test_is], score2_rel[test_is], f_0
    )

    return (data_train, data_valid, data_test)

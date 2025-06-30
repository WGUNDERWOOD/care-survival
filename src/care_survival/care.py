import numpy as np

from care_survival import data as care_data
from care_survival import embedding as care_embedding
from care_survival import aggregation as care_aggregation


def care(
    X_train,
    T_train,
    I_train,
    f_train,
    X_valid,
    T_valid,
    I_valid,
    f_valid,
    kernel,
    method,
    n_gammas,
    gamma_min,
    gamma_max,
    simplex_resolution,
    with_concordance,
    verbose=False,
):
    # get parameters from shape of data
    n_train = X_train.shape[0]
    n_valid = X_valid.shape[0]
    d = X_train.shape[1]
    m = f_train.shape[1]

    # rescale T
    T_min = min(np.min(T_train), np.min(T_valid))
    T_max = max(np.max(T_train), np.max(T_valid))
    T_range = T_max - T_min
    T_train = (T_train - T_min) / T_range
    T_valid = (T_valid - T_min) / T_range

    # construct empty test data
    X_test = np.empty((0, d))
    T_test = np.empty(0)
    I_test = np.empty(0)
    f_test = np.empty((0, m))

    # no ground truth
    f_0_train = np.full(n_train, np.nan)
    f_0_valid = np.full(n_valid, np.nan)
    f_0_test = np.full(0, np.nan)

    # get data objects
    data_train = care_data.Data(X_train, T_train, I_train, f_train, f_0_train)
    data_valid = care_data.Data(X_valid, T_valid, I_valid, f_valid, f_0_valid)
    data_test = care_data.Data(X_test, T_test, I_test, f_test, f_0_test)

    # compute kernel embedding
    embedding = care_embedding.Embedding(
        data_train, data_valid, data_test, kernel, method
    )

    # fit CARE estimator
    care = care_aggregation.CARE(
        embedding,
        gamma_min,
        gamma_max,
        n_gammas,
        simplex_resolution,
        with_concordance,
        verbose,
    )
    care.fit()

    return care

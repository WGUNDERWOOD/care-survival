import sys
from datetime import datetime
import os
import numpy as np
import pandas as pd

from care_survival import kernels as care_kernels
from care_survival import embedding as care_embedding
from care_survival import aggregation as care_aggregation
from care_survival import distributions as care_distributions


def main():
    dgp = int(sys.argv[1])
    today = datetime.now().strftime("%Y-%m-%d")
    np.random.seed(4)

    # data
    n = 200
    distribution = care_distributions.get_distribution(dgp)
    data_train = distribution.sample(n)
    data_valid = distribution.sample(n)
    data_test = distribution.sample(n)

    # kernel
    a = 1
    kernel = care_kernels.ShiftedFirstOrderSobolevKernel(a)
    method = "kernel"
    embedding = care_embedding.Embedding(
        data_train, data_valid, data_test, kernel, method
    )

    # run CARE
    n_gammas = 50
    gamma_min = 1e-5
    gamma_max = 1e1
    simplex_resolution = 1
    with_concordance = []
    verbose = True
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

    # write validation results
    path = f"./data/{today}/simulation/illustration_validation_dgp_{dgp}.csv"
    write_validation(care, path)

    # write best estimator results
    best = care.best["kernel"]["ln"]["valid"]
    path = f"./data/{today}/simulation/illustration_estimator_dgp_{dgp}.csv"
    write_estimator(best, path)


def write_validation(care, path):
    ks = care.kernel_estimators
    results = pd.DataFrame(
        {
            "gamma": [k.kernel_estimator.gamma for k in ks],
            "ln_train": [k.score["ln"]["train"] for k in ks],
            "ln_valid": [k.score["ln"]["valid"] for k in ks],
            "l2": [k.score["l2"]["test"] for k in ks],
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path)


def write_estimator(convex_estimator, path):
    kernel_estimator = convex_estimator.kernel_estimator
    embedding = kernel_estimator.embedding
    data_test = embedding.data["test"]
    d = data_test.d
    X = pd.DataFrame(embedding.data["test"].X, columns=[f"X{j + 1}" for j in range(d)])
    n = data_test.n
    results = pd.DataFrame(
        {
            "gamma": [kernel_estimator.gamma for _ in range(n)],
            "T": data_test.T,
            "I": data_test.I,
            "f_hat": kernel_estimator.f_hat["test"],
            "f_0": data_test.f_0,
            "breslow": data_test.breslow,
            "l2": [convex_estimator.score["l2"]["test"] for _ in range(n)],
        }
    )
    results = pd.concat([X, results], axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path)


if __name__ == "__main__":
    main()

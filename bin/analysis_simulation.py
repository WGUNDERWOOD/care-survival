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
    rep = int(sys.argv[2])
    today = datetime.now().strftime("%Y-%m-%d")

    ns = [
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        120,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
    ]
    n_test = 500

    distribution = care_distributions.get_distribution(dgp)
    a = 1
    kernel = care_kernels.ShiftedFirstOrderSobolevKernel(a)
    n_gammas = 5
    gamma_min = 1e-5
    gamma_max = 1e1
    ns.sort(reverse=True)
    method = "kernel"
    simplex_resolution = 0.05
    np.random.seed(rep)
    with_concordance = []
    verbose = False
    cares = []

    for n in ns:
        now = datetime.now().strftime("%H:%M:%S.%f")
        print(f"{now}, dgp = {dgp}, rep = {rep}, n = {n}", flush=True)
        # data
        data_train = distribution.sample(n)
        data_valid = distribution.sample(n)
        data_test = distribution.sample(n_test)
        embedding = care_embedding.Embedding(
            data_train, data_valid, data_test, kernel, method
        )

        # fit care estimator
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
        cares.append(care)

    # write summary results
    path = f"./data/{today}/simulation/analysis/"
    path += f"analysis_simulation_dgp_{dgp}_rep_{rep}.csv"
    write_summary(cares, rep, path)


def write_summary(cares, rep, path):
    results = pd.concat([pd.DataFrame(c.summary) for c in cares], axis=0)
    results["n"] = results.n_train
    results["rep"] = rep
    results = results.drop(["n_train", "n_valid"], axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path)


if __name__ == "__main__":
    main()

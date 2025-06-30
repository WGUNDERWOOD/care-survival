import sys
import os
import pandas as pd
from datetime import datetime

from care_survival import aggregation as care_aggregation
from care_survival import embedding as care_embedding
from care_survival import kernels as care_kernels
from care_survival import score2 as care_score2


def main():
    # args
    today = datetime.now().strftime("%Y-%m-%d")
    model = int(sys.argv[1])
    sex = sys.argv[2]
    rep = int(sys.argv[3])
    method = "feature_map"
    _n_female = 162682
    _n_male = 121333
    n_female_over_3 = 54227
    n_male_over_3 = 40444
    dry_run = False

    # set up parameters
    if dry_run:
        ns = [10, 20]
        n_test = 20
    else:
        ns = [
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
            12000,
            14000,
            16000,
            18000,
            20000,
            25000,
            30000,
            35000,
        ]
        if sex == "female":
            ns.append([40000, 45000, 50000, n_female_over_3])
            n_test = n_female_over_3
        elif sex == "male":
            ns.append([n_male_over_3])
            n_test = n_male_over_3

    # more set-up
    n_gammas = 50
    gamma_min = 1e-8
    gamma_max = 1e-2
    covs = get_covs(model)
    simplex_resolution = 0.05
    a = 1
    p = 2
    kernel = care_kernels.PolynomialKernel(a, p)
    with_concordance = ["test"]
    verbose = False
    ns.sort(reverse=True)
    cares = []

    for n in ns:
        now = datetime.now().strftime("%H:%M:%S.%f")
        print(f"{now}, model = {model}, sex = {sex}, rep = {rep}, n = {n}", flush=True)

        # get data
        (data_train, data_valid, data_test) = care_score2.get_score2_data(
            n, n, n_test, covs, sex, dry_run, rep
        )
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
    path = f"./data/{today}/score2/analysis/"
    path += f"analysis_score2_model_{model}_{sex}_rep_{rep}.csv"
    write_summary(cares, rep, model, sex, path)


def get_covs(model):
    score2_covs = [
        "age",
        "hdl",
        "sbp",
        "tchol",
        "smoking",
        "age_hdl",
        "age_sbp",
        "age_tchol",
        "age_smoking",
    ]
    if model == 1:
        score2_covs += ["imd"]
    elif model == 2:
        score2_covs += ["imd", "pgs000018", "pgs000039"]

    return score2_covs


def write_summary(cares, rep, model, sex, path):
    results = pd.concat([pd.DataFrame(c.summary) for c in cares], axis=0)
    results["n"] = results.n_train
    results["model"] = model
    results["sex"] = sex
    results["rep"] = rep
    results = results.drop(
        [
            "n_train",
            "n_valid",
            "n_test",
            "gamma_star",
            "gamma_dagger",
            "theta_dagger",
            "l2_star",
            "l2_hat",
            "l2_dagger",
            "l2_check",
            "l2_tilde",
            "concordance_star",
            "concordance_dagger",
        ],
        axis=1,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path)


if __name__ == "__main__":
    main()

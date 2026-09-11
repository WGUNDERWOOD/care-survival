import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

from care_survival import aggregation as care_aggregation
from care_survival import embedding as care_embedding
from care_survival import kernels as care_kernels
from care_survival import score2 as care_score2
from care_survival import kernel_estimator as care_kernel_estimator
from care_survival import metrics as care_metrics


def main():
    # args
    today = datetime.now().strftime("%Y-%m-%d")
    model = int(sys.argv[1])
    sex = sys.argv[2]
    rep = int(sys.argv[3])
    _n_female = 162682
    _n_male = 121333
    n_female_over_3 = 54227
    n_male_over_3 = 40444
    dry_run = False
    #dry_run = True

    # set up parameters
    if dry_run:
        n = 20
        n_test = 20
    else:
        if sex == "female":
            n = n_female_over_3
            n_test = n_female_over_3
            #n = 10000
            #n_test = 10000
        elif sex == "male":
            n = n_male_over_3
            n_test = n_male_over_3
            #n = 4000
            #n_test = 4000

    # more set-up
    a = 1
    (covs, p, gamma_min, gamma_max, n_gammas, kernel, method) = get_model_params(model, a)
    simplex_resolution = 0.05
    with_metrics = {
        "ln": ["train", "valid", "test"],
        "concordance": [],
        "brier": [],
    }
    n_brier_ts = 25
    brier_ts = np.linspace(0, 1, num=n_brier_ts)

    verbose = True
    #verbose = False
    cares = []

    nystrom_m = 100
    now = datetime.now().strftime("%H:%M:%S.%f")
    print(f"{now}, model = {model}, sex = {sex}, rep = {rep}, n = {n}", flush=True)

    # get data
    (data_train, data_valid, data_test) = care_score2.get_score2_data(
        n, n, n_test, covs, sex, dry_run, rep
    )
    embedding = care_embedding.Embedding(
        data_train, data_valid, data_test, kernel, method, nystrom_m
    )

    # fit care estimator
    care = care_aggregation.CARE(
        embedding,
        gamma_min,
        gamma_max,
        n_gammas,
        simplex_resolution,
        with_metrics,
        brier_ts,
        verbose,
    )
    care.fit()
    cares.append(care)

    # model survival probability
    ts = np.array([1.0])
    f_check_train = care.best["aggregated"]["ln"]["valid"].f_check["train"]
    f_check_test = care.best["aggregated"]["ln"]["valid"].f_check["test"]
    T = embedding.data["train"].T
    N = embedding.data["train"].N
    sn_check = care_kernel_estimator.get_sn(embedding.data["train"], np.exp(f_check_train))
    survival_prob_check = care_metrics.get_survival_prob(T, N, sn_check, f_check_test, ts)

    # SCORE2 survival probability
    f_tilde_train = care.best["external"]["ln"]["valid"].f_check["train"]
    f_tilde_test = care.best["external"]["ln"]["valid"].f_check["test"]
    sn_tilde = care_kernel_estimator.get_sn(embedding.data["train"], np.exp(f_tilde_train))
    survival_prob_tilde = care_metrics.get_survival_prob(T, N, sn_tilde, f_tilde_test, ts)

    results = pd.DataFrame({
        "survival_prob_tilde": survival_prob_tilde,
        "survival_prob_check": survival_prob_check,
    })
    #print(results)

    # plot
    #import plotille
    #fig = plotille.Figure()
    #fig.width = 60
    #fig.height = 30
    #fig.scatter(1-survival_prob_tilde, 1-survival_prob_check)
    #print(fig.show())

    # write results
    path = f"./data/{today}/score2/scatter/"
    path += f"scatter_score2_model_{model}_{sex}_rep_{rep}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path)
    print(f"{now}, model = {model}, sex = {sex}, rep = {rep}, done", flush=True)


def get_model_params(model, a):
    covs = [
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

    # model 1: no new covariates, RKHS predictor
    # model 2: add imd, RKHS predictor
    # model 3: add imd and pgs, RKHS predictor
    # model 4: add imd, linear predictor, no regularisation
    # model 5: add imd and pgs, linear predictor, no regularisation

    # models 6--10 are the same but with a Sobolev kernel and Nystrom

    # estimator
    if model in [1, 2, 3, 6, 7, 8]:
        p = 2
        gamma_min = 1e-8
        gamma_max = 1e-2
        n_gammas = 50
    elif model in [4, 5, 9, 10]:
        p = 1
        gamma_min = 0.0
        gamma_max = 0.0
        n_gammas = 1
    if model in [1, 2, 3, 4, 5]:
        kernel = care_kernels.PolynomialKernel(a, p)
        method = "feature_map"
    elif model in [6, 7, 8, 9, 10]:
        kernel = care_kernels.ShiftedFirstOrderSobolevKernel(a)
        method = "kernel"

    # covariates
    if model in [2, 4, 7, 9]:
        covs += ["imd"]
    elif model in [3, 5, 8, 10]:
        covs += ["imd", "pgs000018", "pgs000039"]

    return (covs, p, gamma_min, gamma_max, n_gammas, kernel, method)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import sys
import glob
import os
import pandas as pd
import numpy as np
import common

common.init()


def plot_regularisation(csv_path, plot_path, dgp):
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    csv_files = [f for f in csv_files if "dgp_" + dgp in f]
    df_all = pd.concat(pd.read_csv(f) for f in csv_files)
    df_all["log_gamma_hat"] = np.log(df_all["gamma_hat"])
    df_all["log_gamma_star"] = np.log(df_all["gamma_star"])
    df_all = df_all[df_all["n"] >= 30]
    df = df_all.groupby("n").mean()
    n_rep = df_all["rep"].nunique()
    df_sd = df_all.groupby("n").std() / (n_rep**0.5)
    cols = ["log_gamma_hat", "log_gamma_star"]
    for c in cols:
        df[c + "_std"] = df_sd[c]
    df = df.sort_values(by="n")
    (fig, ax) = plt.subplots(figsize=(4, 3))

    # plot averages
    plt.plot(
        df.index,
        np.exp(df["log_gamma_hat"]),
        c="k",
        lw=1,
        label="Cross-validated $\\hat\\gamma$",
    )
    plt.plot(
        df.index,
        np.exp(df["log_gamma_star"]),
        c="k",
        lw=1,
        ls="--",
        label="Oracle $\\gamma^\\star$",
    )

    # plot error band
    for c in cols:
        plt.fill_between(
            df.index,
            np.exp(df[c] - 2 * df[c + "_std"]),
            np.exp(df[c] + 2 * df[c + "_std"]),
            fc=common.std_col(),
        )

    if dgp == "1":
        plt.ylim([9.2e-4, 1.1e-2])
    elif dgp == "2":
        plt.ylim([9.2e-4, 3.3e-2])

    ax.set_yscale("log")
    plt.xlabel("Sample size $n$")
    plt.ylabel("Regularisation parameter $\\gamma$")
    plt.legend()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


for dgp in ["1", "2"]:
    date = sys.argv[1]
    csv_path = "data/" + date + "/simulation/analysis/"
    plot_path = "plot/regularisation_dgp_" + dgp + ".pdf"
    plot_regularisation(csv_path, plot_path, dgp)

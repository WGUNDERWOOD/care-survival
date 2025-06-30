import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
import os
import pandas as pd
import common


def plot_aggregation(csv_path, plot_path, dgp):
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    csv_files = [f for f in csv_files if "dgp_" + dgp in f]
    df_all = pd.concat(pd.read_csv(f) for f in csv_files)
    df_all = df_all[df_all["n"] >= 30]
    df = df_all.groupby("n").mean()
    n_rep = df_all["rep"].nunique()
    df_sd = df_all.groupby("n").std() / (n_rep**0.5)
    cols = ["l2_tilde", "l2_hat", "l2_check", "l2_dagger"]
    for c in cols:
        df[c + "_std"] = df_sd[c]
    df = df.sort_values(by="n")
    df["l2_tilde"] = np.mean(df["l2_tilde"])
    (fig, ax) = plt.subplots(figsize=(4, 3))

    # plot error band
    for c in cols:
        if c != "l2_tilde":
            plt.fill_between(
                df.index,
                df[c] - 2 * df[c + "_std"],
                df[c] + 2 * df[c + "_std"],
                fc=common.std_col(),
            )

    # plot averages
    plt.plot(
        df.index,
        df["l2_check"],
        c="k",
        lw=1,
        label="CARE method $\\check f_{n,\\check\\gamma,\\check\\theta}$",
    )
    plt.plot(
        df.index,
        df["l2_hat"],
        c="k",
        lw=1,
        ls="-.",
        label="Kernel estimator $\\hat f_{n,\\hat\\gamma}$",
    )
    plt.plot(
        df.index,
        df["l2_dagger"],
        c="k",
        lw=1,
        ls="--",
        label="Oracle $\\check f_{n,\\gamma^\\dagger,\\theta^\\dagger}$",
    )
    plt.plot(
        df.index, df["l2_tilde"], c="k", lw=1, ls=":", label="External $\\tilde f$"
    )

    if dgp == "2":
        plt.ylim([0.33, 1.22])

    plt.xlabel("Sample size $n$")
    plt.ylabel("$L_2$-error")
    plt.legend()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


for dgp in ["1", "2"]:
    date = sys.argv[1]
    csv_path = "data/" + date + "/simulation/analysis/"
    plot_path = "plot/aggregation_dgp_" + dgp + ".pdf"
    plot_aggregation(csv_path, plot_path, dgp)

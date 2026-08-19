import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
import os
import pandas as pd
import common


def plot_aggregation(csv_path, plot_path, dgp, metric):
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    csv_files = [f for f in csv_files if "dgp_" + dgp in f]
    df_all = pd.concat(pd.read_csv(f) for f in csv_files)
    df_all = df_all[df_all["n"] >= 30]
    df = df_all.groupby("n").mean()
    n_rep = df_all["rep"].nunique()
    df_sd = df_all.groupby("n").std() / (n_rep**0.5)
    cols = [f"{metric}_tilde", f"{metric}_hat", f"{metric}_check", f"{metric}_dagger"]
    for c in cols:
        df[c + "_std"] = df_sd[c]
    df = df.sort_values(by="n")
    df[f"{metric}_tilde"] = np.mean(df[f"{metric}_tilde"])
    (fig, ax) = plt.subplots(figsize=(4, 3))


    # plot error band
    for c in cols:
        if c != f"{metric}_tilde":
            plt.fill_between(
                df.index,
                df[c] - 2 * df[c + "_std"],
                df[c] + 2 * df[c + "_std"],
                fc=common.std_col(),
            )

    # plot averages
    plt.plot(
        df.index,
        df[f"{metric}_check"],
        c="k",
        lw=1,
        label="CARE method $\\check f_{n,\\check\\gamma,\\check\\theta}$",
    )
    plt.plot(
        df.index,
        df[f"{metric}_hat"],
        c="k",
        lw=1,
        ls="-.",
        label="Kernel estimator $\\hat f_{n,\\hat\\gamma}$",
    )
    plt.plot(
        df.index,
        df[f"{metric}_dagger"],
        c="k",
        lw=1,
        ls="--",
        label="Oracle $\\check f_{n,\\gamma^\\dagger,\\theta^\\dagger}$",
    )
    plt.plot(
        df.index, df[f"{metric}_tilde"], c="k", lw=1, ls=":", label="External $\\tilde f$"
    )

    if metric == "l2":
        plt.ylabel("$L_2$-error")
    elif metric == "concordance":
        plt.ylabel("Concordance index")
    elif metric == "brier":
        plt.ylabel("Integrated Brier score")

    plt.xlabel("Sample size $n$")
    plt.legend()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


#for dgp in [str(i) for i in range(1, 7)]:
for dgp in ["4", "6"]:
    #for metric in ["l2", "concordance", "brier"]:
    for metric in ["l2"]:
        date = sys.argv[1]
        csv_path = "data/" + date + "/simulation/analysis/"
        plot_path = "plot/aggregation_dgp_" + dgp + "_" + metric + ".pdf"
        plot_aggregation(csv_path, plot_path, dgp, metric)

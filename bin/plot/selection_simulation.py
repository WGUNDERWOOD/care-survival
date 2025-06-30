import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import numpy as np
import common
import sys


def plot_selection(csv_path, plot_path, dgp):
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    csv_files = [f for f in csv_files if "dgp_" + dgp in f]
    df_all = pd.concat(pd.read_csv(f) for f in csv_files)
    df_all = df_all[df_all["n"] >= 30]
    df = df_all.groupby("n").mean()
    n_rep = df_all["rep"].nunique()
    df_sd = df_all.groupby("n").std() / (n_rep**0.5)
    cols = ["theta_dagger", "theta_check"]
    for c in cols:
        df[c + "_std"] = df_sd[c]
    df = df.sort_values(by="n")
    (fig, ax) = plt.subplots(figsize=(4, 3))

    # plot averages
    plt.plot(
        df.index,
        df["theta_check"],
        c="k",
        lw=1,
        label="Cross-validated $\\check\\theta$",
    )
    plt.plot(
        df.index,
        df["theta_dagger"],
        c="k",
        lw=1,
        ls="--",
        label="Oracle $\\theta^\\dagger$",
    )

    # plot error band
    for c in cols:
        plt.fill_between(
            df.index,
            np.maximum(df[c] - 2 * df[c + "_std"], 0),
            np.minimum(df[c] + 2 * df[c + "_std"], 1),
            fc=common.std_col(),
        )

    plt.xlabel("Sample size $n$")
    plt.ylabel("Convex combination $\\theta$")
    plt.legend()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


for dgp in ["1", "2"]:
    date = sys.argv[1]
    csv_path = "data/" + date + "/simulation/analysis/"
    plot_path = "plot/selection_dgp_" + dgp + ".pdf"
    plot_selection(csv_path, plot_path, dgp)

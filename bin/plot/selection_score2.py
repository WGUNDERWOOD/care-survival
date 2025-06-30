import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os
import pandas as pd
import numpy as np
import common
import sys


def plot_selection_score2(csv_path, plot_path, sex, model):
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    csv_files = [f for f in csv_files if "model_" + model + "_" + sex in f]
    df_all = pd.concat(pd.read_csv(f) for f in csv_files)
    df_all = df_all.drop(["sex"], axis=1)
    df = df_all.groupby("n").mean()
    n_rep = df_all["rep"].nunique()
    df_sd = df_all.groupby("n").std() / (n_rep**0.5)
    cols = ["theta_check"]

    for c in cols:
        df[c + "_std"] = df_sd[c]
    df = df.sort_values(by="n")
    (fig, ax) = plt.subplots(figsize=(4, 3))

    # plot error bands
    for c in cols:
        plt.fill_between(
            df.index,
            np.maximum(df[c] - 2 * df[c + "_std"], 0),
            np.minimum(df[c] + 2 * df[c + "_std"], 1),
            fc=common.std_col(),
        )

    # plot averages
    plt.plot(
        df.index,
        df["theta_check"],
        c="k",
        lw=1,
        label="Cross-validated $\\check\\theta$",
    )

    plt.xlabel("Training/validation sample size $n$")
    plt.ylabel("Convex combination $\\theta$")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.legend()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


for model in ["1", "2"]:
    for sex in ["female", "male"]:
        date = sys.argv[1]
        csv_path = "data/" + date + "/score2/analysis/"
        plot_path = "plot/selection_score2_model_" + model + "_" + sex + ".pdf"
        plot_selection_score2(csv_path, plot_path, sex, model)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os
import pandas as pd
import numpy as np
import sys
import common


def plot_aggregation_score2(csv_path, plot_path, sex):
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    csv_files = [f for f in csv_files if "model_" + model + "_" + sex in f]
    df_all = pd.concat(pd.read_csv(f) for f in csv_files)
    df_all = df_all.drop(["sex"], axis=1)
    df = df_all.groupby("n").mean()
    n_rep = df_all["rep"].nunique()
    df_sd = df_all.groupby("n").std() / (n_rep**0.5)
    ct_all = df["concordance_tilde"][df.index == max(df.index)]
    cc_all = df["concordance_check"][df.index == max(df.index)]
    print(100 * ((cc_all - ct_all) / ct_all).values[0])
    cols = ["concordance_check", "concordance_hat", "concordance_tilde"]

    for c in cols:
        df[c + "_std"] = df_sd[c]
    df = df.sort_values(by="n")
    df["concordance_tilde"] = np.mean(df["concordance_tilde"])
    (fig, ax) = plt.subplots(figsize=(4, 3))

    # plot error bands
    for c in cols:
        if c != "concordance_tilde":
            plt.fill_between(
                df.index,
                df[c] - 2 * df[c + "_std"],
                df[c] + 2 * df[c + "_std"],
                fc=common.std_col(),
            )

    # plot averages
    plt.plot(
        df.index,
        df["concordance_check"],
        c="k",
        lw=1,
        label="CARE method $\\check f_{n,\\check\\gamma,\\check\\theta}$",
    )
    plt.plot(
        df.index,
        df["concordance_hat"],
        c="k",
        lw=1,
        ls="-.",
        label="Kernel estimator $\\hat f_{n,\\hat\\gamma}$",
    )
    plt.plot(
        df.index,
        df["concordance_tilde"],
        c="k",
        lw=1,
        ls=":",
        label="SCORE2 model $\\tilde f$",
    )

    if sex == "male" and model == "2":
        plt.legend(loc="lower right", bbox_to_anchor=(0.5, 0.08, 0.5, 0.5))
    else:
        plt.legend()

    if model == "2" and sex == "female":
        plt.ylim([0.7345, 0.753])
        plt.yticks([0.735, 0.740, 0.745, 0.750])
    if model == "1" and sex == "male":
        plt.ylim([0.684, 0.7005])
        plt.yticks([0.685, 0.690, 0.695, 0.700])

    plt.xlabel("Training/validation sample size $n$")
    plt.ylabel("Concordance index")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


for model in ["1", "2"]:
    for sex in ["female", "male"]:
        date = sys.argv[1]
        csv_path = "data/" + date + "/score2/analysis/"
        plot_path = "plot/aggregation_score2_model_" + model + "_" + sex + ".pdf"
        plot_aggregation_score2(csv_path, plot_path, sex)

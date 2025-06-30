import matplotlib.pyplot as plt
import glob
import sys
import os
import pandas as pd
import common


def plot_kernel(csv_path, plot_path, dgp):
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    csv_files = [f for f in csv_files if "dgp_" + dgp in f]
    df_all = pd.concat(pd.read_csv(f) for f in csv_files)
    df_all = df_all[df_all["n"] >= 30]
    df = df_all.groupby("n").mean()
    n_rep = df_all["rep"].nunique()
    df_sd = df_all.groupby("n").std() / (n_rep**0.5)
    cols = ["l2_hat", "l2_star"]
    for c in cols:
        df[c + "_std"] = df_sd[c]
    df = df.sort_values(by="n")
    (fig, ax) = plt.subplots(figsize=(4, 3))

    # plot error band
    for c in cols:
        plt.fill_between(
            df.index,
            df[c] - 2 * df[c + "_std"],
            df[c] + 2 * df[c + "_std"],
            fc=common.std_col(),
        )

    # plot averages
    plt.plot(
        df.index,
        df["l2_hat"],
        c="k",
        lw=1,
        label="Cross-validated $\\hat f_{n,\\hat\\gamma}$",
    )
    plt.plot(
        df.index,
        df["l2_star"],
        c="k",
        lw=1,
        ls="--",
        label="Oracle $\\hat f_{n,\\gamma^\\star}$",
    )

    plt.xlabel("Sample size $n$")
    plt.ylabel("$L_2$-error")
    plt.legend()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


for dgp in ["1", "2"]:
    date = sys.argv[1]
    csv_path = "data/" + date + "/simulation/analysis/"
    plot_path = "plot/kernel_dgp_" + dgp + ".pdf"
    plot_kernel(csv_path, plot_path, dgp)

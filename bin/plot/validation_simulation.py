import matplotlib.pyplot as plt
import pandas as pd
import sys
import common

common.init()


def plot_validation(csv_path, plot_path, dgp):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="gamma")
    (fig, ax) = plt.subplots(figsize=(4, 3))

    # validation curves
    plt.plot(
        df["gamma"],
        df["ln_train"],
        c="k",
        lw=1,
        ls="--",
        label="Training loss $\\ell_n(\\hat f_{n,\\gamma})$",
    )
    plt.plot(
        df["gamma"],
        df["ln_valid"],
        c="k",
        lw=1,
        label="Validation loss $\\tilde \\ell_n(\\hat f_{n,\\gamma})$",
    )

    # indicate gamma_hat
    best_ln = min(df["ln_valid"])
    best_index = [i for i in df.index if df["ln_valid"][i] == best_ln][0]
    best_gamma = df["gamma"][best_index]
    plt.scatter(
        best_gamma,
        best_ln,
        lw=1,
        c="k",
        marker="|",
        label="Cross-validated $\\hat\\gamma$",
    )

    plt.ylabel("Negative partial log-likelihood")
    plt.xlabel("Regularisation parameter $\\gamma$")
    ax.set_xscale("log")
    ax.legend()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


for dgp in ["1", "2"]:
    date = sys.argv[1]
    csv_path = "data/" + date + "/simulation"
    csv_path = csv_path + "/illustration_validation_dgp_" + dgp + ".csv"
    plot_path = "plot/validation_dgp_" + dgp + ".pdf"
    plot_validation(csv_path, plot_path, dgp)

import matplotlib.pyplot as plt
import pandas as pd
import sys
import common

common.init()


def plot_scatter(csv_path, plot_path, dgp):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="X1")
    (fig, ax) = plt.subplots(figsize=(4, 3))
    plt.scatter(
        df[~df["I"]]["X1"],
        df[~df["I"]]["T"],
        c="k",
        s=6,
        label="True events ($I_i = 0$)",
    )
    plt.scatter(
        df[df["I"]]["X1"],
        df[df["I"]]["T"],
        facecolors="none",
        s=8,
        edgecolors="k",
        lw=0.6,
        label="Censored events ($I_i = 1$)",
    )
    plt.xlabel("Covariate $X_i$")
    plt.ylabel("Time $T_i$")
    plt.legend()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


dgp = "1"
date = sys.argv[1]
csv_path = "data/" + date + "/simulation"
csv_path = csv_path + "/illustration_estimator_dgp_" + dgp + ".csv"
plot_path = "plot/scatter_dgp_" + dgp + ".pdf"
plot_scatter(csv_path, plot_path, dgp)

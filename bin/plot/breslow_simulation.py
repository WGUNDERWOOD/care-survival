import matplotlib.pyplot as plt
import sys
import pandas as pd
import common

common.init()


def plot_breslow(csv_path, plot_path, dgp):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="T")
    (fig, ax) = plt.subplots(figsize=(4, 3))
    plt.step(
        df["T"], df["breslow"], where="post", c="k", lw=1, label="Breslow estimator"
    )
    plt.scatter(
        df[df["I"]]["T"],
        df[df["I"]]["breslow"],
        facecolors="none",
        s=16,
        edgecolors="k",
        lw=0.9,
        label="Censored events ($I_i = 1$)",
    )
    plt.xlabel("Time $t$")
    plt.ylabel("Survival function $\\hat P(t)$")
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.legend()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


for dgp in ["1", "2"]:
    date = sys.argv[1]
    csv_path = "data/" + date + "/simulation"
    csv_path = csv_path + "/illustration_estimator_dgp_" + dgp + ".csv"
    plot_path = "plot/breslow_dgp_" + dgp + ".pdf"
    plot_breslow(csv_path, plot_path, dgp)

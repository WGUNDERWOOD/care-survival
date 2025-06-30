import matplotlib.pyplot as plt
import pandas as pd
import common
import sys

common.init()


def plot_estimator(csv_path, plot_path, dgp):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="X1")
    (fig, ax) = plt.subplots(figsize=(4, 3))
    plt.plot(df["X1"], df["f_0"], c="k", lw=1, label="True relative risk $f_0(x)$")
    plt.plot(
        df["X1"],
        df["f_hat"],
        c="k",
        lw=1,
        ls="--",
        label="Cross-validated $\\hat f_{n,\\hat\\gamma}(x)$",
    )
    plt.legend(loc="lower right")
    plt.xlabel("Covariate $x$")
    plt.ylabel("Relative risk")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


dgp = "1"
date = sys.argv[1]
csv_path = "data/" + date + "/simulation"
csv_path = csv_path + "/illustration_estimator_dgp_" + dgp + ".csv"
plot_path = "plot/estimator_dgp_" + dgp + ".pdf"
plot_estimator(csv_path, plot_path, dgp)

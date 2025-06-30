import matplotlib.pyplot as plt
import pandas as pd
import sys
import common


def plot_breslow_score2(csv_path, plot_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="T")
    df = df.drop_duplicates()
    df = common.filter_cumulative_increase(df, 0.005, "T")
    df_high = df[~df["breslow_high"].isna()]
    df_low = df[~df["breslow_low"].isna()]
    (fig, ax) = plt.subplots(figsize=(4, 3))

    # low IMD
    plt.plot(
        10 * df_low["T"],
        df_low["breslow_low"],
        c="k",
        lw=1,
        label="Low \\texttt{imd}",
        ls="dotted",
    )

    # high IMD
    plt.plot(
        10 * df_high["T"],
        df_high["breslow_high"],
        c="k",
        lw=1,
        label="High \\texttt{imd}",
        ls="dashed",
    )

    # baseline
    plt.plot(10 * df["T"], df["breslow"], c="k", lw=1, label="All")

    plt.xlabel("Time $t$ (years)")
    plt.ylabel("Survival function $\\hat P(t)$")
    plt.legend()

    if "female" in csv_path:
        plt.yticks([0.98, 0.99, 1.00])

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close("all")


for sex in ["female", "male"]:
    date = sys.argv[1]
    csv_path = "data/" + date
    csv_path = csv_path + "/score2/illustration_score2_" + sex + ".csv"
    plot_path = "plot/breslow_score2_" + sex + ".pdf"
    plot_breslow_score2(csv_path, plot_path)

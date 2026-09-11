import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os
import pandas as pd
import numpy as np
import sys
import common


def plot_scatter_score2(csv_path, plot_path, sex):
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    csv_files = [f for f in csv_files if "model_" + model + "_" + sex in f]
    df = pd.concat(pd.read_csv(f) for f in csv_files)
    df = df.drop(["Unnamed: 0"], axis=1)

    # scatter
    (fig, ax) = plt.subplots(figsize=(4, 3))
    min_val = np.min(df)
    plt.plot([0, 1 - min_val], [0, 1 - min_val], c = "#999999", ls="--")
    plt.scatter(1 - df["survival_prob_tilde"], 1 - df["survival_prob_check"],
                rasterized=True, c="#000000", alpha=0.4, s=4, ec="none")
    plt.xlabel("SCORE2 predicted 10-year risk")
    plt.ylabel("CARE predicted 10-year risk")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close("all")

def plot_box_score2(csv_path, plot_path, sex):
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    csv_files = [f for f in csv_files if "model_" + model + "_" + sex in f]
    df = pd.concat(pd.read_csv(f) for f in csv_files)
    df = df.drop(["Unnamed: 0"], axis=1)

    # box
    (fig, ax) = plt.subplots(figsize=(4, 3))
    plt.boxplot([1 - df["survival_prob_tilde"], 1 - df["survival_prob_check"]],
                whis=(0, 100),
                showfliers=False,
                )
    #plt.xlabel("SCORE2 predicted 10-year risk")
    #plt.ylabel("CARE predicted 10-year risk")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close("all")

#for model in ["1"]:
for model in [str(s) for s in range(6, 9)]:
    #for sex in ["female"]:
    for sex in ["female", "male"]:
        print(f"Model {model}, {sex}")
        date = sys.argv[1]
        csv_path = "data/" + date + "/score2/scatter/"
        plot_scatter_path = "plot/scatter_score2_model_" + model + "_" + sex + ".pdf"
        plot_box_path = "plot/box_score2_model_" + model + "_" + sex + ".pdf"
        plot_scatter_score2(csv_path, plot_scatter_path, sex)
        #plot_box_score2(csv_path, plot_box_path, sex)

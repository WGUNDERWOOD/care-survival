import numpy as np
import pandas as pd


def main():
    directory = "~/rds/rds-ceu-ukbiobank-RtePkTecWB4"
    directory += "projects/P7439/lambertlab/wgu21/data"
    path = directory + "df_SCORE2_withexclusions.csv"
    df = pd.read_csv(path)
    df = df[df["imd_country"] == "England"]

    # transform columns
    df["cage"] = (df["ages"] - 60) / 5
    df["chdl"] = (df["hdl"] - 1.3) / 0.5
    df["csbp"] = (df["sbp"] - 120) / 20
    df["ctchol"] = (df["tchol"] - 6) / 1
    df["smoking"] = (df["smallbin"] == "Current") * 1
    df["egfrcreat"] = df["eGFRcreat"]
    df["imd"] = df["imd_min"]
    df["pgs000018"] = df["PGS000018"]
    df["pgs000039"] = df["PGS000039"]
    df["time"] = df["FOLLOWUPTIME_Incident_10year"]
    df["indicator"] = df["PHENOTYPE_Incident_10year"]
    df["score2_rel"] = np.nan

    # add interaction terms
    df["cage_chdl"] = df["cage"] * df["chdl"]
    df["cage_csbp"] = df["cage"] * df["csbp"]
    df["cage_ctchol"] = df["cage"] * df["ctchol"]
    df["cage_smoking"] = df["cage"] * df["smoking"]

    # add SCORE2 relative risk
    for i, row in df.iterrows():
        lp = 0
        cage = row["cage"]
        smoking = int(row["smoking"])
        csbp = row["csbp"]
        ctchol = row["ctchol"]
        chdl = row["chdl"]

        if row["sex"] == "Male":
            lp += cage * np.log(1.50)
            lp += smoking * np.log(1.77)
            lp += csbp * np.log(1.33)
            lp += ctchol * np.log(1.13)
            lp += chdl * np.log(0.80)
            lp += cage * smoking * np.log(0.92)
            lp += cage * csbp * np.log(0.98)
            lp += cage * ctchol * np.log(0.98)
            lp += cage * chdl * np.log(1.04)

        else:
            lp += cage * np.log(1.64)
            lp += smoking * np.log(2.09)
            lp += csbp * np.log(1.39)
            lp += ctchol * np.log(1.11)
            lp += chdl * np.log(0.81)
            lp += cage * smoking * np.log(0.89)
            lp += cage * csbp * np.log(0.97)
            lp += cage * ctchol * np.log(0.98)
            lp += cage * chdl * np.log(1.06)

        df.loc[i, "score2_rel"] = lp

    # rescale columns
    rescale_cols = [
        "cage",
        "chdl",
        "csbp",
        "ctchol",
        "cage_chdl",
        "cage_csbp",
        "cage_ctchol",
        "cage_smoking",
        "crp",
        "egfrcreat",
        "imd",
        "pgs000018",
        "pgs000039",
    ]
    df_min = df.min(axis=0)
    df_max = df.max(axis=0)

    for c in rescale_cols:
        col_min = df_min[c]
        col_max = df_max[c]
        col_range = col_max - col_min
        col_name = f"{c}_scaled"
        df[col_name] = (df[c] - col_min) / col_range

    # select and rename cols
    df["age"] = df["cage_scaled"]
    df["hdl"] = df["chdl_scaled"]
    df["sbp"] = df["csbp_scaled"]
    df["tchol"] = df["ctchol_scaled"]
    df["age_hdl"] = df["cage_chdl_scaled"]
    df["age_sbp"] = df["cage_csbp_scaled"]
    df["age_tchol"] = df["cage_ctchol_scaled"]
    df["age_smoking"] = df["cage_smoking_scaled"]
    df["crp"] = df["crp_scaled"]
    df["egfrcreat"] = df["egfrcreat_scaled"]
    df["imd"] = df["imd_scaled"]
    df["pgs000018"] = df["pgs000018_scaled"]
    df["pgs000039"] = df["pgs000039_scaled"]
    df["time"] = df["time"] / 10
    df["censored"] = 1 - df["indicator"]

    final_cols = [
        "sex",
        "age",
        "hdl",
        "sbp",
        "tchol",
        "smoking",
        "age_hdl",
        "age_sbp",
        "age_tchol",
        "age_smoking",
        "crp",
        "egfrcreat",
        "imd",
        "pgs000018",
        "pgs000039",
        "score2_rel",
        "time",
        "censored",
    ]

    df = df[final_cols]

    df_female = df[df["sex"] == "Female"]
    df_female = df_female.drop("sex").dropna()

    df_male = df[df["sex"] == "Male"]
    df_male = df_male.drop("sex").dropna()

    out_path_male = directory + "df_scaled_male.csv"
    out_path_female = directory + "df_scaled_female.csv"

    df_female.to_csv(out_path_female)
    df_male.to_csv(out_path_male)

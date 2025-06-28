from care_survival import data as care_data

import numpy as np
import pandas as pd

def get_score2_data(n_train, n_valid, n_test, covs, sex, dry_run, rep):
    # read data
    if dry_run:
        file = pd.read_csv("./data/score2_test.csv")

    else:
        path = "~/rds/rds-ceu-ukbiobank-RtePkTecWB4/projects/"
        path += f"P7439/lambertlab/wgu21/data/df_scaled_{sex}.csv"
        file = pd.read_csv(path)

    n = n_train + n_valid + n_test
    n_avail = len(file)
    print(file[0:5])

    # CSV format: X1, ..., Xd, score2_rel, T, I
    d_all = file.shape[1] - 3;
    all_covs = file.columns[0:d_all]
    cov_indices = [i for i in range(d_all) if all_covs[i] in covs]
    d = len(cov_indices)

    # get a random ordering of all the samples
    np.random.seed(0)
    all_is = np.array(list(range(n_avail)))
    np.random.shuffle(all_is)

    # fix the first n_test for a predictable test set
    test_is = all_is[0:n_test]

    # shuffle the remaining samples by the value of rep
    np.random.seed(rep)
    other_is = all_is[n_test:n_avail]
    np.random.shuffle(other_is)
    train_is = other_is[0:n_train]
    valid_is = other_is[n_train:n_train+n_valid]

    # get the data
    X = np.array(file[covs])
    T = np.array(file["time"])
    I = np.array(file["censored"])
    score2_rel = np.array(file[["score2_rel"]])
    f_0 = np.full((len(I), 1), np.nan)

    data_train = care_data.Data(X[train_is], T[train_is], I[train_is], score2_rel[train_is], f_0)
    data_valid = care_data.Data(X[valid_is], T[valid_is], I[valid_is], score2_rel[valid_is], f_0)
    data_test = care_data.Data(X[test_is], T[test_is], I[test_is], score2_rel[test_is], f_0)

    assert len(data_train.T) == n_train
    assert len(data_valid.T) == n_valid
    assert len(data_test.T) == n_test

    return (data_train, data_valid, data_test, score2_rel)

##[must_use]
#pub fn get_score2_rel(x: ArrayView1<f64>, sex: Sex) -> f64 {
#    // beta male
#    let age_beta_male = 1.50_f64.ln();
#    let smoking_beta_male = 1.77_f64.ln();
#    let sbp_beta_male = 1.33_f64.ln();
#    let tchol_beta_male = 1.13_f64.ln();
#    let hdl_beta_male = 0.80_f64.ln();
#    let age_smoking_beta_male = 0.92_f64.ln();
#    let age_sbp_beta_male = 0.98_f64.ln();
#    let age_tchol_beta_male = 0.98_f64.ln();
#    let age_hdl_beta_male = 1.04_f64.ln();
#
#    // beta female
#    let age_beta_female = 1.64_f64.ln();
#    let smoking_beta_female = 2.09_f64.ln();
#    let sbp_beta_female = 1.39_f64.ln();
#    let tchol_beta_female = 1.11_f64.ln();
#    let hdl_beta_female = 0.81_f64.ln();
#    let age_smoking_beta_female = 0.89_f64.ln();
#    let age_sbp_beta_female = 0.97_f64.ln();
#    let age_tchol_beta_female = 0.98_f64.ln();
#    let age_hdl_beta_female = 1.06_f64.ln();
#
#    // scaling min and max
#    let age_min = -3.978;
#    let age_max = 1.9998;
#    let hdl_min = -2.148;
#    let hdl_max = 6.202;
#    let sbp_min = -2.9;
#    let sbp_max = 7.4;
#    let tchol_min = -4.449;
#    let tchol_max = 9.46;
#    let smoking_min = 0.0;
#    let smoking_max = 1.0;
#    let age_hdl_min = -15.330_861_6;
#    let age_hdl_max = 8.137_912_8;
#    let age_sbp_min = -17.076_6;
#    let age_sbp_max = 12.084_8;
#    let age_tchol_min = -23.260_386;
#    let age_tchol_max = 16.704_215_4;
#    let age_smoking_min = -3.972_6;
#    let age_smoking_max = 1.999_8;
#
#    match sex {
#        Sex::Female => {
#            age_beta_female * ((age_max - age_min) * x[0] + age_min)
#                + hdl_beta_female * ((hdl_max - hdl_min) * x[1] + hdl_min)
#                + sbp_beta_female * ((sbp_max - sbp_min) * x[2] + sbp_min)
#                + tchol_beta_female
#                    * ((tchol_max - tchol_min) * x[3] + tchol_min)
#                + smoking_beta_female
#                    * ((smoking_max - smoking_min) * x[4] + smoking_min)
#                + age_hdl_beta_female
#                    * ((age_hdl_max - age_hdl_min) * x[5] + age_hdl_min)
#                + age_sbp_beta_female
#                    * ((age_sbp_max - age_sbp_min) * x[6] + age_sbp_min)
#                + age_tchol_beta_female
#                    * ((age_tchol_max - age_tchol_min) * x[7] + age_tchol_min)
#                + age_smoking_beta_female
#                    * ((age_smoking_max - age_smoking_min) * x[8]
#                        + age_smoking_min)
#        }
#        Sex::Male => {
#            age_beta_male * ((age_max - age_min) * x[0] + age_min)
#                + hdl_beta_male * ((hdl_max - hdl_min) * x[1] + hdl_min)
#                + sbp_beta_male * ((sbp_max - sbp_min) * x[2] + sbp_min)
#                + tchol_beta_male * ((tchol_max - tchol_min) * x[3] + tchol_min)
#                + smoking_beta_male
#                    * ((smoking_max - smoking_min) * x[4] + smoking_min)
#                + age_hdl_beta_male
#                    * ((age_hdl_max - age_hdl_min) * x[5] + age_hdl_min)
#                + age_sbp_beta_male
#                    * ((age_sbp_max - age_sbp_min) * x[6] + age_sbp_min)
#                + age_tchol_beta_male
#                    * ((age_tchol_max - age_tchol_min) * x[7] + age_tchol_min)
#                + age_smoking_beta_male
#                    * ((age_smoking_max - age_smoking_min) * x[8]
#                        + age_smoking_min)
#        }
#    }
#}

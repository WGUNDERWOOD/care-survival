##[must_use]
#pub fn get_score2_data(
#    n_train: usize,
#    n_valid: usize,
#    n_test: usize,
#    covs: &[String],
#    sex: Sex,
#    dry_run: bool,
#    rep: usize,
#) -> (Data, Data, Data, Array1<f64>) {
#    // read data
#    let file = if dry_run {
#        let mut dir = current_dir().unwrap();
#        dir = dir.join("data");
#        let path = dir.join("score2_test.csv");
#        read_to_string(&path).unwrap()
#    } else {
#        let mut dir = home_dir().unwrap();
#        dir = dir.join("rds/rds-ceu-ukbiobank-RtePkTecWB4");
#        dir = dir.join("projects/P7439/lambertlab/wgu21/data");
#        let path = dir.join(format!("df_scaled_{sex}.csv"));
#        read_to_string(&path).unwrap()
#    };
#
#    let n = n_train + n_valid + n_test;
#    let n_avail = file.lines().count() - 1;
#    //dbg!(n_avail);
#    assert!(n <= n_avail);
#    // CSV format: X1, ..., Xd, score2_rel, T, I
#    let d_all = file.lines().next().unwrap().split(',').count() - 3;
#    let all_covs: Vec<String> = file
#        .lines()
#        .next()
#        .unwrap()
#        .split(',')
#        .take(d_all)
#        .map(std::borrow::ToOwned::to_owned)
#        .collect();
#    let cov_indices: Vec<usize> = (0..d_all)
#        .filter(|&i| covs.contains(&all_covs[i]))
#        .collect();
#    let d = cov_indices.len();
#    let mut X: Array2<f64> = Array2::zeros((n_avail, d));
#    let mut score2_rel: Array1<f64> = Array1::zeros(n_avail);
#    let mut T: Array1<f64> = Array1::zeros(n_avail);
#    let mut I: Array1<bool> = Array1::from_shape_fn(n_avail, |_| false);
#
#    // extract data
#    for (i, line) in file.lines().skip(1).take(n_avail).enumerate() {
#        let split: Vec<&str> = line.split(',').collect();
#        for j in 0..d {
#            let k = cov_indices[j];
#            X[[i, j]] = split[k].parse().unwrap();
#        }
#        score2_rel[i] = split[d_all].parse().unwrap();
#        T[i] = split[d_all + 1].parse().unwrap();
#        I[i] = split[d_all + 2].parse().unwrap();
#    }
#
#    // randomise order of rows
#    // the first n_test rows will be the test set
#    // so they should remain fixed
#
#    // get a random ordering of all the samples
#    let mut rng = StdRng::seed_from_u64(0);
#    let mut all_is: Vec<usize> = (0..n_avail).collect();
#    all_is.shuffle(&mut rng);
#
#    // fix the first n_test for predictable test set
#    let mut test_is: Vec<usize> = all_is[0..n_test].to_vec();
#
#    // shuffle the remaining samples by the value of rep
#    let mut rng = StdRng::seed_from_u64(rep as u64);
#    let mut other_is: Vec<usize> = all_is[n_test..n_avail].to_vec();
#    other_is.shuffle(&mut rng);
#
#    // join the indices and apply the premutation
#    test_is.append(&mut other_is);
#    let is = test_is;
#    //dbg!(&is[0..n_test]);
#    //dbg!(&is[n_test..]);
#    let X = X.select(Axis(0), &is);
#    let T = T.select(Axis(0), &is);
#    let I = I.select(Axis(0), &is);
#    let score2_rel = score2_rel.select(Axis(0), &is);
#
#    // split data into train, validate, test
#    let data_test = Data {
#        n: n_test,
#        d,
#        X: X.slice(s![..n_test, ..]).to_owned(),
#        T: T.slice(s![..n_test]).to_owned(),
#        I: I.slice(s![..n_test]).to_owned(),
#        f0_vals: None,
#    };
#
#    let data_train = Data {
#        n: n_train,
#        d,
#        X: X.slice(s![n_test..n_train + n_test, ..]).to_owned(),
#        T: T.slice(s![n_test..n_train + n_test]).to_owned(),
#        I: I.slice(s![n_test..n_train + n_test]).to_owned(),
#        f0_vals: None,
#    };
#
#    let data_valid = Data {
#        n: n_valid,
#        d,
#        X: X.slice(s![n_train + n_test..n, ..]).to_owned(),
#        T: T.slice(s![n_train + n_test..n]).to_owned(),
#        I: I.slice(s![n_train + n_test..n]).to_owned(),
#        f0_vals: None,
#    };
#
#    assert!(data_train.I.len() == n_train);
#    assert!(data_valid.I.len() == n_valid);
#    assert!(data_test.I.len() == n_test);
#
#    (data_train, data_valid, data_test, score2_rel)
#}
#
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

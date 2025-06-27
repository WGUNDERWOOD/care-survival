##![allow(non_snake_case)]
#use chrono::Local;
#use ndarray::{ArrayView1, Axis};
#use std::env::{args, current_dir};
#
#use rkhs_survival::common::*;
#use rkhs_survival::embedding::*;
#use rkhs_survival::external::*;
#use rkhs_survival::kernel::*;
#use rkhs_survival::selection::*;
#
#const EPS: f64 = 1e-8;
#
#fn main() {
#    // args
#    let args: Vec<String> = args().collect();
#    let model: usize = args[1].parse().unwrap();
#    let sex = Sex::from(&args[2]);
#    let rep: usize = args[3].parse().unwrap();
#    let method = Method::FeatureMap;
#    let _n_female = 162682;
#    let _n_male = 121333;
#    let n_female_over_3 = 54227;
#    let n_male_over_3 = 40444;
#
#    // set up parameters
#    let mut ns: Vec<usize> = match sex {
#        Sex::Female => vec![
#            2000,
#            3000,
#            4000,
#            5000,
#            6000,
#            7000,
#            8000,
#            9000,
#            10000,
#            12000,
#            14000,
#            16000,
#            18000,
#            20000,
#            25000,
#            30000,
#            35000,
#            40000,
#            45000,
#            50000,
#            n_female_over_3,
#        ],
#        Sex::Male => vec![
#            2000,
#            3000,
#            4000,
#            5000,
#            6000,
#            7000,
#            8000,
#            9000,
#            10000,
#            12000,
#            14000,
#            16000,
#            18000,
#            20000,
#            25000,
#            30000,
#            35000,
#            n_male_over_3,
#        ],
#    };
#    let n_test = match sex {
#        Sex::Female => n_female_over_3,
#        Sex::Male => n_male_over_3,
#    };
#    let dry_run = false;
#
#    // more set-up
#    let n_gammas = 50;
#    let gamma_min = 1e-8;
#    let gamma_max = 1e-2;
#    let covs = get_covs(model);
#    let simplex_resolution = 0.05;
#    let a = 1.0;
#    let p = 2;
#    let kernel = PolynomialKernel::new(a, p);
#    ns.sort();
#    ns.reverse();
#    let mut selection_results = Score2SelectionResults::new();
#
#    for n in ns {
#        println!(
#            "{}, model: {}, sex: {}, rep: {}, n: {}",
#            Local::now(),
#            model,
#            sex,
#            rep,
#            n
#        );
#
#        // get data
#        let (data_train, data_valid, data_test, score2_rel) =
#            get_score2_data(n, n, n_test, &covs, sex, dry_run, rep);
#
#        // check SCORE2 calculation
#        #[allow(clippy::type_complexity)]
#        let f_tilde: Box<dyn Fn(ArrayView1<f64>) -> f64> =
#            Box::new(move |x: ArrayView1<f64>| -> f64 {
#                get_score2_rel(x, sex)
#            });
#        let calculated_score2_rel = data_test.X.map_axis(Axis(1), &f_tilde);
#        if !dry_run {
#            for i in 0..n_test {
#                assert!((score2_rel[i] - calculated_score2_rel[i]).abs() < EPS);
#            }
#        }
#
#        // embedding
#        let embedding = Embedding::new(
#            &data_train,
#            &data_valid,
#            &data_test,
#            &kernel,
#            method,
#        );
#
#        // run model selection
#        let external = External::new(&embedding, &f_tilde);
#        let externals = vec![external];
#        let mut selection = ModelSelection::new(
#            &embedding,
#            gamma_min,
#            gamma_max,
#            n_gammas,
#            simplex_resolution,
#            &externals,
#        );
#        selection.select();
#
#        // extract values
#        selection_results.model = Some(model);
#        selection_results.sex = Some(sex);
#        selection_results.rep = Some(rep);
#        selection_results.ns.push(n);
#        selection_results.gamma_hats.push(selection.get_gamma_hat());
#        selection_results
#            .gamma_checks
#            .push(selection.get_gamma_check());
#        selection_results
#            .theta_checks
#            .push(selection.get_theta_check()[0]);
#        selection_results.ln_hats.push(selection.get_ln_hat());
#        selection_results.ln_checks.push(selection.get_ln_check());
#        selection_results.ln_tildes.push(selection.get_ln_tilde());
#        selection_results
#            .concordance_hats
#            .push(selection.get_concordance_hat());
#        selection_results
#            .concordance_checks
#            .push(selection.get_concordance_check());
#        selection_results
#            .concordance_tildes
#            .push(selection.get_concordance_tilde());
#    }
#
#    // save data
#    let file_name =
#        format!("analysis_score2_model_{model}_{sex}_rep_{rep}.csv");
#    let path = current_dir()
#        .unwrap()
#        .join("data")
#        .join(format!("{}", Local::now().format("%Y-%m-%d")))
#        .join("score2")
#        .join("analysis")
#        .join(file_name);
#    selection_results.write(&path);
#}
#
#fn get_covs(model: usize) -> Vec<String> {
#    let score2_covs = [
#        "age",
#        "hdl",
#        "sbp",
#        "tchol",
#        "smoking",
#        "age_hdl",
#        "age_sbp",
#        "age_tchol",
#        "age_smoking",
#    ];
#    let new_covs = match model {
#        1 => vec![],
#        2 => vec!["imd"],
#        3 => vec!["imd", "pgs000018", "pgs000039"],
#        _ => panic!(),
#    };
#    score2_covs
#        .iter()
#        .chain(new_covs.iter())
#        .map(|&s| s.into())
#        .collect()
#}

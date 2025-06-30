import sys
from datetime import datetime

from care_survival import kernels as care_kernels
from care_survival import score2 as care_score2


def main():
    # args
    model = int(sys.argv[1])
    sex = sys.argv[2]
    rep = int(sys.argv[3])
    method = "feature_map"
    n_female = 162682
    n_male = 121333
    n_female_over_3 = 54227
    n_male_over_3 = 40444
    dry_run = True

    # set up parameters
    if dry_run:
        ns = [10, 15, 20]
        n_test = 20
    else:
        ns = [
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
            12000,
            14000,
            16000,
            18000,
            20000,
            25000,
            30000,
            35000,
        ]
        if sex == "female":
            ns.append([40000, 45000, 50000, n_female_over_3])
            n_test = n_female_over_3
        elif sex == "male":
            ns.append([n_male_over_3])
            n_test = n_male_over_3

    # more set-up
    n_gammas = 50
    gamma_min = 1e-8
    gamma_max = 1e-2
    covs = get_covs(model)
    simplex_resolution = 0.05
    a = 1
    p = 2
    kernel = care_kernels.PolynomialKernel(a, p)
    ns.sort(reverse=True)
    # mut selection_results = Score2SelectionResults::new();

    for n in ns:
        now = datetime.now().strftime("%H:%M:%S.%f")
        print(f"{now}, model = {model}, sex = {sex}, rep = {rep}, n = {n}", flush=True)

        # get data
        (data_train, data_valid, data_test, score2_rel) = care_score2.get_score2_data(
            n, n, n_test, covs, sex, dry_run, rep
        )


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
# }


def get_covs(model):
    score2_covs = [
        "age",
        "hdl",
        "sbp",
        "tchol",
        "smoking",
        "age_hdl",
        "age_sbp",
        "age_tchol",
        "age_smoking",
    ]
    if model == 1:
        score2_covs.append("imd")
    elif model == 2:
        score2_covs.append(["imd", "pgs000018", "pgs000039"])

    return score2_covs


if __name__ == "__main__":
    main()

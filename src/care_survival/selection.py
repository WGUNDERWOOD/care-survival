#use ndarray::Array1;
#use std::fs::{create_dir_all, File};
#use std::io::Write;
#use std::path::Path;
#
#use crate::common::Sex;
#use crate::embedding::Embedding;
#use crate::estimator::{Estimator, Score};
#use crate::external::External;
#use crate::simplex::{get_simplex, SimplexSelection};
#use crate::validation::{cmp, get_gammas, Best};
#
#const EPS: f64 = 2.0 * f64::EPSILON;
#
##[derive(Debug)]
#pub struct ModelSelection<'a> {
#    pub embedding: &'a Embedding,
#    pub gamma_min: f64,
#    pub gamma_max: f64,
#    pub n_gammas: usize,
#    pub gammas: Array1<f64>,
#    pub simplex_resolution: f64,
#    pub thetas: Vec<Array1<f64>>,
#    pub externals: &'a [External<'a>],
#    pub simplex_selections: Vec<SimplexSelection<'a>>,
#    pub best: Best,
#}
#
#impl ModelSelection<'_> {
#    #[must_use]
#    pub fn new<'a>(
#        embedding: &'a Embedding,
#        gamma_min: f64,
#        gamma_max: f64,
#        n_gammas: usize,
#        simplex_resolution: f64,
#        externals: &'a [External<'a>],
#    ) -> ModelSelection<'a> {
#        let simplex_dimension = externals.len();
#        ModelSelection {
#            embedding,
#            gamma_min,
#            gamma_max,
#            n_gammas,
#            gammas: get_gammas(gamma_min, gamma_max, n_gammas),
#            simplex_resolution,
#            thetas: get_simplex(simplex_dimension, simplex_resolution),
#            externals,
#            simplex_selections: Vec::with_capacity(n_gammas),
#            best: Best::new(),
#        }
#    }
#
#    pub fn select(&mut self) {
#        let mut beta_hat = self.embedding.train.get_default_beta();
#        let mut inv_hessian_hat =
#            self.embedding.train.get_default_inv_hessian();
#
#        for i in 0..self.n_gammas {
#            let gamma = self.gammas[i];
#            //println!("gamma: {}", gamma);
#            let mut estimator = Estimator::new(self.embedding, gamma);
#            estimator.optimise(&beta_hat, &inv_hessian_hat);
#            let simplex_dimension = self.externals.len();
#            inv_hessian_hat = estimator.inv_hessian_hat.clone();
#            beta_hat = estimator.beta_hat.clone();
#            let mut simplex_selection = SimplexSelection::new(
#                self.embedding,
#                estimator,
#                self.externals,
#                simplex_dimension,
#                self.simplex_resolution,
#            );
#            simplex_selection.select();
#            self.simplex_selections.push(simplex_selection);
#        }
#
#        // compute best values
#        self.best.ln.train = self.get_best_by(|s| s.best.ln.train.map(|b| b.1));
#        self.best.ln.valid = self.get_best_by(|s| s.best.ln.valid.map(|b| b.1));
#        self.best.ln.test = self.get_best_by(|s| s.best.ln.test.map(|b| b.1));
#
#        self.best.rmse.train =
#            self.get_best_by(|s| s.best.rmse.train.map(|b| b.1));
#        self.best.rmse.valid =
#            self.get_best_by(|s| s.best.rmse.valid.map(|b| b.1));
#        self.best.rmse.test =
#            self.get_best_by(|s| s.best.rmse.test.map(|b| b.1));
#    }
#
#    pub fn get_best_by(
#        &self,
#        by: fn(&SimplexSelection) -> Option<f64>,
#    ) -> Option<(usize, f64)> {
#        let best = self
#            .simplex_selections
#            .iter()
#            .map(by)
#            .min_by(|&x, &y| cmp(x, y))
#            .unwrap();
#        let index = best.map(|_| {
#            (0..self.n_gammas)
#                .find(|&i| by(&self.simplex_selections[i]) == best)
#                .unwrap()
#        });
#        index.map(|i| (i, best.unwrap()))
#    }
#
#    #[must_use]
#    pub fn get_gamma_hat(&self) -> f64 {
#        let kernel_scores: Vec<Score> = self
#            .simplex_selections
#            .iter()
#            .map(|s| s.combinations[0].score)
#            .collect();
#        let best_kernel_ln_valid = kernel_scores
#            .iter()
#            .map(|s| s.ln.valid.unwrap())
#            .min_by(f64::total_cmp)
#            .unwrap();
#        let best_kernel_ln_valid_index = (0..self.n_gammas)
#            .find(|&i| {
#                (kernel_scores[i].ln.valid.unwrap() - best_kernel_ln_valid)
#                    .abs()
#                    < EPS
#            })
#            .unwrap();
#        self.simplex_selections[best_kernel_ln_valid_index].combinations[0]
#            .estimator
#            .gamma
#    }
#
#    #[must_use]
#    pub fn get_gamma_star(&self) -> f64 {
#        let kernel_scores: Vec<Score> = self
#            .simplex_selections
#            .iter()
#            .map(|s| s.combinations[0].score)
#            .collect();
#        let best_kernel_rmse_test = kernel_scores
#            .iter()
#            .map(|s| s.rmse.test.unwrap())
#            .min_by(f64::total_cmp)
#            .unwrap();
#        let best_kernel_rmse_test_index = (0..self.n_gammas)
#            .find(|&i| {
#                (kernel_scores[i].rmse.test.unwrap() - best_kernel_rmse_test)
#                    .abs()
#                    < EPS
#            })
#            .unwrap();
#        self.simplex_selections[best_kernel_rmse_test_index].combinations[0]
#            .estimator
#            .gamma
#    }
#
#    #[must_use]
#    pub fn get_gamma_dagger(&self) -> f64 {
#        let best_rmse_test_index = self.best.rmse.test.unwrap().0;
#        let best_rmse_test_sub_index = self.simplex_selections
#            [best_rmse_test_index]
#            .best
#            .rmse
#            .test
#            .unwrap()
#            .0;
#        self.simplex_selections[best_rmse_test_index].combinations
#            [best_rmse_test_sub_index]
#            .estimator
#            .gamma
#    }
#
#    #[must_use]
#    pub fn get_gamma_check(&self) -> f64 {
#        let best_ln_valid_index = self.best.ln.valid.unwrap().0;
#        let best_ln_valid_sub_index = self.simplex_selections
#            [best_ln_valid_index]
#            .best
#            .ln
#            .valid
#            .unwrap()
#            .0;
#        self.simplex_selections[best_ln_valid_index].combinations
#            [best_ln_valid_sub_index]
#            .estimator
#            .gamma
#    }
#
#    #[must_use]
#    pub fn get_theta_dagger(&self) -> Array1<f64> {
#        let best_rmse_test_index = self.best.rmse.test.unwrap().0;
#        let best_rmse_test_sub_index = self.simplex_selections
#            [best_rmse_test_index]
#            .best
#            .rmse
#            .test
#            .unwrap()
#            .0;
#        self.simplex_selections[best_rmse_test_index].combinations
#            [best_rmse_test_sub_index]
#            .theta
#            .clone()
#    }
#
#    #[must_use]
#    pub fn get_theta_check(&self) -> Array1<f64> {
#        let best_ln_valid_index = self.best.ln.valid.unwrap().0;
#        let best_ln_valid_sub_index = self.simplex_selections
#            [best_ln_valid_index]
#            .best
#            .ln
#            .valid
#            .unwrap()
#            .0;
#        self.simplex_selections[best_ln_valid_index].combinations
#            [best_ln_valid_sub_index]
#            .theta
#            .clone()
#    }
#
#    #[must_use]
#    pub fn get_rmse_star(&self) -> f64 {
#        let kernel_scores: Vec<Score> = self
#            .simplex_selections
#            .iter()
#            .map(|s| s.combinations[0].score)
#            .collect();
#        kernel_scores
#            .iter()
#            .map(|s| s.rmse.test.unwrap())
#            .min_by(f64::total_cmp)
#            .unwrap()
#    }
#
#    #[must_use]
#    pub fn get_rmse_hat(&self) -> f64 {
#        let kernel_scores: Vec<Score> = self
#            .simplex_selections
#            .iter()
#            .map(|s| s.combinations[0].score)
#            .collect();
#        let best_kernel_ln_valid = kernel_scores
#            .iter()
#            .map(|s| s.ln.valid.unwrap())
#            .min_by(f64::total_cmp)
#            .unwrap();
#        let best_kernel_ln_valid_index = (0..self.n_gammas)
#            .find(|&i| {
#                (kernel_scores[i].ln.valid.unwrap() - best_kernel_ln_valid)
#                    .abs()
#                    < EPS
#            })
#            .unwrap();
#        self.simplex_selections[best_kernel_ln_valid_index].combinations[0]
#            .estimator
#            .score
#            .rmse
#            .test
#            .unwrap()
#    }
#
#    #[must_use]
#    pub fn get_rmse_dagger(&self) -> f64 {
#        self.best.rmse.test.unwrap().1
#    }
#
#    #[must_use]
#    pub fn get_rmse_check(&self) -> f64 {
#        let best_ln_valid_index = self.best.ln.valid.unwrap().0;
#        let best_ln_valid_sub_index = self.simplex_selections
#            [best_ln_valid_index]
#            .best
#            .ln
#            .valid
#            .unwrap()
#            .0;
#        self.simplex_selections[best_ln_valid_index].combinations
#            [best_ln_valid_sub_index]
#            .score
#            .rmse
#            .test
#            .unwrap()
#    }
#
#    #[must_use]
#    pub fn get_rmse_tilde(&self) -> f64 {
#        let tilde_score: Score = self
#            .simplex_selections
#            .iter()
#            .map(|s| s.combinations.last().unwrap().score)
#            .next()
#            .unwrap();
#        tilde_score.rmse.test.unwrap()
#    }
#
#    #[must_use]
#    pub fn get_ln_hat(&self) -> f64 {
#        let kernel_scores: Vec<Score> = self
#            .simplex_selections
#            .iter()
#            .map(|s| s.combinations[0].score)
#            .collect();
#        let best_kernel_ln_valid = kernel_scores
#            .iter()
#            .map(|s| s.ln.valid.unwrap())
#            .min_by(f64::total_cmp)
#            .unwrap();
#        let best_kernel_ln_valid_index = (0..self.n_gammas)
#            .find(|&i| {
#                (kernel_scores[i].ln.valid.unwrap() - best_kernel_ln_valid)
#                    .abs()
#                    < EPS
#            })
#            .unwrap();
#        self.simplex_selections[best_kernel_ln_valid_index].combinations[0]
#            .estimator
#            .score
#            .ln
#            .test
#            .unwrap()
#    }
#
#    #[must_use]
#    pub fn get_ln_check(&self) -> f64 {
#        let best_ln_valid_index = self.best.ln.valid.unwrap().0;
#        let best_ln_valid_sub_index = self.simplex_selections
#            [best_ln_valid_index]
#            .best
#            .ln
#            .valid
#            .unwrap()
#            .0;
#        self.simplex_selections[best_ln_valid_index].combinations
#            [best_ln_valid_sub_index]
#            .score
#            .ln
#            .test
#            .unwrap()
#    }
#
#    #[must_use]
#    pub fn get_ln_tilde(&self) -> f64 {
#        let tilde_score: Score = self
#            .simplex_selections
#            .iter()
#            .map(|s| s.combinations.last().unwrap().score)
#            .next()
#            .unwrap();
#        tilde_score.ln.test.unwrap()
#    }
#
#    #[must_use]
#    pub fn get_concordance_hat(&self) -> f64 {
#        let kernel_scores: Vec<Score> = self
#            .simplex_selections
#            .iter()
#            .map(|s| s.combinations[0].score)
#            .collect();
#        let best_kernel_ln_valid = kernel_scores
#            .iter()
#            .map(|s| s.ln.valid.unwrap())
#            .min_by(f64::total_cmp)
#            .unwrap();
#        let best_kernel_ln_valid_index = (0..self.n_gammas)
#            .find(|&i| {
#                (kernel_scores[i].ln.valid.unwrap() - best_kernel_ln_valid)
#                    .abs()
#                    < EPS
#            })
#            .unwrap();
#        self.simplex_selections[best_kernel_ln_valid_index].combinations[0]
#            .estimator
#            .score
#            .concordance
#            .test
#            .unwrap()
#    }
#
#    #[must_use]
#    pub fn get_concordance_check(&self) -> f64 {
#        let best_ln_valid_index = self.best.ln.valid.unwrap().0;
#        let best_ln_valid_sub_index = self.simplex_selections
#            [best_ln_valid_index]
#            .best
#            .ln
#            .valid
#            .unwrap()
#            .0;
#        self.simplex_selections[best_ln_valid_index].combinations
#            [best_ln_valid_sub_index]
#            .score
#            .concordance
#            .test
#            .unwrap()
#    }
#
#    #[must_use]
#    pub fn get_concordance_tilde(&self) -> f64 {
#        let tilde_score: Score = self
#            .simplex_selections
#            .iter()
#            .map(|s| s.combinations.last().unwrap().score)
#            .next()
#            .unwrap();
#        tilde_score.concordance.test.unwrap()
#    }
#}
#
##[derive(Debug)]
#pub struct SelectionResults {
#    pub rep: Option<usize>,
#    pub ns: Vec<usize>,
#    pub gamma_stars: Vec<f64>,
#    pub gamma_hats: Vec<f64>,
#    pub gamma_daggers: Vec<f64>,
#    pub gamma_checks: Vec<f64>,
#    pub theta_daggers: Vec<f64>,
#    pub theta_checks: Vec<f64>,
#    pub rmse_stars: Vec<f64>,
#    pub rmse_hats: Vec<f64>,
#    pub rmse_daggers: Vec<f64>,
#    pub rmse_checks: Vec<f64>,
#    pub rmse_tildes: Vec<f64>,
#}
#
#impl SelectionResults {
#    #[must_use]
#    #[allow(clippy::new_without_default)]
#    pub fn new() -> Self {
#        Self {
#            rep: None,
#            ns: vec![],
#            gamma_stars: vec![],
#            gamma_hats: vec![],
#            gamma_daggers: vec![],
#            gamma_checks: vec![],
#            theta_daggers: vec![],
#            theta_checks: vec![],
#            rmse_stars: vec![],
#            rmse_hats: vec![],
#            rmse_daggers: vec![],
#            rmse_checks: vec![],
#            rmse_tildes: vec![],
#        }
#    }
#
#    pub fn write(&self, path: &Path) {
#        let mut s: String =
#            "n,rep,gamma_star,gamma_hat,gamma_dagger,gamma_check,".into();
#        s.push_str("theta_dagger,theta_check,rmse_star,rmse_hat,");
#        s.push_str("rmse_dagger,rmse_check,rmse_tilde\n");
#        let k = self.ns.len();
#        for i in 0..k {
#            s.push_str(&self.ns[i].to_string());
#            s.push(',');
#            s.push_str(&self.rep.unwrap().to_string());
#            s.push(',');
#            s.push_str(&self.gamma_stars[i].to_string());
#            s.push(',');
#            s.push_str(&self.gamma_hats[i].to_string());
#            s.push(',');
#            s.push_str(&self.gamma_daggers[i].to_string());
#            s.push(',');
#            s.push_str(&self.gamma_checks[i].to_string());
#            s.push(',');
#            s.push_str(&self.theta_daggers[i].to_string());
#            s.push(',');
#            s.push_str(&self.theta_checks[i].to_string());
#            s.push(',');
#            s.push_str(&self.rmse_stars[i].to_string());
#            s.push(',');
#            s.push_str(&self.rmse_hats[i].to_string());
#            s.push(',');
#            s.push_str(&self.rmse_daggers[i].to_string());
#            s.push(',');
#            s.push_str(&self.rmse_checks[i].to_string());
#            s.push(',');
#            s.push_str(&self.rmse_tildes[i].to_string());
#            s.push('\n');
#        }
#        create_dir_all(path.parent().unwrap()).unwrap();
#        let mut file = File::create(path).unwrap();
#        file.write_all(s.as_bytes()).unwrap();
#    }
#}
#
##[derive(Debug)]
#pub struct Score2SelectionResults {
#    pub model: Option<usize>,
#    pub sex: Option<Sex>,
#    pub rep: Option<usize>,
#    pub ns: Vec<usize>,
#    pub gamma_hats: Vec<f64>,
#    pub gamma_checks: Vec<f64>,
#    pub theta_checks: Vec<f64>,
#    pub ln_hats: Vec<f64>,
#    pub ln_checks: Vec<f64>,
#    pub ln_tildes: Vec<f64>,
#    pub concordance_hats: Vec<f64>,
#    pub concordance_checks: Vec<f64>,
#    pub concordance_tildes: Vec<f64>,
#}
#
#impl Score2SelectionResults {
#    #[must_use]
#    #[allow(clippy::new_without_default)]
#    pub fn new() -> Self {
#        Self {
#            model: None,
#            sex: None,
#            rep: None,
#            ns: vec![],
#            gamma_hats: vec![],
#            gamma_checks: vec![],
#            theta_checks: vec![],
#            ln_hats: vec![],
#            ln_checks: vec![],
#            ln_tildes: vec![],
#            concordance_hats: vec![],
#            concordance_checks: vec![],
#            concordance_tildes: vec![],
#        }
#    }
#
#    pub fn write(&self, path: &Path) {
#        let mut s: String =
#            "n,model,sex,rep,gamma_hat,gamma_check,theta_check,".into();
#        s.push_str("ln_hat,ln_check,ln_tilde,");
#        s.push_str("concordance_hat,concordance_check,concordance_tilde\n");
#        let k = self.ns.len();
#        for i in 0..k {
#            s.push_str(&self.ns[i].to_string());
#            s.push(',');
#            s.push_str(&self.model.unwrap().to_string());
#            s.push(',');
#            s.push_str(&self.sex.unwrap().to_string());
#            s.push(',');
#            s.push_str(&self.rep.unwrap().to_string());
#            s.push(',');
#            s.push_str(&self.gamma_hats[i].to_string());
#            s.push(',');
#            s.push_str(&self.gamma_checks[i].to_string());
#            s.push(',');
#            s.push_str(&self.theta_checks[i].to_string());
#            s.push(',');
#            s.push_str(&self.ln_hats[i].to_string());
#            s.push(',');
#            s.push_str(&self.ln_checks[i].to_string());
#            s.push(',');
#            s.push_str(&self.ln_tildes[i].to_string());
#            s.push(',');
#            s.push_str(&self.concordance_hats[i].to_string());
#            s.push(',');
#            s.push_str(&self.concordance_checks[i].to_string());
#            s.push(',');
#            s.push_str(&self.concordance_tildes[i].to_string());
#            s.push('\n');
#        }
#        create_dir_all(path.parent().unwrap()).unwrap();
#        let mut file = File::create(path).unwrap();
#        file.write_all(s.as_bytes()).unwrap();
#    }
#}

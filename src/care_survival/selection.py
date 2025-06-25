import numpy as np

from care_survival import simplex as care_simplex
from care_survival import estimator as care_estimator
from care_survival import metrics as care_metrics


class CARE:
    def __init__(self, embedding, gamma_min, gamma_max, n_gammas, simplex_resolution):
        self.embedding = embedding
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.n_gammas = n_gammas
        self.gammas = get_gammas(gamma_min, gamma_max, n_gammas)
        self.simplex_resolution = simplex_resolution
        self.simplex_dimension = np.shape(embedding.data["train"].f_tilde)[1]
        self.thetas = care_simplex.get_simplex(
            self.simplex_dimension, simplex_resolution
        )

    def fit(self):
        beta_hat = self.embedding.data["train"].get_default_beta()
        inv_hessian_hat = self.embedding.data["train"].get_default_inv_hessian()
        self.simplex_selections = [None for _ in range(self.n_gammas)]

        for i in range(self.n_gammas):
            gamma = self.gammas[i]
            estimator = care_estimator.Estimator(self.embedding, gamma)
            estimator.optimise(beta_hat, inv_hessian_hat)
            inv_hessian_hat = estimator.inv_hessian_hat
            beta_hat = estimator.beta_hat
            simplex_selection = care_simplex.SimplexSelection(
                estimator, self.simplex_resolution
            )
            simplex_selection.fit()
            self.simplex_selections[i] = simplex_selection

        self.best = {}
        for metric in care_metrics.get_metrics():
            self.best[metric] = {}
            for split in care_metrics.get_splits():
                self.best[metric][split] = self.best_by(metric, split)

    def best_by(self, metric, split):
        scores = [
            c
            for s in self.simplex_selections
            for c in s.combinations
            if c.score[metric][split] is not None
        ]

        def key(c):
            return c.score[metric][split]

        return min(scores, key=key)


def get_gammas(gamma_min, gamma_max, n_gammas):
    ratio = (gamma_max / gamma_min) ** (1 / (n_gammas - 1))
    return [gamma_min * ratio**i for i in reversed(range(n_gammas))]


#
##[derive(Debug)]
# pub struct SelectionResults {
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
# }
#
# impl SelectionResults {
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
# }
#

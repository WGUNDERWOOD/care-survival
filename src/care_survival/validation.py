import numpy as np

from care_survival import estimator as care_estimator


class Validation:
    def __init__(self, embedding, gamma_min, gamma_max, n_gammas):
        self.embedding = embedding
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.n_gammas = n_gammas
        self.gammas = get_gammas(gamma_min, gamma_max, n_gammas)
        self.estimators = [None for _ in range(n_gammas)]

    def validate(self):
        beta_hat = self.embedding.train.get_default_beta()
        inv_hessian_hat = self.embedding.train.get_default_inv_hessian()
        best_split = BestSplit(None, None, None)
        self.best = Best(best_split, best_split, best_split, best_split)

        for i in range(self.n_gammas):
            gamma = self.gammas[i]
            estimator = care_estimator.Estimator(self.embedding, gamma)
            estimator.optimise(beta_hat, inv_hessian_hat)
            beta_hat = estimator.beta_hat
            inv_hessian_hat = estimator.inv_hessian_hat
            self.estimators[i] = estimator

        self.best.ln.train = self.get_best_by(lambda e: e.score.ln.train)
        self.best.ln.valid = self.get_best_by(lambda e: e.score.ln.valid)
        self.best.ln.test = self.get_best_by(lambda e: e.score.ln.test)

        self.best.lng.train = self.get_best_by(lambda e: e.score.lng.train)
        self.best.lng.valid = self.get_best_by(lambda e: e.score.lng.valid)
        self.best.lng.test = self.get_best_by(lambda e: e.score.lng.test)

        self.best.rmse.train = self.get_best_by(lambda e: e.score.rmse.train)
        self.best.rmse.valid = self.get_best_by(lambda e: e.score.rmse.valid)
        self.best.rmse.test = self.get_best_by(lambda e: e.score.rmse.test)

    def get_best_by(self, by):
        best = np.min([by(e) for e in self.estimators if by(e) is not None])
        index = [i for i in range(self.n_gammas) if by(self.estimators[i]) == best][0]
        return (index, float(best))


def get_gammas(gamma_min, gamma_max, n_gammas):
    ratio = (gamma_max / gamma_min) ** (1 / (n_gammas - 1))
    return [gamma_min * ratio**i for i in reversed(range(n_gammas))]


class BestSplit:
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test


class Best:
    def __init__(self, ln, lng, rmse, concordance):
        self.ln = ln
        self.lng = lng
        self.rmse = rmse
        self.concordance = concordance

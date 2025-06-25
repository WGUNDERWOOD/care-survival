class Combination:
    def __init__(self, estimator, theta):
        self.estimator = estimator
        self.theta = theta
        self.f_check_train = get_f_check_split(estimator, theta, "train")
        self.f_check_valid = get_f_check_split(estimator, theta, "valid")
        self.f_check_test = get_f_check_split(estimator, theta, "test")
        self.score = self.get_score()

    def get_ln_split(self, split):
        if split == "train":
            embedding_data = self.estimator.embedding.train
            f = self.f_check_train
        elif split == "valid":
            embedding_data = self.estimator.embedding.valid
            f = self.f_check_valid
        elif split == "test":
            embedding_data = self.estimator.embedding.test
            f = self.f_check_test

        n = embedding_data.n
        f_max = np.max(f)
        f_expt = expt(f, f_max)
        sn = get_sn(embedding_data, f_expt)
        N = embedding_data.N
        ln_cent = embedding_data.ln_cent
        return np.sum((np.log(sn) + f_max - f) * N) / n - ln_cent

    def get_ln(self):
        ln_train = self.get_ln_split("train")
        ln_valid = self.get_ln_split("valid")
        ln_test = self.get_ln_split("test")
        return ScoreSplit(ln_train, ln_valid, ln_test)

    def get_rmse_split(self, split):
        if split == "train":
            f = self.f_check_train
            return get_rmse(f, self.estimator.embedding.train.f_0)
        elif split == "valid":
            f = self.f_check_valid
            return get_rmse(f, self.estimator.embedding.valid.f_0)
        elif split == "test":
            f = self.f_check_test
            return get_rmse(f, self.estimator.embedding.test.f_0)

    def get_rmse(self):
        rmse_train = self.get_rmse_split("train")
        rmse_valid = self.get_rmse_split("valid")
        rmse_test = self.get_rmse_split("test")
        return ScoreSplit(rmse_train, rmse_valid, rmse_test)

    def get_concordance(self):
        concordance_train = get_concordance(
            self.f_check_train, self.estimator.embedding.train
        )
        concordance_valid = get_concordance(
            self.f_check_valid, self.estimator.embedding.valid
        )
        concordance_test = get_concordance(
            self.f_check_test, self.estimator.embedding.test
        )
        return ScoreSplit(concordance_train, concordance_valid, concordance_test)

    def get_score(self):
        ln = self.get_ln()
        lng = ScoreSplit(None, None, None)
        rmse = self.get_rmse()
        concordance = self.get_concordance()
        return Score(ln, lng, rmse, concordance)


class SimplexSelection:
    def __init__(estimator, simplex_dimension, simplex_resolution):
        self.estimator = estimator
        self.simplex_dimension = simplex_dimension
        self.simplex_resolution = simplex_resolution
        self.thetas = get_simplex(simplex_dimension, simplex_resolution)
        self.n_thetas = len(thetas)
        self.combinations = [None for _ in range(n_thetas)]

    def select(self):
        best_split = BestSplit(None, None, None)
        self.best = Best(best_split, best_split, best_split, best_split)

        for i in range(self.n_thetas):
            theta = self.thetas[i]
            self.combinations[i] = Combination(self.estimator, theta)

        self.best.ln.train = self.get_best_by(lambda c: c.score.ln.train)
        self.best.ln.valid = self.get_best_by(lambda c: c.score.ln.valid)
        self.best.ln.test = self.get_best_by(lambda c: c.score.ln.test)

        self.best.rmse.train = self.get_best_by(lambda c: c.score.rmse.train)
        self.best.rmse.valid = self.get_best_by(lambda c: c.score.rmse.valid)
        self.best.rmse.test = self.get_best_by(lambda c: c.score.rmse.test)

    def get_best_by(self, by):
        best = np.min([by(c) for c in self.combinations if by(e) is not None])
        index = [i for i in range(self.n_gammas) if by(self.combinations[i]) == best][0]
        return (index, float(best))


def get_simplex(simplex_dimension, simplex_resolution):
    n_values = np.ceil(1 / simplex_resolution)
    values = [i * simplex_resolution for i in range(n_values)]
    values.append(1)
    values = list(set(values))
    values_rep = [values for _ in range(simplex_dimension)]
    simplex = list(itertools.product(*values_rep))
    simplex = [np.array(s) for s in simplex if np.sum(s) <= 1]


def get_f_check_split(estimator, theta, split):
    simplex_dimension = len(theta)
    theta_0 = 1.0 - np.sum(theta)

    if split == "train":
        f_check = theta_0 * estimator.f_hat_train
        for i in range(simplex_dimension):
            f_check = f_check + theta[i] * estimator.embedding.train.f_tilde[:, i]
        return f_check

    elif split == "valid":
        f_check = theta_0 * estimator.f_hat_valid
        for i in range(simplex_dimension):
            f_check = f_check + theta[i] * estimator.embedding.valid.f_tilde[:, i]
        return f_check

    elif split == "test":
        f_check = theta_0 * estimator.f_hat_test
        for i in range(simplex_dimension):
            f_check = f_check + theta[i] * estimator.embedding.test.f_tilde[:, i]
        return f_check

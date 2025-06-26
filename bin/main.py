import numpy as np
import pprint

from care_survival import data as care_data
from care_survival import kernels as care_kernels
from care_survival import embedding as care_embedding
from care_survival import aggregation as care_aggregation


def get_random_data():
    n = 5
    # n = 1000
    d = 5
    m = 1
    X = np.random.random((n, d))
    T = np.random.random(n)
    I = np.random.randint(0, 2, size=n)
    f_tilde = np.random.random((n, m))
    f_0 = np.random.random(n)
    return care_data.Data(X, T, I, f_tilde, f_0)


def main():
    # data
    data_train = get_random_data()
    data_valid = get_random_data()
    data_test = get_random_data()

    # kernel
    a = 1
    p = 2
    # sigma = 0.5
    kernel = care_kernels.PolynomialKernel(a, p)
    # kernel = care_kernels.ShiftedGaussianKernel(a, sigma)
    # kernel = care_kernels.ShiftedFirstOrderSobolevKernel(a)

    # optimisation method
    # method = "kernel"
    method = "feature_map"

    # kernel/feature embedding
    embedding = care_embedding.Embedding(
        data_train, data_valid, data_test, kernel, method
    )

    # fit CARE
    gamma_min = 1e-3
    gamma_max = 1e0
    n_gammas = 2
    simplex_resolution = 0.5
    care = care_aggregation.CARE(
        embedding, gamma_min, gamma_max, n_gammas, simplex_resolution
    )
    care.fit()
    # best = care.best["aggregated"]["ln"]["test"]
    # print(best.theta)
    # print(best.score["ln"]["test"])
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(care.summary)


if __name__ == "__main__":
    main()

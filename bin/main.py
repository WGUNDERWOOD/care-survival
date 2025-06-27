import sys
import numpy as np
# import pprint

# from care_survival import data as care_data
from care_survival import distributions as care_distributions
from care_survival import kernels as care_kernels
from care_survival import embedding as care_embedding

# from care_survival import aggregation as care_aggregation
from care_survival import kernel_estimator as care_kernel_estimator


def main():
    dgp = int(sys.argv[1])

    # data
    n = 5
    n_train = n
    n_valid = n
    n_test = n

    distribution = care_distributions.get_distribution(dgp)
    np.random.seed(4)
    data_train = distribution.sample(n_train)
    data_valid = distribution.sample(n_valid)
    data_test = distribution.sample(n_test)

    # kernel
    a = 1
    # p = 2
    # sigma = 0.5
    # kernel = care_kernels.PolynomialKernel(a, p)
    # kernel = care_kernels.ShiftedGaussianKernel(a, sigma)
    kernel = care_kernels.ShiftedFirstOrderSobolevKernel(a)

    # optimisation method
    method = "kernel"
    # method = "feature_map"

    # kernel/feature embedding
    embedding = care_embedding.Embedding(
        data_train, data_valid, data_test, kernel, method
    )
    print(embedding.data["train"].K)
    print(embedding.data["train"].K_bar)
    print(embedding.data["train"].K_tilde)
    print(embedding.data["train"].K_hat)

    # fit a kernel estimator
    gamma = 1
    kernel_estimator = care_kernel_estimator.KernelEstimator(embedding, gamma)
    beta_init = embedding.data["train"].get_default_beta()
    inv_hessian_init = embedding.data["train"].get_default_inv_hessian()
    kernel_estimator.fit(beta_init, inv_hessian_init)
    # beta = np.random.random(n)
    # beta = np.zeros(n)
    # print(kernel_estimator.get_f(beta, "train"))
    # print(get_sn(beta, "train"))

    # fit CARE
    # gamma_min = 1e-3
    # gamma_max = 1e0
    # n_gammas = 2
    # simplex_resolution = 0.5
    # care = care_aggregation.CARE(
    # embedding, gamma_min, gamma_max, n_gammas, simplex_resolution
    # )
    # care.fit()
    # best = care.best["aggregated"]["ln"]["test"]
    # print(best.theta)
    # print(best.score["ln"]["test"])
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(care.summary)


if __name__ == "__main__":
    main()

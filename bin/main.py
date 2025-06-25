import numpy as np
from care_survival import data as care_data
from care_survival import kernel as care_kernel
from care_survival import embedding as care_embedding
from care_survival import estimator as care_estimator


def get_random_data():
    n = 6
    # n = 100
    d = 3
    m = 2
    X = np.random.random((n, d))
    T = np.random.random(n)
    I = np.random.randint(0, 2, size=n)
    f_tilde = np.random.random((n, m))
    f_0 = np.random.random(n)
    return care_data.Data(X, T, I, f_tilde, f_0)


def main():
    data_train = get_random_data()
    data_valid = get_random_data()
    data_test = get_random_data()
    a = 1
    p = 2
    # sigma = 0.5
    kernel = care_kernel.PolynomialKernel(a, p)
    # kernel = care_kernel.ShiftedGaussianKernel(a, sigma)
    # kernel = care_kernel.ShiftedFirstOrderSobolevKernel(a)

    # print(data.X)
    # print(kernel.k(data.X, data.X))
    # print(kernel.norm_one())
    # print(kernel.phi(data.X).shape)
    # method = "kernel"
    method = "feature_map"
    embedding = care_embedding.Embedding(
        data_train, data_valid, data_test, kernel, method
    )
    gamma = 0.5
    estimator = care_estimator.Estimator(embedding, gamma)

    if method == "kernel":
        beta = np.random.random(data_train.n)
    elif method == "feature_map":
        beta = np.random.random(embedding.train.feature_dim)

    # print(estimator.beta_hat)
    # print(estimator.get_f(estimator.beta_hat, "train"))
    print(estimator.get_dlng_split(estimator.beta_hat, "train"))
    # print(embed.train.K)
    # print(embed.train.Phi)
    # print(data.f_0)


if __name__ == "__main__":
    main()

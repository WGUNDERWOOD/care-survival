import sys
import numpy as np

from care_survival import kernels as care_kernels
from care_survival import embedding as care_embedding
from care_survival import aggregation as care_aggregation
from care_survival import distributions as care_distributions


def main():
    dgp = int(sys.argv[1])

    # data
    n = 200
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
    kernel = care_kernels.ShiftedFirstOrderSobolevKernel(a)
    method = "kernel"
    embedding = care_embedding.Embedding(
        data_train, data_valid, data_test, kernel, method
    )

    # run cross-validation
    n_gammas = 50
    gamma_min = 1e-5
    gamma_max = 1e1
    simplex_resolution = 1
    care = care_aggregation.CARE(
        embedding, gamma_min, gamma_max, n_gammas, simplex_resolution
    )
    care.fit()
    print(care.summary)
    # best = care.best["kernel"]["ln"]["valid"]

    # TODO write results
    # path = current_dir()
    # .unwrap()
    # .join("data")
    # .join(format!("{}", Local::now().format("%Y-%m-%d")))
    # .join("simulation")
    # .join(format!("illustration_validation_dgp_{dgp}.csv"))
    # validation.write(&path)

    # best estimator
    # best_ln_valid_index = validation.best.ln.valid.unwrap().0
    # estimator_hat = &mut validation.estimators[best_ln_valid_index]
    # estimator_hat.get_breslow()
    # path = current_dir()
    # .unwrap()
    # .join("data")
    # .join(format!("{}", Local::now().format("%Y-%m-%d")))
    # .join("simulation")
    # .join(format!("illustration_estimator_dgp_{dgp}.csv"))
    # estimator_hat.write(&path)


if __name__ == "__main__":
    main()

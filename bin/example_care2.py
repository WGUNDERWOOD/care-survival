import numpy as np
import care_survival

# generate some sample data
n = 60
d = 2
X = np.random.random((n, d))
T = X.sum(axis=1) + 1 + np.random.random(n)
I = (np.random.random(n) > 0.8) * 1
f = (-X.sum(axis=1) / 2 + np.random.random(n)).reshape(-1, 1)

# split data into training and validation sets
n_train = int(np.ceil(n / 2))
X_train = X[0:n_train]
T_train = T[0:n_train]
I_train = I[0:n_train]
f_train = f[0:n_train]
X_valid = X[n_train:n]
T_valid = T[n_train:n]
I_valid = I[n_train:n]
f_valid = f[n_train:n]

# define a kernel
a = 1
p = 2
kernel = care_survival.PolynomialKernel(a, p)
method = "kernel"
#method = "feature_map"

# set up the kernel tuning parameters
n_gammas = 20
gamma_min = 1e-6
gamma_max = 1e1

# compute concordance score on all data
with_concordance = ["train", "valid"]

# embedding
embedding = care_survival.embed(
    X_train,
    T_train,
    I_train,
    f_train,
    X_valid,
    T_valid,
    I_valid,
    f_valid,
    kernel,
    method,
)

# fit CARE2
#gamma = 0.1
#kernel_estimator = care_survival.KernelEstimatorCARE2(
    #embedding, gamma, with_concordance
    #)
#kernel_estimator.fit(None, None, None)
#print(kernel_estimator.score)



care2 = care_survival.CARE2(embedding, gamma_min, gamma_max, n_gammas,
                            verbose=False)
care2.fit()

#print(care2.beta_hat)

## view diagnostics
best = care2.best["aggregated"]["ln"]["valid"]
print("best theta value:", best.theta_hat)
print("best gamma value:", best.gamma)
print("concordance index:", best.score["concordance"]["valid"])

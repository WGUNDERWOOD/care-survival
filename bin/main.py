import numpy as np
from care_survival import data as care_data
from care_survival import kernel_polynomial as kernel_polynomial
#from care_survival import embedding as care_embedding

def get_test_data():
    n = 6
    d = 3
    m = 2
    X = np.random.random((n, d))
    T = np.random.random(n)
    I = np.random.randint(0, 2, size=n)
    f_tilde = np.random.random((n, m))
    f_0 = np.random.random(n)
    return care_data.Data(X, T, I, f_tilde, f_0)

def main():
    data = get_test_data()
    a = 1
    p = 2
    kernel = kernel_polynomial.PolynomialKernel(a, p)
    #print(data.X)
    #print(kernel.k(data.X, data.X))
    #print(kernel.norm_one())
    print(kernel.phi(data.X).shape)
    #embed = care_embedding.EmbeddingData(data)
    #print(data.f_0)

if __name__ == "__main__":
    main()

import numpy as np
from care_survival import data

def get_test_data():
    n = 20
    d = 3
    m = 2
    X = np.random.random((n, d))
    T = np.random.random(n)
    I = np.random.randint(0, 2, size=n)
    f_tilde = np.random.random((n, m))
    f_0 = np.random.random(n)
    return data.Data(X, T, I, f_tilde, f_0)

def main():
    data = get_test_data()
    print(data.f_0)

if __name__ == "__main__":
    main()

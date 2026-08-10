import pytest
import numpy as np
from care_survival.embedding import get_R, get_Z
from care_survival.metrics import get_concordance

def test_get_concordance():
    for rep in range(1000):
        n = 7
        T = np.array([0, 0, 0.5, 0.5, 0.6, 0.7, 1])
        f = np.random.random(n)
        N = np.random.random(n) > 0.5
        Z = get_Z(T)
        R = get_R(T)

        numerator = 0.0
        denominator = 0.0
        for i in range(n):
            for j in range(n):
                denominator += (T[i] > T[j]) * N[j]
                numerator += (T[i] > T[j]) * N[j] * (f[i] < f[j])
        if denominator > 0:
            concordance = numerator / denominator
        else:
            concordance = 0

        assert np.allclose(get_concordance(f, N, Z, R, True), concordance)
        assert np.allclose(get_concordance(f, N, Z, R, False), concordance)

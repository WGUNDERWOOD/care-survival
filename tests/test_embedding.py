import pytest
import numpy as np
from care_survival.embedding import get_R, get_Z, get_R_bar

def test_get_R():
    T = np.array([0, 0, 0.5, 0.5, 1])
    R = np.array([0, 0, 2, 2, 4])
    assert np.allclose(get_R(T), R)

def test_get_Z():
    T = np.array([0, 0, 0.5, 0.5, 1])
    Z = np.array([1, 1, 3, 3, 4])
    assert np.allclose(get_Z(T), Z)

def test_get_R_bar():
    T = np.array([0, 0, 0.5, 0.5, 1])
    R_bar = np.array([5, 5, 3, 3, 1]) / 5
    assert np.allclose(get_R_bar(T), R_bar)

import numpy as np
import math
from regression_classes import indices_from_mobius, indices_from_shapley

def gpt_test():
    import numpy as np
    np.random.seed(0)
    X = np.random.rand(10,5)

    phi_m, I_m = indices_from_mobius(X, k_add=2)
    phi_s, I_s = indices_from_shapley(X,   k_add=2)

    # shapes must match
    assert phi_m.shape == phi_s.shape     # (10,5)
    assert I_m.shape   == I_s.shape       # (10, C(5,2)=10)

    # values should be identical up to numerical tolerance
    assert np.allclose(phi_m, phi_s,  atol=1e-8)
    assert np.allclose(I_m,   I_s,      atol=1e-8)

def compare(x, tol=1e-8):
    phi_m, I2_m = indices_from_mobius(x)
    phi_s, I2_s = indices_from_shapley(x)
    print("max |φ_m−φ_s| =", np.max(np.abs(phi_m - phi_s)))
    print("max |I2_m−I2_s| =", np.max(np.abs(I2_m - I2_s)))
    assert np.allclose(phi_m, phi_s, atol=tol), "φ mismatch"
    assert np.allclose(I2_m, I2_s, atol=tol), "I2 mismatch"

def test_random():
    np.random.seed(0)
    x = np.random.rand(10, 5)
    compare(x)

def test_zero_matrix():
    x = np.zeros((5, 3))
    compare(x)
    phi, I2 = indices_from_shapley(x)
    assert np.allclose(phi, 0), "φ should be zero for zero matrix"
    assert np.allclose(I2, 0), "I2 should be zero for zero matrix"

def test_constant_columns():
    x = np.ones((8, 2))
    compare(x)
    phi, _ = indices_from_shapley(x)
    # identical columns => identical indices
    assert np.allclose(phi[:, 0], phi[:, 1], atol=1e-8), "φ unequal for identical columns"


def test_identity_matrix():
    x = np.eye(4)
    compare(x)

    # only up to 2-way interactions => C(4,2)=6 columns
    phi, I2 = indices_from_mobius(x, k_add=2)    # changed
    # φ should have same shape as x
    assert phi.shape == x.shape, (
        f"φ has wrong shape: expected {x.shape}, got {phi.shape}"
    )

    # I2 should be (n_samples, C(n_features,2))
    n, p = x.shape
    expected_pairs = math.comb(p, 2)
    assert I2.shape == (n, expected_pairs), (
        f"I2 has wrong shape: expected {(n, expected_pairs)}, got {I2.shape}"
    )


if __name__ == "__main__":
    gpt_test()
    test_random()
    test_zero_matrix()
    test_constant_columns()
    test_identity_matrix()
    print("All tests passed.")
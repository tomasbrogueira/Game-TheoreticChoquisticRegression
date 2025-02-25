import numpy as np
import itertools
from itertools import chain, combinations
from math import comb
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Utility functions for k-additive models (if needed) ---

def nParam_kAdd(kAdd, nAttr):
    """Return the number of parameters in a k-additive model for nAttr attributes."""
    total = 1  # intercept (or empty set)
    for r in range(1, kAdd+1):
        total += comb(nAttr, r)
    return total

def powerset(iterable, k_add):
    """
    Return the powerset (all subsets up to size k_add) of the given iterable.
    (Includes the empty set.)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(0, k_add+1))

# --- Choquet Transformation Functions ---

def choquet_matrix(X_orig):
    """
    Compute the full Choquet integral transformation (general version).
    WARNING: This version is computationally feasible only for small nAttr.
    
    For each sample, the transformation computes differences between consecutive ordered values.
    Returns both the transformed feature matrix (of shape (n_samples, 2**nAttr - 1)) and a
    list of all nonempty coalitions in a fixed order.
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    # For a fixed ordering of coalitions, we consider all nonempty subsets of {0, 1, ..., nAttr-1}
    all_coalitions = []
    for r in range(1, nAttr+1):
        all_coalitions.extend(list(itertools.combinations(range(nAttr), r)))
    nParams = len(all_coalitions)
    data_opt = np.zeros((nSamp, nParams))
    # For each sample, we first sort the features in ascending order.
    # Then, for j=0,..., nAttr-1, we assign the difference between the j-th and previous value
    # to the coalition corresponding to the set of indices from the sorted order starting at j.
    for i in range(nSamp):
        order = np.argsort(X_orig[i])
        sorted_vals = np.sort(X_orig[i])
        prev = 0.0
        for j in range(nAttr):
            # The coalition is defined as the set of original indices corresponding to the j-th 
            # smallest value and all larger ones.
            coalition = tuple(sorted(order[j:]))
            # Find the index in our fixed ordering:
            try:
                idx = all_coalitions.index(coalition)
            except ValueError:
                continue
            diff = sorted_vals[j] - prev
            prev = sorted_vals[j]
            data_opt[i, idx] = diff
    return data_opt, all_coalitions

def choquet_matrix_2add(X_orig):
    """
    Compute the 2-additive Choquet integral transformation.
    
    For each sample, we create features corresponding to:
      - the original (singleton) features, and
      - for each pair (i,j), the value min(x_i, x_j), with appropriate adjustments.
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    n_singletons = nAttr
    n_pairs = comb(nAttr, 2)
    nParams = n_singletons + n_pairs
    data_opt = np.zeros((nSamp, nParams))
    # Singleton features: use the original features.
    data_opt[:, :n_singletons] = X_orig
    # Pairwise features: for each pair (i, j), use min(x_i, x_j)
    idx = n_singletons
    for i in range(nAttr):
        for j in range(i+1, nAttr):
            data_opt[:, idx] = np.minimum(X_orig[:, i], X_orig[:, j])
            idx += 1
    return data_opt

def mlm_matrix(X_orig):
    """
    Compute the multilinear model transformation.
    
    For each nonempty subset of features, compute:
        prod_{i in subset} x_i * prod_{j not in subset} (1 - x_j)
    (This transformation grows exponentially with nAttr.)
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    subsets = list(chain.from_iterable(combinations(range(nAttr), r) for r in range(1, nAttr+1)))
    nParams = len(subsets)
    data_opt = np.zeros((nSamp, nParams))
    for i, subset in enumerate(subsets):
        prod_x = np.prod(X_orig[:, list(subset)], axis=1)
        complement = [j for j in range(nAttr) if j not in subset]
        if complement:
            prod_1_minus_x = np.prod(1 - X_orig[:, complement], axis=1)
        else:
            prod_1_minus_x = np.ones(nSamp)
        data_opt[:, i] = prod_x * prod_1_minus_x
    return data_opt

def mlm_matrix_2add(X_orig):
    """
    Compute the 2-additive multilinear model transformation.
    
    This transformation uses:
      - the original features, and
      - for each pair (i,j), the product x_i * x_j.
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    n_singletons = nAttr
    n_pairs = comb(nAttr, 2)
    nParams = n_singletons + n_pairs
    data_opt = np.zeros((nSamp, nParams))
    data_opt[:, :n_singletons] = X_orig
    idx = n_singletons
    for i in range(nAttr):
        for j in range(i+1, nAttr):
            data_opt[:, idx] = X_orig[:, i] * X_orig[:, j]
            idx += 1
    return data_opt

# --- Transformer and Classifier Classes ---

class ChoquetTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for Choquet or multilinear based feature transformations.
    
    Parameters
    ----------
    method : str, default="choquet_2add"
        The type of transformation to use. Options:
          - "choquet"      : full Choquet integral transformation (general version; use only for small nAttr)
          - "choquet_2add" : 2-additive Choquet integral transformation
          - "mlm"          : multilinear model transformation
          - "mlm_2add"     : 2-additive multilinear model transformation
    """
    def __init__(self, method="choquet_2add"):
        self.method = method
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        # For the full choquet method, precompute and store the fixed ordering (all coalitions)
        if self.method == "choquet":
            _, all_coalitions = choquet_matrix(X)
            self.all_coalitions_ = all_coalitions
        return self
    
    def transform(self, X):
        if self.method == "choquet":
            X_trans, all_coalitions = choquet_matrix(X)
            # In case fit wasn't called, store ordering now.
            if not hasattr(self, "all_coalitions_"):
                self.all_coalitions_ = all_coalitions
            return X_trans
        elif self.method == "choquet_2add":
            return choquet_matrix_2add(X)
        elif self.method == "mlm":
            return mlm_matrix(X)
        elif self.method == "mlm_2add":
            return mlm_matrix_2add(X)
        else:
            raise ValueError("Unknown method. Choose from 'choquet', 'choquet_2add', 'mlm', 'mlm_2add'.")

class ChoquisticRegression(BaseEstimator, ClassifierMixin):
    """
    Choquistic Regression classifier.
    
    This estimator first optionally scales the input data, then applies a Choquet (or multilinear) 
    transformation, and finally fits a LogisticRegression classifier.
    
    Parameters
    ----------
    method : str, default="choquet_2add"
        Transformation method to use. Options:
          - "choquet"      : full Choquet integral transformation (general version; use only for small nAttr)
          - "choquet_2add" : 2-additive Choquet integral transformation
          - "mlm"          : multilinear model transformation
          - "mlm_2add"     : 2-additive multilinear model transformation
    logistic_params : dict, default=None
        Additional keyword arguments for the underlying LogisticRegression.
    scale_data : bool, default=True
        If True, standardize the input features (zero mean, unit variance) before transformation.
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    def __init__(self, method="choquet_2add", logistic_params=None, scale_data=True, random_state=None):
        self.method = method
        self.logistic_params = logistic_params if logistic_params is not None else {}
        self.scale_data = scale_data
        self.random_state = random_state
    
    def fit(self, X, y):
        X = np.array(X)
        # Optionally scale data
        if self.scale_data:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
        # Transform features
        self.transformer_ = ChoquetTransformer(method=self.method)
        X_transformed = self.transformer_.fit_transform(X_scaled)
        # Fit logistic regression on transformed features
        self.classifier_ = LogisticRegression(random_state=self.random_state, **self.logistic_params)
        self.classifier_.fit(X_transformed, y)
        return self
    
    def predict(self, X):
        X = np.array(X)
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.transform(X_scaled)
        return self.classifier_.predict(X_transformed)
    
    def predict_proba(self, X):
        X = np.array(X)
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.transform(X_scaled)
        return self.classifier_.predict_proba(X_transformed)
    
    def score(self, X, y):
        return self.classifier_.score(
            self.transformer_.transform(self.scaler_.transform(X)) if self.scale_data else self.transformer_.transform(X),
            y
        )
    
    def compute_shapley_values(self):
        """
        Compute the Shapley values (marginal contributions) for each feature based on the learned
        game parameters. This method is implemented only for the full "choquet" method.
        
        Returns
        -------
        phi : ndarray of shape (n_features,)
            The Shapley value for each feature.
        """
        if self.method != "choquet":
            raise ValueError("Shapley value computation is only implemented for the full 'choquet' method.")
        # Get the number of features.
        m = self.transformer_.n_features_in_
        # Get the fixed list of coalitions (each is a tuple of feature indices).
        all_coalitions = self.transformer_.all_coalitions_
        # Get the learned coefficients (v) from logistic regression.
        # Here we assume a single-output classifier and take the first row.
        v = self.classifier_.coef_[0]
        # Compute the factorial denominator.
        from math import factorial
        denom = factorial(m)
        phi = np.zeros(m)
        # For each feature j, sum over all subsets B ⊆ M\{j}
        for j in range(m):
            # Iterate over all subsets of M without feature j.
            for r in range(0, m):  # r = |B|
                # All subsets B of size r that do not contain j.
                for B in itertools.combinations([i for i in range(m) if i != j], r):
                    # v(B) is 0 if B is empty; otherwise, find its coefficient.
                    if len(B) == 0:
                        vB = 0.0
                    else:
                        try:
                            vB = v[all_coalitions.index(tuple(sorted(B)))]
                        except ValueError:
                            # If not found, skip (should not occur)
                            continue
                    # v(B U {j}) must exist (B U {j} is nonempty)
                    Bj = tuple(sorted(B + (j,)))
                    try:
                        vBj = v[all_coalitions.index(Bj)]
                    except ValueError:
                        continue
                    weight = (factorial(m - r - 1) * factorial(r)) / denom
                    phi[j] += weight * (vBj - vB)
        return phi







# Interpretability functions

from math import factorial


def compute_shapley_values_explicit(v, m, all_coalitions):
    """
    Compute Shapley values explicitly given learned coefficients v, number of features m,
    and the list of all nonempty coalitions (in a fixed order).
    """
    phi = np.zeros(m)
    denom = factorial(m)
    for j in range(m):
        # Iterate over all subsets B of M without j.
        for r in range(0, m):
            for B in itertools.combinations([i for i in range(m) if i != j], r):
                # v(B) is taken as 0 for empty set, otherwise lookup in all_coalitions
                vB = 0.0 if len(B) == 0 else v[all_coalitions.index(tuple(sorted(B)))]
                # B ∪ {j}
                Bj = tuple(sorted(B + (j,)))
                try:
                    vBj = v[all_coalitions.index(Bj)]
                except ValueError:
                    continue
                weight = factorial(m - r - 1) * factorial(r) / denom
                phi[j] += weight * (vBj - vB)
    return phi


def compute_banzhaf_indices(v, m, all_coalitions):
    """
    Compute Banzhaf indices explicitly given learned coefficients v, number of features m,
    and the list of all nonempty coalitions (in a fixed order).
    """
    phi_b = np.zeros(m)
    norm = 2 ** (m - 1)
    for j in range(m):
        for r in range(0, m):
            for B in itertools.combinations([i for i in range(m) if i != j], r):
                vB = 0.0 if len(B) == 0 else v[all_coalitions.index(tuple(sorted(B)))]
                Bj = tuple(sorted(B + (j,)))
                try:
                    vBj = v[all_coalitions.index(Bj)]
                except ValueError:
                    continue
                phi_b[j] += (vBj - vB)
    return phi_b / norm

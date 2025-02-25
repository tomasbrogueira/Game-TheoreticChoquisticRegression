import os
import numpy as np
import pandas as pd
import itertools
from itertools import chain, combinations
from math import comb, factorial
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Utility functions for k-additive models
# =============================================================================
def nParam_kAdd(kAdd, nAttr):
    """Return the number of parameters in a k-additive model for nAttr attributes."""
    total = 1  # intercept (or empty set)
    for r in range(1, kAdd + 1):
        total += comb(nAttr, r)
    return total

def powerset(iterable, k_add):
    """
    Return the powerset (all subsets up to size k_add) of the given iterable.
    (Includes the empty set.)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(0, k_add + 1))

# =============================================================================
# Choquet Transformation Functions
# =============================================================================
def choquet_matrix(X_orig):
    """
    Compute the full Choquet integral transformation (general version).
    WARNING: This version is computationally feasible only for small nAttr.
    Returns both the transformed feature matrix and a list of all nonempty coalitions.
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    all_coalitions = []
    for r in range(1, nAttr + 1):
        all_coalitions.extend(list(itertools.combinations(range(nAttr), r)))
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}

    data_opt = np.zeros((nSamp, len(all_coalitions)))
    for i in range(nSamp):
        order = np.argsort(X_orig[i])
        sorted_vals = np.sort(X_orig[i])
        prev = 0.0
        for j in range(nAttr):
            coalition = tuple(sorted(order[j:]))
            idx = coalition_to_index.get(coalition)
            if idx is None:
                continue
            diff = sorted_vals[j] - prev
            prev = sorted_vals[j]
            data_opt[i, idx] = diff
    return data_opt, all_coalitions

def choquet_matrix_2add(X_orig):
    """
    Compute the 2-additive Choquet integral transformation.
    For each sample, creates features corresponding to:
      - the original (singleton) features, and
      - for each pair (i, j), the value min(x_i, x_j)
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    n_singletons = nAttr
    n_pairs = comb(nAttr, 2)
    data_opt = np.zeros((nSamp, n_singletons + n_pairs))
    data_opt[:, :n_singletons] = X_orig
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
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    subsets = list(chain.from_iterable(combinations(range(nAttr), r) for r in range(1, nAttr + 1)))
    data_opt = np.zeros((nSamp, len(subsets)))
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
    Uses:
      - the original features, and
      - for each pair (i, j), the product x_i * x_j.
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    n_singletons = nAttr
    n_pairs = comb(nAttr, 2)
    data_opt = np.zeros((nSamp, n_singletons + n_pairs))
    data_opt[:, :n_singletons] = X_orig
    idx = n_singletons
    for i in range(nAttr):
        for j in range(i+1, nAttr):
            data_opt[:, idx] = X_orig[:, i] * X_orig[:, j]
            idx += 1
    return data_opt

# =============================================================================
# Choquet Transformer
# =============================================================================
class ChoquetTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for Choquet or multilinear based feature transformations.
    
    Parameters
    ----------
    method : str, default="choquet_2add"
        The transformation method. Options: "choquet", "choquet_2add", "mlm", "mlm_2add".
    """
    def __init__(self, method="choquet_2add"):
        self.method = method
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if self.method == "choquet":
            _, all_coalitions = choquet_matrix(X)
            self.all_coalitions_ = all_coalitions
        return self
    
    def transform(self, X):
        if self.method == "choquet":
            X_trans, all_coalitions = choquet_matrix(X)
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

# =============================================================================
# Implementation 1: Composition-based ChoquisticRegression
# =============================================================================
class ChoquisticRegression_Composition(BaseEstimator, ClassifierMixin):
    """
    Choquistic Regression classifier.
    
    This estimator first optionally scales the input data, then applies a Choquet (or multilinear) 
    transformation, and finally fits a LogisticRegression classifier.
    
    Parameters
    ----------
    method : str, default="choquet_2add"
        Transformation method to use.
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
        self.classifier_ = LogisticRegression(random_state=self.random_state, max_iter=10000, **self.logistic_params)
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
        X = np.array(X)
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.transform(X_scaled)
        return self.classifier_.score(X_transformed, y)
    
    def decision_function(self, X):
        """
        Compute the decision function (log-odds) for the input samples.
        """
        X = np.array(X)
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.transform(X_scaled)
        return self.classifier_.decision_function(X_transformed)
    
    def compute_shapley_values(self):
        """
        Compute the Shapley values (marginal contributions) for each feature based on the learned
        game parameters. This method is implemented only for the full "choquet" method.
        """
        if self.method != "choquet":
            raise ValueError("Shapley value computation is only implemented for the full 'choquet' method.")
        m = self.transformer_.n_features_in_
        all_coalitions = self.transformer_.all_coalitions_
        v = self.classifier_.coef_[0]
        from math import factorial
        denom = factorial(m)
        phi = np.zeros(m)
        import itertools
        for j in range(m):
            for r in range(0, m):
                for B in itertools.combinations([i for i in range(m) if i != j], r):
                    vB = 0.0 if len(B) == 0 else v[all_coalitions.index(tuple(sorted(B)))]
                    Bj = tuple(sorted(B + (j,)))
                    try:
                        vBj = v[all_coalitions.index(Bj)]
                    except ValueError:
                        continue
                    weight = (factorial(m - r - 1) * factorial(r)) / denom
                    phi[j] += weight * (vBj - vB)
        return phi


# =============================================================================
# Implementation 2: Inheritance-based ChoquisticRegression
# =============================================================================
class ChoquisticRegression_Inheritance(LogisticRegression):
    def __init__(self, method="choquet_2add", scale_data=True, logistic_params=None, random_state=None):
        self.method = method
        self.scale_data = scale_data
        self.random_state = random_state
        if logistic_params is None:
            logistic_params = {}
        # Set random_state and default solver if not provided.
        logistic_params.setdefault('random_state', random_state)
        logistic_params.setdefault('solver', 'newton-cg')
        self.logistic_params = logistic_params
        # Call the parent constructor with the provided parameters.
        super().__init__(**self.logistic_params)
        self.transformer_ = ChoquetTransformer(method=self.method)
        if self.scale_data:
            self.scaler_ = StandardScaler()
        else:
            self.scaler_ = None

    def fit(self, X, y, **fit_params):
        X = np.array(X)
        # Store the raw feature count separately.
        self.raw_n_features_in_ = X.shape[1]
        if self.scale_data:
            self.scaler_ = StandardScaler().fit(X)
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.fit_transform(X_scaled)
        # The parent's fit() uses the transformed features.
        return super().fit(X_transformed, y, **fit_params)

    def _transform(self, X):
        X = np.array(X)
        # If X has the raw feature count, apply scaling and transformation.
        # Otherwise, assume it's already transformed.
        if X.shape[1] == self.raw_n_features_in_:
            if self.scale_data:
                X_scaled = self.scaler_.transform(X)
            else:
                X_scaled = X
            return self.transformer_.transform(X_scaled)
        else:
            return X

    def predict(self, X):
        X_transformed = self._transform(X)
        return super().predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self._transform(X)
        return super().predict_proba(X_transformed)

    def decision_function(self, X):
        X_transformed = self._transform(X)
        return super().decision_function(X_transformed)



    def score(self, X, y):
        X = np.array(X)
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.transform(X_scaled)
        return super().score(X_transformed, y)

    def compute_shapley_values(self):
        """
        Compute Shapley values for the full Choquet method.
        Only implemented when method == "choquet".
        """
        if self.method != "choquet":
            raise ValueError("Shapley value computation is only implemented for the full 'choquet' method.")
        m = self.transformer_.n_features_in_
        all_coalitions = self.transformer_.all_coalitions_
        v = self.coef_[0]
        denom = factorial(m)
        phi = np.zeros(m)
        for j in range(m):
            for r in range(0, m):
                for B in itertools.combinations([i for i in range(m) if i != j], r):
                    vB = 0.0 if len(B) == 0 else v[all_coalitions.index(tuple(sorted(B)))]
                    Bj = tuple(sorted(B + (j,)))
                    try:
                        vBj = v[all_coalitions.index(Bj)]
                    except ValueError:
                        continue
                    weight = (factorial(m - r - 1) * factorial(r)) / denom
                    phi[j] += weight * (vBj - vB)
        return phi

# =============================================================================
# Explicit Interpretability Functions (shared utilities)
# =============================================================================
def compute_shapley_values_explicit(v, m, all_coalitions):
    """
    Compute Shapley values explicitly given learned coefficients v, number of features m,
    and the list of all nonempty coalitions.
    """
    phi = np.zeros(m)
    denom = factorial(m)
    for j in range(m):
        for r in range(0, m):
            for B in itertools.combinations([i for i in range(m) if i != j], r):
                vB = 0.0 if len(B) == 0 else v[all_coalitions.index(tuple(sorted(B)))]
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
    and the list of all nonempty coalitions.
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

# =============================================================================
# Choose Default Implementation
# =============================================================================
# You can switch the default implementation here.
# For example, to use the inheritance-based version, change the next line accordingly.
#ChoquisticRegression = ChoquisticRegression_Composition
# Alternatively:
ChoquisticRegression = ChoquisticRegression_Inheritance

import numpy as np
import itertools
from itertools import chain, combinations
from math import comb
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Utility functions for k-additive models ---

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

# --- Choquet / Multilinear Transformation Functions ---

def choquet_matrix(X_orig):
    """
    Compute the full Choquet integral transformation (general version).
    WARNING: This version is computationally feasible only for small nAttr.
    
    For each sample, the transformation computes differences between ordered values.
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    # Sort features for each sample (rowwise)
    X_sorted = np.sort(X_orig, axis=1)
    X_sort_ext = np.concatenate((np.zeros((nSamp, 1)), X_sorted), axis=1)
    # Build list of all nonempty subsets (each represented as a tuple of indices)
    all_coalitions = []
    for r in range(1, nAttr+1):
        all_coalitions.extend(list(itertools.combinations(range(nAttr), r)))
    nParams = len(all_coalitions)
    data_opt = np.zeros((nSamp, nParams))
    # For each sample, use its sorted order to assign differences to the corresponding coalition.
    # (This is a direct but not highly efficient implementation.)
    for i in range(nSamp):
        # Get the order of the features for sample i
        order = np.argsort(X_orig[i])
        sorted_vals = np.sort(X_orig[i])
        prev = 0.0
        for j in range(nAttr):
            # Coalition: indices in the original order from current sorted index to the end
            coalition = tuple(sorted(order[j:]))
            # Find the index of this coalition in our list
            try:
                idx = all_coalitions.index(coalition)
            except ValueError:
                continue
            diff = sorted_vals[j] - prev
            prev = sorted_vals[j]
            data_opt[i, idx] = diff
    return data_opt

def choquet_matrix_2add(X_orig):
    """
    Compute the 2-additive Choquet integral transformation.
    
    For each sample, we create features corresponding to:
      - the original (singleton) features,
      - and for each pair (i,j), the value min(x_i, x_j).
    
    (Additional adjustments from the paper can be incorporated as needed.)
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    n_singletons = nAttr
    n_pairs = comb(nAttr, 2)
    nParams = n_singletons + n_pairs
    data_opt = np.zeros((nSamp, nParams))
    # Singleton features: copy original features
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
    
    For each nonempty subset of features, compute the product:
        prod_{i in subset} x_i * prod_{j not in subset} (1-x_j)
    (This transformation grows exponentially with nAttr.)
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    # Exclude empty set
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
    Transformer for Choquet/multilinear based feature transformations.
    
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
        return self
    
    def transform(self, X):
        if self.method == "choquet":
            return choquet_matrix(X)
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
    
    This estimator first applies a Choquet (or multilinear) integral based transformation
    to the input features and then performs logistic regression on the transformed features.
    
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
        return self.classifier_.score(
            self.transformer_.transform(self.scaler_.transform(X)) if self.scale_data else self.transformer_.transform(X),
            y
        )

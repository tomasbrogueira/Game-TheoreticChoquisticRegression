import numpy as np
import itertools
from math import comb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# --- Utility functions for k-additive models ---
def nParam_kAdd(kAdd, nAttr):
    total = 1  # intercept (or empty set)
    for r in range(1, kAdd + 1):
        total += comb(nAttr, r)
    return total

def powerset(iterable, k_add):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(0, k_add + 1))

# --- Choquet Transformation Functions ---
def choquet_matrix(X_orig):
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    all_coalitions = []
    for r in range(1, nAttr + 1):
        all_coalitions.extend(list(itertools.combinations(range(nAttr), r)))
    data_opt = np.zeros((nSamp, len(all_coalitions)))
    for i in range(nSamp):
        order = np.argsort(X_orig[i])
        sorted_vals = np.sort(X_orig[i])
        prev = 0.0
        for j in range(nAttr):
            coalition = tuple(sorted(order[j:]))
            try:
                idx = all_coalitions.index(coalition)
            except ValueError:
                continue
            diff = sorted_vals[j] - prev
            prev = sorted_vals[j]
            data_opt[i, idx] = diff
    return data_opt, all_coalitions

def choquet_matrix_2add(X_orig):
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
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    subsets = list(itertools.chain.from_iterable(itertools.combinations(range(nAttr), r) for r in range(1, nAttr + 1)))
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

# --- Choquet Transformer ---
class ChoquetTransformer(BaseEstimator, TransformerMixin):
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

# --- Choquistic Regression via Inheritance from LogisticRegression ---
class ChoquisticRegression(LogisticRegression):
    """
    Choquistic Regression implemented by inheriting from LogisticRegression.
    It transforms the input using a Choquet (or multilinear) transformation before 
    applying logistic regression. All parameters for LogisticRegression should be passed 
    via the logistic_params dictionary, except for random_state which is explicitly handled.
    
    Parameters:
      method : str, default="choquet_2add"
          The transformation method to use. Options: "choquet", "choquet_2add", "mlm", "mlm_2add".
      scale_data : bool, default=True
          Whether to standardize input data.
      logistic_params : dict, default=None
          A dictionary of parameters to pass to the underlying LogisticRegression.
          For example: {'penalty': None, 'max_iter': 10000}
      random_state : int or None, default=None
          Random state for reproducibility. This is also passed into logistic_params.
    """
    def __init__(self, method="choquet_2add", scale_data=True, logistic_params=None, random_state=None):
        self.method = method
        self.scale_data = scale_data
        self.random_state = random_state
        # Set default logistic_params if not provided
        if logistic_params is None:
            logistic_params = {}
        # Ensure random_state is set in the parameters for LogisticRegression
        logistic_params.setdefault('random_state', random_state)
        self.logistic_params = logistic_params
        # Initialize the parent LogisticRegression with the given parameters
        super().__init__(**self.logistic_params)
        self.transformer_ = ChoquetTransformer(method=self.method)
        if self.scale_data:
            self.scaler_ = StandardScaler()
        else:
            self.scaler_ = None

    def fit(self, X, y, **fit_params):
        X = np.array(X)
        if self.scale_data:
            self.scaler_ = StandardScaler().fit(X)
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.fit_transform(X_scaled)
        return super().fit(X_transformed, y, **fit_params)

    def predict(self, X):
        X = np.array(X)
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.transform(X_scaled)
        return super().predict(X_transformed)

    def predict_proba(self, X):
        X = np.array(X)
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.transform(X_scaled)
        return super().predict_proba(X_transformed)

    def score(self, X, y):
        X = np.array(X)
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.transform(X_scaled)
        return super().score(X_transformed, y)

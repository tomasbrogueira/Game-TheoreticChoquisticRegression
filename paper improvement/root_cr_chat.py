import os
import numpy as np
import pandas as pd
import itertools
from itertools import chain, combinations
from math import comb, factorial
import math
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Utility functions for k-additive models (using the alternative equations)
# =============================================================================
def nParam_kAdd(kAdd, nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr, ii+1)
    return aux_numb

def powerset(iterable, k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes
    powerset([1,2,..., m],m) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m) ... (1, ..., m)
    powerset([1,2,..., m],2) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add+1))

def choquet_matrix_mobius(X_orig, kadd):
    nSamp, nAttr = X_orig.shape  # Number of samples and attributes
    k_add_numb = nParam_kAdd(kadd, nAttr)
    data_opt = np.zeros((nSamp, k_add_numb - 1))
    # Note: This version expects X_orig to be a pandas DataFrame.
    for i, s in enumerate(powerset(range(nAttr), kadd)):
        s = list(s)
        if len(s) > 0:
            data_opt[:, i-1] = np.min(X_orig.iloc[:, s], axis=1)
    return data_opt

def choquet_matrix(X_orig):
    X_orig_sort = np.sort(X_orig)
    X_orig_sort_ind = np.array(np.argsort(X_orig))
    nSamp, nAttr = X_orig.shape  # Number of samples and attributes
    X_orig_sort_ext = np.concatenate((np.zeros((nSamp, 1)), X_orig_sort), axis=1)
    
    sequence = np.arange(nAttr)
    
    combin = (99) * np.ones((2**nAttr - 1, nAttr))
    count = 0
    for ii in range(nAttr):
        combin[count:count+comb(nAttr, ii+1), 0:ii+1] = np.array(list(itertools.combinations(sequence, ii+1)))
        count += comb(nAttr, ii+1)
    
    data_opt = np.zeros((nSamp, 2**nAttr - 1))
    for ii in range(nAttr):
        for jj in range(nSamp):
            list1 = combin.tolist()
            aux = list1.index(np.concatenate((np.sort(X_orig_sort_ind[jj, ii:]), 99 * np.ones((ii,))), axis=0).tolist())
            data_opt[jj, aux] = X_orig_sort_ext[jj, ii+1] - X_orig_sort_ext[jj, ii]
    return data_opt

def choquet_matrix_2add(X_orig):
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape  # Number of samples and attributes
    k_add = 2
    k_add_numb = nParam_kAdd(k_add, nAttr)
    
    coalit = np.zeros((k_add_numb, nAttr))
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
        
    data_opt = np.zeros((nSamp, k_add_numb))
    for i in range(nAttr):
        data_opt[:, i+1] = data_opt[:, i+1] + X_orig[:, i]
        for i2 in range(i+1, nAttr):
            data_opt[:, (coalit[:, [i, i2]] == 1).all(axis=1)] = np.min([X_orig[:, i], X_orig[:, i2]], axis=0).reshape(nSamp, 1)
        for ii in range(nAttr+1, len(coalit)):
            if coalit[ii, i] == 1:
                data_opt[:, ii] = data_opt[:, ii] + (-1/2) * X_orig[:, i]
    return data_opt[:, 1:]

def mlm_matrix(X_orig):
    nSamp, nAttr = X_orig.shape  # Number of samples and attributes
    X_orig = np.array(X_orig)
    data_opt = np.zeros((nSamp, 2**nAttr))
    coalitions = list(powerset(range(nAttr), nAttr))
    for i, s in enumerate(coalitions):
        for j in range(nSamp):
            prod_in = np.prod(X_orig[j, list(s)]) if len(s) > 0 else 1
            complement = [idx for idx in range(nAttr) if idx not in s]
            prod_out = np.prod(1 - X_orig[j, complement]) if len(complement) > 0 else 1
            data_opt[j, i] = prod_in * prod_out
    return data_opt[:, 1:]

def mlm_matrix_2add(X_orig):
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape  # Number of samples and attributes
    k_add = 2
    k_add_numb = nParam_kAdd(k_add, nAttr)
    
    coalit = np.zeros((k_add_numb, nAttr))
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
        
    data_opt = np.zeros((nSamp, k_add_numb))
    for i in range(nAttr):
        data_opt[:, i+1] = data_opt[:, i+1] + X_orig[:, i]
        for i2 in range(i+1, nAttr):
            data_opt[:, (coalit[:, [i, i2]] == 1).all(axis=1)] = (X_orig[:, i] * X_orig[:, i2]).reshape(nSamp, 1)
        for ii in range(nAttr+1, len(coalit)):
            if coalit[ii, i] == 1:
                data_opt[:, ii] = data_opt[:, ii] + (-1/2) * X_orig[:, i]
    return data_opt[:, 1:]

def tr_shap2game(nAttr, k_add):
    '''Return the transformation matrix from Shapley interaction indices to game, given a k-additive model'''
    from scipy.special import bernoulli
    nBern = bernoulli(k_add)  # Bernoulli numbers
    k_add_numb = nParam_kAdd(k_add, nAttr)
    
    coalit = np.zeros((k_add_numb, nAttr))
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
        
    matrix_shap2game = np.zeros((k_add_numb, k_add_numb))
    for i in range(coalit.shape[0]):
        for i2 in range(k_add_numb):
            aux2 = int(sum(coalit[i2, :]))
            aux3 = int(sum(coalit[i, :] * coalit[i2, :]))
            aux4 = 0
            for i3 in range(int(aux3 + 1)):
                aux4 += comb(aux3, i3) * nBern[aux2 - i3]
            matrix_shap2game[i, i2] = aux4
    return matrix_shap2game

def tr_banz2game(nAttr, k_add):
    '''Return the transformation matrix from Banzhaf interaction indices, given a k-additive model, to game'''
    from scipy.special import bernoulli
    nBern = bernoulli(k_add)  # Bernoulli numbers
    k_add_numb = nParam_kAdd(k_add, nAttr)
    
    coalit = np.zeros((k_add_numb, nAttr))
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
        
    matrix_banz2game = np.zeros((k_add_numb, k_add_numb))
    for i in range(coalit.shape[0]):
        A = coalit[i, :]
        cardA = int(sum(A))
        for i2 in range(k_add_numb):
            B = coalit[i2, :]
            cardB = int(sum(B))
            cardBminusA = sum((B - A) > 0)
            matrix_banz2game[i, i2] = ((1/2)**cardB) * ((-1)**cardBminusA)
    return matrix_banz2game

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

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
            _ = choquet_matrix(X)
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
        self.classifier_ = LogisticRegression(**self.logistic_params)
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
        # Note: Detailed computation of Shapley values would require tracking the ordering of coalitions.
        # Here we provide a placeholder implementation.
        v = self.classifier_.coef_[0]
        denom = factorial(m)
        phi = np.zeros(m)
        import itertools
        for j in range(m):
            for r in range(0, m):
                for B in itertools.combinations([i for i in range(m) if i != j], r):
                    vB = 0.0 if len(B) == 0 else v[0]  # Placeholder
                    Bj = tuple(sorted(B + (j,)))
                    try:
                        vBj = v[0]  # Placeholder
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
        if self.method != "choquet":
            raise ValueError("Shapley value computation is only implemented for the full 'choquet' method.")
        m = self.transformer_.n_features_in_
        v = self.coef_[0]
        denom = factorial(m)
        phi = np.zeros(m)
        import itertools
        for j in range(m):
            for r in range(0, m):
                for B in itertools.combinations([i for i in range(m) if i != j], r):
                    vB = 0.0 if len(B) == 0 else v[0]
                    Bj = tuple(sorted(B + (j,)))
                    try:
                        vBj = v[0]
                    except ValueError:
                        continue
                    weight = (factorial(m - r - 1) * factorial(r)) / denom
                    phi[j] += weight * (vBj - vB)
        return phi

# =============================================================================
# Explicit Interpretability Functions (shared utilities)
# =============================================================================
def compute_shapley_values_explicit(v, m, all_coalitions):
    phi = np.zeros(m)
    denom = factorial(m)
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
                phi[j] += (factorial(m - r - 1) * factorial(r)) / denom * (vBj - vB)
    return phi

def compute_banzhaf_indices(v, m, all_coalitions):
    phi_b = np.zeros(m)
    norm = 2 ** (m - 1)
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
                phi_b[j] += (vBj - vB)
    return phi_b / norm

# =============================================================================
# Choose Default Implementation
# =============================================================================
ChoquisticRegression = ChoquisticRegression_Composition
# Alternatively:
# ChoquisticRegression = ChoquisticRegression_Inheritance

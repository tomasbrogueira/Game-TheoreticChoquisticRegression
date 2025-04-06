"""
Multi-Linear Model (MLM) aggregation functions for logistic regression extensions.

This module implements the Multi-Linear Model (MLM) aggregation functions,
which are an alternative to the Choquet integral for capturing non-linear
interactions between features.
"""

import numpy as np
from itertools import chain, combinations
from .choquet import nParam_kAdd, powerset


def mlm_matrix(X_orig):
    """
    Compute the multilinear model transformation.
    
    The multilinear extension is defined as:
    MLM(x) = Σ_{T⊆N} m(T) * Π_{i∈T} x_i * Π_{j∉T} (1-x_j)
    
    This function computes the basis functions for each subset T.
    
    Parameters:
    -----------
    X_orig : array-like
        Original feature matrix (must be scaled to [0,1] range by the caller)
        
    Returns:
    --------
    numpy.ndarray : Full MLM basis transformation
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape

    subsets = list(chain.from_iterable(combinations(range(nAttr), r) for r in range(1, nAttr + 1)))
    data_opt = np.zeros((nSamp, len(subsets)))
    
    # X_orig is already scaled from [0,1]
    for i, subset in enumerate(subsets):
        # Calculate product of selected features: Π_{i∈T} x_i
        
        prod_x = np.prod(X_orig[:, list(subset)], axis=1)
        
        # Calculate product of complements: Π_{j∉T} (1-x_j)
        complement = [j for j in range(nAttr) if j not in subset]
        if complement:
            prod_1_minus_x = np.prod(1 - X_orig[:, complement], axis=1)
        else:
            prod_1_minus_x = np.ones(nSamp)
        
        # Compute the basis function: Π_{i∈T} x_i * Π_{j∉T} (1-x_j)
        data_opt[:, i] = prod_x * prod_1_minus_x
    
    return data_opt


def mlm_matrix_2add(X_orig):
    """
    Compute the 2-additive multilinear model transformation.
    
    In a 2-additive MLM, the formula is:
    f_ML(v, x_i) = ∑_j x_i,j(φ_j^B - (1/2)∑_{j'≠j} I_{j,j'}^B) + ∑_{j≠j'} x_i,j x_i,j' I_{j,j'}^B
    
    Parameters:
    -----------
    X_orig : array-like
        Original feature matrix (should be scaled to [0,1])
        
    Returns:
    --------
    numpy.ndarray : 2-additive MLM basis transformation
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
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
            data_opt[:, (coalit[:, [i, i2]]==1).all(axis=1)] = (X_orig[:, i] * X_orig[:, i2]).reshape(nSamp, 1)
        for ii in range(nAttr+1, len(coalit)):
            if coalit[ii, i] == 1:
                data_opt[:, ii] = data_opt[:, ii] + (-1/2)*X_orig[:, i]
    return data_opt[:, 1:]


class MLMTransformer:
    """
    Transformer for Multi-Linear Model (MLM) based feature transformations.
    
    This transformer implements the MLM aggregation functions:
    - Full multilinear model 
    - 2-additive multilinear model
    
    Parameters
    ----------
    method : str, default="mlm_2add"
        The transformation method. Options:
        - "mlm": Full multilinear model
        - "mlm_2add": 2-additive multilinear model
    k_add : int or None, default=None
        Additivity level for k-additive models. If not specified and method is 
        "mlm", defaults to using all features. Ignored for "mlm_2add" (where k_add=2 is implicit).
    """

    def __init__(self, method="mlm_2add", k_add=None):
        self.method = method
        self.k_add = k_add
        
        valid_methods = ["mlm", "mlm_2add"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
        if k_add is not None:
            if not isinstance(k_add, int) or k_add < 1:
                raise ValueError("k_add must be a positive integer")
            
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        y : array-like of shape (n_samples,), optional
            Target values. Ignored.
            
        Returns
        -------
        self : object
            Returns self.
        """
        from sklearn.utils.validation import check_array
        import warnings
        
        X = check_array(X, ensure_min_features=1)
        self.n_features_in_ = X.shape[1]
        if self.k_add is not None and self.k_add > self.n_features_in_:
            warnings.warn(
                f"k_add ({self.k_add}) is greater than the number of features ({self.n_features_in_}). "
                f"Setting k_add to {self.n_features_in_}."
            )
            self.k_add = self.n_features_in_
            self.k_add_ = self.n_features_in_
        else:
            self.k_add_ = self.k_add
        return self
    
    def transform(self, X):
        """
        Transform the data using the selected MLM method.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_transformed_features)
            Transformed data
        """
        from sklearn.utils.validation import check_array, check_is_fitted
        import warnings
        
        check_is_fitted(self, ["n_features_in_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but expected {self.n_features_in_}.")
        
        if self.method == "mlm":
            if self.k_add is not None:
                warnings.warn("k-additive MLM not yet fully implemented. Using full MLM instead.")
            return mlm_matrix(X)
        elif self.method == "mlm_2add":
            return mlm_matrix_2add(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")


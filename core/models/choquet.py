"""
Choquet integral and related aggregation functions for logistic regression extensions.

This module implements various representations of the Choquet integral and related
aggregation functions, including:
- Game representation
- Mobius representation
- Shapley representation

Each representation is a different mathematical basis for the same underlying concept,
but they have different interpretability properties.
"""

import numpy as np
import itertools
from itertools import chain, combinations
from math import comb, factorial
from sklearn.base import BaseEstimator, TransformerMixin
import warnings


# =============================================================================
# Utility functions for k-additive models
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


# =============================================================================
# Choquet Transformation Functions
# =============================================================================

def choquet_k_additive_game(X_orig, k_add=None):
    """
    Compute the k-additive Choquet integral transformation using game representation.
    
    Parameters:
    -----------
    X_orig : array-like of shape (n_samples, n_features)
        Original feature matrix
    k_add : int, optional
        Level of additivity. If None, uses full model.
        
    Returns:
    --------
    numpy.ndarray : Transformed feature matrix using game representation
    """
    X_orig = np.asarray(X_orig)
    nSamp, nAttr = X_orig.shape

    if k_add is None:
        k_add = nAttr
    elif k_add > nAttr:
        raise ValueError("k_add cannot be greater than the number of attributes.")

    # Generate all valid coalitions up to size k_add
    all_coalitions = []
    for r in range(1, min(k_add, nAttr)+1):
        all_coalitions.extend(list(combinations(range(nAttr), r)))
    
    
    # Calculate number of features in the transformed space
    n_transformed = len(all_coalitions)
    
    # Initialize output matrix
    transformed = np.zeros((nSamp, n_transformed))
    
    # Create a mapping from coalition tuples to indices
    coalition_to_idx = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    
    # Process each sample following the original method
    for i in range(nSamp):
        x = X_orig[i]
        
        # Sort feature indices by their values (ascending)
        sorted_indices = np.argsort(x)
        sorted_values = x[sorted_indices]
        
        # Add a sentinel value (0) at the beginning
        sorted_values_ext = np.concatenate([[0], sorted_values])
        
        # For each position in the sorted list
        for j in range(nAttr):
            # Calculate difference with previous value
            diff = sorted_values_ext[j+1] - sorted_values_ext[j]
            
            # Get the current set of "active" features (those from position j onward)
            # This matches the original algorithm's logic for finding the right coalition
            active_features = tuple(sorted(sorted_indices[j:]))
            
            # Skip if the active features set is too large for our k_add restriction
            if len(active_features) > k_add:
                continue
                
            # If this exact coalition exists, assign the difference to it
            if active_features in coalition_to_idx:
                idx = coalition_to_idx[active_features]
                transformed[i, idx] = diff
    
    return transformed


def choquet_k_additive_mobius(X_orig, k_add=None):
    """
    Compute the k-additive Choquet integral transformation using Mobius representation.
    
    Parameters:
    -----------
    X_orig : array-like of shape (n_samples, n_features)
        Original feature matrix
    k_add : int, optional
        Level of additivity. If None, uses full model.
        
    Returns:
    --------
    numpy.ndarray : Transformed feature matrix using Mobius representation
    """
    X_orig = np.asarray(X_orig)
    nSamp, nAttr = X_orig.shape

    if k_add is None:
        k_add = nAttr
    elif k_add > nAttr:
        raise ValueError("k_add cannot be greater than the number of attributes.")

    # Generate all valid coalitions up to size k_add
    all_coalitions = []
    for r in range(1, min(k_add, nAttr)+1):
        all_coalitions.extend(list(combinations(range(nAttr), r)))

    # Calculate number of features in the transformed space
    n_transformed = len(all_coalitions)

    # Initialize output matrix (no longer restricted to non-negative values)
    transformed = np.zeros((nSamp, n_transformed))

    # Process each sample directly without sorting
    for i in range(nSamp):
        x = X_orig[i]

        # For each coalition, compute its value directly
        for idx, coalition in enumerate(all_coalitions):
            # For singleton coalition, use the feature value directly
            if len(coalition) == 1:
                transformed[i, idx] = x[coalition[0]]
            # For larger coalitions, use the minimum value across the coalition
            else:
                coalition_values = [x[j] for j in coalition]
                transformed[i, idx] = min(coalition_values)

    return transformed


def choquet_matrix_2add(X_orig):
    """
    Compute the 2-additive Choquet integral transformation using Shapley representation.
    
    In a 2-additive Choquet integral, the formula is:
    f_CI(v, x_i) = ∑_j x_i,j(φ_j^S - (1/2)∑_{j'≠j} I_{j,j'}^S) + ∑_{j≠j'} (x_i,j ∧ x_i,j') I_{j,j'}^S
    
    Parameters:
    -----------
    X_orig : array-like
        Original feature matrix
        
    Returns:
    --------
    numpy.ndarray : 2-additive Choquet integral basis transformation using Shapley representation
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
            data_opt[:, (coalit[:, [i, i2]]==1).all(axis=1)] = (np.min([X_orig[:, i], X_orig[:, i2]], axis=0)).reshape(nSamp, 1)
        for ii in range(nAttr+1, len(coalit)):
            if coalit[ii, i] == 1:
                data_opt[:, ii] = data_opt[:, ii] + (-1/2)*X_orig[:, i]
    return data_opt[:, 1:]


# =============================================================================
# ChoquetTransformer Class
# =============================================================================
class ChoquetTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for Choquet integral based feature transformations.
    
    This transformer implements various fuzzy measure-based transformations:
    - Choquet integral 
        - k-additive Choquet integral with game representation
        - k-additive Choquet integral with Möbius representation
        - 2-additive Choquet integral with shapely representation
    
    Parameters
    ----------
    method : str, default="choquet_2add"
        The transformation method. Options:
        - "choquet": General Choquet integral
        - "choquet_2add": 2-additive Choquet integral shapely representation
    representation : str, default="game"
        For method="choquet", defines the representation to use:
        - "game": Uses game-based representation
        - "mobius": Uses Möbius representation
        Ignored for other methods.
    k_add : int or None, default=None
        Additivity level for k-additive models. If not specified and method is 
        "choquet", defaults to using all features. Ignored for methods ending 
        with "_2add" (where k_add=2 is implicit).
    """

    def __init__(self, method="choquet_2add", representation="game", k_add=None):
        self.method = method
        self.representation = representation
        self.k_add = k_add
        
        valid_methods = ["choquet", "choquet_2add"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        if method == "choquet":
            valid_representations = ["game", "mobius"]
            if representation not in valid_representations:
                raise ValueError(f"For method='choquet', representation must be one of {valid_representations}")
            
        valid_representations = ["game", "mobius"]
        if representation not in valid_representations:
            raise ValueError(f"Representation must be one of {valid_representations}")
    
        if k_add is not None:
            if not isinstance(k_add, int) or k_add < 1:
                raise ValueError("k_add must be a positive integer")
            # For methods ending with '_2add' k_add is implicit; otherwise, k_add is used.
            
    
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
        from sklearn.utils.validation import check_array, check_is_fitted
        
        X = check_array(X, ensure_min_features=1)
        self.n_features_in_ = X.shape[1]
        if self.k_add is not None and self.k_add > self.n_features_in_:
            warnings.warn(
                f"k_add ({self.k_add}) is greater than the number of features ({self.n_features_in_}). "
                f"Setting k_add to {self.n_features_in_}."
            )
            self.k_add = self.n_features_in_  # <-- Update self.k_add too
            self.k_add_ = self.n_features_in_
        else:
            self.k_add_ = self.k_add
        return self
    
    def transform(self, X):
        """
        Transform the data using the selected Choquet integral method.
        
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
        
        check_is_fitted(self, ["n_features_in_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but expected {self.n_features_in_}.")
        
        if self.method == "choquet":
            if self.representation == "game":
                return choquet_k_additive_game(X, k_add=self.k_add)
            elif self.representation == "mobius":
                return choquet_k_additive_mobius(X, k_add=self.k_add)
            else:
                raise ValueError(f"Unknown representation: {self.representation}")
        elif self.method == "choquet_2add":
            return choquet_matrix_2add(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")


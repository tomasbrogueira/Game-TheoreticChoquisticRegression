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
from math import comb
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

def choquet_k_additive_mobius(X_orig, k_add=None):
    """
    Möbius basis: one feature per coalition A, |A|<=k_add, m_x(A)=sum_{B⊆A}(-1)^{|A|-|B|}min(x[B]).
    Returns shape (N, Σ_{r=1}^k_add C(n,r)).
    """
    X = np.asarray(X_orig, float)
    N, n = X.shape
    if k_add is None:
        k_add = n
    elif k_add > n:
        raise ValueError
    # list coalitions
    coalitions = []
    for r in range(1, k_add+1):
        coalitions += list(combinations(range(n), r))
    # include empty for inversion
    full_coals = [()] + coalitions
    idx = {A:i for i,A in enumerate(full_coals)}
    # compute v_x(A)
    V = np.zeros((N, len(full_coals)))
    for A,i in idx.items():
        if len(A)==0:
            V[:,i]=0
        elif len(A)==1:
            V[:,i]=X[:,A[0]]
        else:
            V[:,i]=X[:,list(A)].min(axis=1)
    # Möbius inversion
    M = np.zeros_like(V)
    for A,iA in idx.items():
        if len(A)==0: continue
        for r in range(len(A)+1):
            for B in combinations(A, r):
                sign = (-1)**(len(A)-len(B))
                M[:,iA] += sign * V[:, idx[B]]
    # drop empty and keep only coalitions
    return M[:,1:1+len(coalitions)]


def choquet_k_additive_shapley(X_orig, k_add=None):
    """
    Shapley basis: one feature per coalition A, |A|<=k_add,
    I_x(A)=sum_{C⊆A}(-1)^{|A|-|C|}/(|A|-|C|+1)*min(x[C]).
    Returns shape (N, Σ_{r=1}^k_add C(n,r)).
    """
    X = np.asarray(X_orig, float)
    N, n = X.shape
    if k_add is None:
        k_add = n
    elif k_add > n:
        raise ValueError
    # list v_coals including empty
    v_coals = [()]
    for r in range(1, k_add+1): v_coals += list(combinations(range(n), r))
    idx = {C:i for i,C in enumerate(v_coals)}
    # compute v_x(C)
    V = np.zeros((N, len(v_coals)))
    for C,i in idx.items():
        if len(C)==0:
            V[:,i]=0
        elif len(C)==1:
            V[:,i]=X[:,C[0]]
        else:
            V[:,i]=X[:,list(C)].min(axis=1)
    # output coalitions
    out_coals = []
    for r in range(1, k_add+1): out_coals += list(combinations(range(n), r))
    T = np.zeros((N, len(out_coals)))
    for j,A in enumerate(out_coals):
        s = len(A)
        for r in range(s+1):
            coeff = (-1)**(s-r)/(s-r+1)
            for C in combinations(A, r):
                T[:,j] += coeff * V[:, idx[C]]
    return T


def choquet_k_additive_game(X_orig, k_add=None, full=False):
    """
    Game basis: enforce k-additivity via Möbius truncation, then suffix-differences.
    If full=True scatter into 2^n-1 vector, else return (N,n) matrix of g_j*mu_k.
    """
    X = np.asarray(X_orig, float)
    N, n = X.shape
    # 1) build full Möbius and truncate
    M_full = choquet_k_additive_mobius(X, k_add=None)
    # reconstruct full_coals index
    full_coals = []
    for r in range(1, n+1): full_coals += list(combinations(range(n), r))
    idx_fc = {A:i for i,A in enumerate(full_coals)}
    # m_trunc shape (N,2^n-1)
    if k_add is None:
        m_trunc = M_full.copy()
    else:
        m_trunc = np.zeros_like(M_full)
        for A,i in idx_fc.items():
            if len(A)<=k_add:
                m_trunc[:,i] = M_full[:,i]
    # 2) compute mu_k on suffixes
    suffixes = [tuple(range(j,n)) for j in range(n)]
    mu_k = np.zeros((N,n))
    for j,S in enumerate(suffixes):
        for r in range(1,len(S)+1):
            for A in combinations(S,r):
                mu_k[:,j] += m_trunc[:, idx_fc[A]]
    # 3) sorted diffs
    order = np.argsort(X, axis=1)
    Xs = np.take_along_axis(X, order, axis=1)
    Xext = np.concatenate([np.zeros((N,1)), Xs], axis=1)
    diffs = Xext[:,1:]-Xext[:,:-1]
    G = diffs * mu_k
    if not full:
        return G
    # 4) scatter into full
    T_full = np.zeros((N, len(full_coals)))
    for i in range(N):
        for j,S in enumerate(suffixes):
            idx = idx_fc[S]
            T_full[i,idx] = G[i,j]
    return T_full


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
    
    Parameters
    ----------
    representation : str, default="game"
        For method="choquet", defines the representation to use:
        - "game": Uses game-based representation
        - "mobius": Uses Möbius representation
        Ignored for other methods.
    k_add : int or None, default=None
        Additivity level for k-additive models. If not specified and method is 
        "choquet", defaults to using all features.
    """

    def __init__(self, representation="game", k_add=None):
        self.representation = representation
        self.k_add = k_add
        
        valid_representations = ["game", "mobius", "shapley"]
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
        
        if self.representation == "game":
            return choquet_k_additive_game(X, k_add=self.k_add)
        elif self.representation == "mobius":
            return choquet_k_additive_mobius(X, k_add=self.k_add)
        elif self.representation == "shapley":
            return choquet_k_additive_shapley(X, k_add=self.k_add)
        else:
            raise ValueError(f"Unknown representation: {self.representation}")


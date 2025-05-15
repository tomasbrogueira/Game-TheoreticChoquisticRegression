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


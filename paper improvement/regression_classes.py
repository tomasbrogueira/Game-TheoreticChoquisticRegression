import numpy as np
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



def choquet_matrix_mobius(X_orig, kadd):
    nSamp, nAttr = X_orig.shape  # Number of samples and attributes
    k_add_numb = nParam_kAdd(kadd, nAttr)
    data_opt = np.zeros((nSamp, k_add_numb-1))
    for i, s in enumerate(powerset(range(nAttr), kadd)):
        s = list(s)
        if len(s) > 0:
            data_opt[:, i-1] = np.min(X_orig.iloc[:, s], axis=1)
    return data_opt


def choquet_matrix(X_orig, all_coalitions=None):
    """
    X_orig : array-like of shape (n_samples, n_features)
    all_coalitions : list of tuples, optional. Pre-computed list of all nonempty coalitions.
    Returns the transformed feature matrix (choquet matrix) and the list of coalitions.
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    if all_coalitions is None:
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
        for j in range(i + 1, nAttr):
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
        for j in range(i + 1, nAttr):
            data_opt[:, idx] = X_orig[:, i] * X_orig[:, j]
            idx += 1
    return data_opt


# =============================================================================
# ChoquetTransformer Class
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
            # Pre-compute and store the coalition structure, which depends solely on the number of features.
            _, all_coalitions = choquet_matrix(X)
            self.all_coalitions_ = all_coalitions
        return self

    def transform(self, X):
        if self.method == "choquet":
            if not hasattr(self, "all_coalitions_"):
                raise ValueError("Transformer not fitted: please call fit() before transform().")
            # Reuse the pre-computed coalition structure to compute the transformed features.
            X_trans, _ = choquet_matrix(X, all_coalitions=self.all_coalitions_)
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
    Choquistic Regression classifier using composition.

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

    def __init__(self, method="choquet_2add", logistic_params={
        "penalty": None,
        "max_iter": 1000,
        "random_state": 0,
        "solver": "newton-cg",
    }, scale_data=True):
        self.method = method
        self.scale_data = scale_data
        self.random_state = logistic_params.get("random_state", 0)
        self.logistic_params = logistic_params
        self.scaler_ = StandardScaler()

    def fit(self, X, y):
        X = np.array(X)
        if self.scale_data:
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
        self.transformer_ = ChoquetTransformer(method=self.method)
        X_transformed = self.transformer_.fit_transform(X_scaled)
        self.classifier_ = LogisticRegression(**self.logistic_params)
        self.classifier_.fit(X_transformed, y)
        # Save the raw feature count for later use in 2-additive case.
        self.n_features_in_ = X.shape[1]
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
        X = np.array(X)
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.transform(X_scaled)
        return self.classifier_.decision_function(X_transformed)

    def compute_shapley_values(self):
        """
        Compute Shapley values for the model.
        For method "choquet" we use the full coalition approach.
        For method "choquet_2add", we compute both:
          - "marginal": the main effect contributions (directly from the regression coefficients)
          - "shapley": the Shapley values computed as main_effect + 0.5 * (sum of interactions)
        Returns:
          - For "choquet": a 1D numpy array (the full Shapley values)
          - For "choquet_2add": a dictionary with keys "marginal" and "shapley"
        """
        if self.method == "choquet":
            m = self.transformer_.n_features_in_
            all_coalitions = self.transformer_.all_coalitions_
            v = self.classifier_.coef_[0]
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

        elif self.method == "choquet_2add":
            # For 2-additive models, Shapley values have a simplified formula:
            # φj = μ({j}) + 0.5 * Σi≠j I({i,j})
            marginal_contrib = coef[:nAttr].copy()  # μ({j}) values
            interactions = coef[nAttr:]             # I({i,j}) values

            # Build interaction matrix from the flat interaction vector
            interaction_matrix = np.zeros((nAttr, nAttr))
            counter = 0
            for i in range(nAttr):
                for j in range(i + 1, nAttr):
                    interaction_matrix[i, j] = interactions[counter]
                    interaction_matrix[j, i] = interactions[counter]  # Symmetry
                    counter += 1

            # Apply the formula: φj = μ({j}) + 0.5 * Σi≠j I({i,j})
            shapley_vals = marginal_contrib + 0.5 * np.sum(interaction_matrix, axis=1)
            return {"marginal": marginal_contrib, "shapley": shapley_vals}
        else:
            raise ValueError("Shapley value computation is only implemented for 'choquet' and 'choquet_2add' methods.")


# =============================================================================
# Implementation 2: Inheritance-based ChoquisticRegression
# =============================================================================
class ChoquisticRegression_Inheritance(LogisticRegression):
    """
    Choquistic Regression classifier using inheritance.
    
    This class extends LogisticRegression and performs scaling and Choquet transformation
    before fitting the model.
    
    Parameters
    ----------
    method : str, default="choquet_2add"
        Transformation method to use.
    scale_data : bool, default=True
        If True, standardize input features.
    logistic_params : dict, default=None
        Additional parameters for LogisticRegression.
    random_state : int or None, default=None
        Random state.
    """

    def __init__(self, method="choquet_2add", scale_data=True, logistic_params={
        "penalty": None,
        "max_iter": 1000,
        "random_state": 0,
        "solver": "newton-cg",
    }):
        self.method = method
        self.scale_data = scale_data
        self.random_state = logistic_params.get("random_state", 0)
        if logistic_params is None:
            logistic_params = {}
        # Set default parameters if not provided.
        logistic_params.setdefault("random_state", 0)
        logistic_params.setdefault("solver", "newton-cg")
        self.logistic_params = logistic_params
        super().__init__(**self.logistic_params)
        self.transformer_ = ChoquetTransformer(method=self.method)

    def fit(self, X, y, **fit_params):
        X = np.array(X)
        self.raw_n_features_in_ = X.shape[1]
        if self.scale_data:
            self.scaler_ = StandardScaler().fit(X)
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_transformed = self.transformer_.fit_transform(X_scaled)
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
        """
        Compute Shapley values for the model.
        For method "choquet" we use the full coalition approach.
        For method "choquet_2add", we compute both:
          - "marginal": the direct main effect contributions (from regression coefficients)
          - "shapley": the Shapley values computed as main_effect + 0.5 * (sum of interactions)
        Returns a dictionary (for choquet_2add) or a 1D array (for choquet).
        """
        if self.method == "choquet":
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
            return phi  # Fixed indentation - moved outside the for loop

        elif self.method == "choquet_2add":
            # For 2-additive models, Shapley values have a simplified formula:
            # φj = μ({j}) + 0.5 * Σi≠j I({i,j})
            marginal_contrib = coef[:nAttr].copy()  # μ({j}) values
            interactions = coef[nAttr:]             # I({i,j}) values

            # Build interaction matrix from the flat interaction vector
            interaction_matrix = np.zeros((nAttr, nAttr))
            counter = 0
            for i in range(nAttr):
                for j in range(i + 1, nAttr):
                    interaction_matrix[i, j] = interactions[counter]
                    interaction_matrix[j, i] = interactions[counter]  # Symmetry
                    counter += 1

            # Apply the formula: φj = μ({j}) + 0.5 * Σi≠j I({i,j})
            shapley_vals = marginal_contrib + 0.5 * np.sum(interaction_matrix, axis=1)
            return {"marginal": marginal_contrib, "shapley": shapley_vals}
        else:
            raise ValueError("Shapley value computation is only implemented for 'choquet' and 'choquet_2add'.")


# =============================================================================
# Utility functions 
# =============================================================================

def compute_m(T, v, all_coalitions, k):
    """
    Compute the Möbius transform m(T) for coalition T.
    For a singleton, m({i}) = v({i}).
    For larger T (with |T| <= k), compute:
         m(T) = v(T) - sum_{U ⊂ T, U ≠ ∅} m(U)
    If T is not provided in all_coalitions, assume v(T)=0.
    """
    if len(T) == 1:
        return v[all_coalitions.index(T)]
    try:
        v_T = v[all_coalitions.index(T)]
    except ValueError:
        v_T = 0.0
    total = 0.0
    for r in range(1, len(T)):
        for U in itertools.combinations(T, r):
            U = tuple(sorted(U))
            total += compute_m(U, v, all_coalitions, k)
    return v_T - total

def compute_v_S(S, v, all_coalitions, k=None):
    """
    Compute the value v(S) for any coalition S.
    If k is None (full model), then S is expected to be in all_coalitions.
    For a k-additive model (e.g., k=2), reconstruct v(S) as:
         v(S) = sum_{T ⊆ S, 1 ≤ |T| ≤ min(|S|, k)} m(T)
    """
    if len(S) == 0:
        return 0.0
    if (k is None) and (S in all_coalitions):
        return v[all_coalitions.index(S)]
    total = 0.0
    max_r = min(len(S), k) if k is not None else len(S)
    for r in range(1, max_r + 1):
        for T in itertools.combinations(S, r):
            T = tuple(sorted(T))
            # Only add m(T) if T is in the provided list; otherwise assume zero.
            if T in all_coalitions:
                total += compute_m(T, v, all_coalitions, k)
    return total


def reconstruct_m(T, v, all_coalitions, k):
    """
    Recursively compute the Möbius coefficient m(T) for coalition T in a k-additive model.
    For a singleton T, m(T) = v(T).
    For T with |T| > 1 (and |T| ≤ k), we set:
         m(T) = v(T) - sum_{U ⊂ T, U ≠ ∅} m(U)
    If T is not explicitly in all_coalitions, we assume v(T)=0.
    """
    T = tuple(sorted(T))
    if len(T) == 0:
        return 0.0
    if len(T) == 1:
        try:
            return v[all_coalitions.index(T)]
        except ValueError:
            return 0.0
    # For coalitions of size > 1:
    try:
        vT = v[all_coalitions.index(T)]
    except ValueError:
        vT = 0.0
    total = 0.0
    # Sum over all proper, nonempty subsets of T.
    for r in range(1, len(T)):
        for U in itertools.combinations(T, r):
            total += reconstruct_m(U, v, all_coalitions, k)
    return vT - total

def reconstruct_v_S(S, v, all_coalitions, k=None):
    """
    Reconstruct the value v(S) for any coalition S.
    
    - If S is in all_coalitions, return v(S) directly.
    - Otherwise (i.e. in a k-additive model, if |S| > k) compute:
          v(S) = sum_{T ⊆ S, 1 ≤ |T| ≤ min(|S|, k)} m(T)
    where m(T) is computed by reconstruct_m.
    
    Parameters:
      S : tuple
          A sorted tuple representing the coalition.
      v : 1D numpy array of learned coefficients.
      all_coalitions : list
          List of coalition tuples (each sorted). In a k-additive model this list
          typically contains only coalitions of size ≤ k.
      k : int or None
          The additivity order. If None, the full model is assumed (so S must be found).
    
    Returns:
      float: The reconstructed value v(S).
    """
    S = tuple(sorted(S))
    # If S is in the learned list, use it.
    if S in all_coalitions:
        return v[all_coalitions.index(S)]
    # Otherwise, if k is specified, reconstruct from all subsets T ⊆ S of size at most k.
    if k is not None:
        total = 0.0
        max_r = min(len(S), k)
        for r in range(1, max_r + 1):
            for T in itertools.combinations(S, r):
                T = tuple(sorted(T))
                total += reconstruct_m(T, v, all_coalitions, k)
        return total
    # If k is None and S is missing, return 0.
    return 0.0


# =============================================================================
# Explicit Interpretability Functions 
# =============================================================================



# =============================================================================
# Power indices 
# =============================================================================
def compute_shapley_values(v, m, all_coalitions):
    """
    Compute Shapley values according to equation 6 in a more efficient way.
    """
    coalition_to_idx = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    phi = np.zeros(m)
    
    # Memoization cache to avoid recomputing the same coalition values
    coalition_values = {}
    
    def get_coalition_value(coalition):
        if not coalition:
            return 0.0  # Empty set has value 0
        
        # Check cache first
        if coalition in coalition_values:
            return coalition_values[coalition]
            
        # Calculate and cache the value
        try:
            value = v[coalition_to_idx[coalition]]
        except (KeyError, IndexError):
            value = 0.0
            
        coalition_values[coalition] = value
        return value
    
    for j in range(m):
        others = [i for i in range(m) if i != j]
        
        # Iterate through all subsets B ⊆ M\{j}
        for r in range(len(others) + 1):
            for B in itertools.combinations(others, r):
                B = tuple(sorted(B))
                Bj = tuple(sorted(B + (j,)))
                
                # Calculate marginal contribution
                vB = get_coalition_value(B)
                vBj = get_coalition_value(Bj)
                
                # Calculate weight: (m-|B|-1)!|B|! / m!
                weight = factorial(m - r - 1) * factorial(r) / factorial(m)
                phi[j] += weight * (vBj - vB)
                
    return phi

def compute_banzhaf_indices(v, m, all_coalitions):
    """
    Compute Banzhaf power indices explicitly given learned coefficients v, number of features m,
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
                phi_b[j] += vBj - vB
    return phi_b / norm


def compute_banzhaf_power_indices(v, m, all_coalitions, k=None):
    """
    Compute Banzhaf power indices for all features (Equation 13).
    
    φ_j^B = (1/2^(m-1)) * ∑_{B⊆M\{j}} [μ(B∪{j}) - μ(B)]
    
    Parameters:
    -----------
    v : array-like
        Capacity/game values for each coalition
    m : int
        Number of features
    all_coalitions : list of tuples
        List of all coalitions in the model
    k : int or None
        Additivity limit for k-additive models (e.g., k=2 for 2-additive)
        
    Returns:
    --------
    numpy.ndarray : Banzhaf power indices for each feature
    """
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    coalition_values = {}  # Memoization cache
    
    phi_b = np.zeros(m)
    norm = 2 ** (m - 1)
    
    def get_value(coalition):
        if not coalition:
            return 0.0
        if coalition in coalition_values:
            return coalition_values[coalition]
        
        if k is None or len(coalition) <= k:
            try:
                value = v[coalition_to_index.get(coalition, -1)] if coalition in coalition_to_index else 0.0
            except (KeyError, IndexError):
                value = 0.0
        else:
            value = reconstruct_v_S(coalition, v, all_coalitions, k)
        coalition_values[coalition] = value
        return value
    
    for j in range(m):
        others = [i for i in range(m) if i != j]
        
        # Iterate through all subsets B ⊆ M\{j}
        for r in range(len(others) + 1):
            for B in itertools.combinations(others, r):
                B = tuple(sorted(B))
                Bj = tuple(sorted(B + (j,)))
                
                # Calculate marginal contribution
                vB = get_value(B)
                vBj = get_value(Bj)
                phi_b[j] += vBj - vB
                
    return phi_b / norm


# =============================================================================
# Interaction matrixes 
# =============================================================================


def compute_banzhaf_interaction_matrix_iterative(v, m, all_coalitions, k=None):
    """
    Compute the Banzhaf interaction matrix for a model.
    
    For features i and j the index is defined as:
      I(i,j) = (1 / 2^(m-2)) * sum_{S ⊆ N\\{i,j}} [v(S ∪ {i,j}) - v(S ∪ {i}) - v(S ∪ {j}) + v(S)]
    
    Parameters:
      v: 1D numpy array of learned coefficients. For a k-additive model, v contains only 
         the coefficients for coalitions of size ≤ k (with the empty coalition assumed to be 0).
      m: number of original features.
      all_coalitions: list of coalition tuples (each tuple is sorted). In a full model, this should
         include every nonempty coalition; in a k-additive model, only those with |coalition| ≤ k.
      k: the additivity order (e.g., 2 for 2-additive). If None, the full model is assumed.
    
    Returns:
      A symmetric m x m numpy array with the Banzhaf interaction indices.
    """
    interaction_matrix = np.zeros((m, m))
    norm = 2 ** (m - 2)
    features = list(range(m))
    for i in range(m):
        for j in range(i + 1, m):
            total = 0.0
            # Compute over all subsets S of N\{i,j}
            others = [f for f in features if f not in (i, j)]
            for r in range(0, len(others) + 1):
                for S in itertools.combinations(others, r):
                    S = tuple(sorted(S))
                    vS   = compute_v_S(S, v, all_coalitions, k)
                    Si   = tuple(sorted(S + (i,)))
                    vSi  = compute_v_S(Si, v, all_coalitions, k)
                    Sj   = tuple(sorted(S + (j,)))
                    vSj  = compute_v_S(Sj, v, all_coalitions, k)
                    Sij  = tuple(sorted(S + (i, j)))
                    vSij = compute_v_S(Sij, v, all_coalitions, k)
                    total += vSij - vSi - vSj + vS
            interaction_matrix[i, j] = total / norm
            interaction_matrix[j, i] = total / norm
    return interaction_matrix


def compute_banzhaf_interaction_matrix(v, m, all_coalitions, k=None, k=None):
    """
    Compute the Banzhaf interaction indices for all pairs of features (Equation 14).
    
    I_{j,j'}^B = (1/2^(m-2)) * ∑_{B⊆M\{j,j'}} [μ(B∪{j,j'}) - μ(B∪{j}) - μ(B∪{j'}) + μ(B)]
    
    With k parameter to limit computation for large feature spaces.
    """
    interaction_matrix = np.zeros((m, m))
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    coalition_values = {}  # Memoization cache
    
    def get_value(coalition):
        if not coalition:
            return 0.0
        if coalition in coalition_values:
            return coalition_values[coalition]
        
        if k is None:
            value = v[coalition_to_index.get(coalition, -1)] if coalition in coalition_to_index else 0.0
        else:
            value = reconstruct_v_S(coalition, v, all_coalitions, k)
        coalition_values[coalition] = value
        return value
    
    # The constant factor in the formula
    norm = 2 ** (m - 2)
    
    # For every distinct pair of features (i,j)
    for i in range(m):
        for j in range(i+1, m):
            total = 0.0
            # Define the set of "other" features (excluding i and j)
            others = [k for k in range(m) if k not in (i, j)]
            
            # Determine the range of subset sizes to iterate over
            max_r = len(others)
            if k is not None:
                max_r = min(max_r, k)
                
            for r in range(max_r + 1):
                for B in itertools.combinations(others, r):
                    B = tuple(sorted(B))
                    
                    # Compute the four required coalitions
                    vB = get_value(B)
                    vBi = get_value(tuple(sorted(B + (i,))))
                    vBj = get_value(tuple(sorted(B + (j,))))
                    vBij = get_value(tuple(sorted(B + (i, j))))
                    
                    # Apply the interaction formula
                    total += vBij - vBi - vBj + vB
                    
            # Normalize and store the result (ensure matrix symmetry)
            interaction_matrix[i, j] = total / norm
            interaction_matrix[j, i] = interaction_matrix[i, j]
            
    return interaction_matrix


def compute_mlm_interaction_matrix(v, m, all_coalitions, k=None):
    """
    Compute the pairwise Banzhaf interaction indices for the multilinear model.
    
    For each distinct pair (i, j), the interaction index is computed as:
    
      IB(i,j) = (1/(2^(m-2))) * Σ_{B ⊆ M\{i,j}} [ v(B ∪ {i,j}) - v(B ∪ {i}) - v(B ∪ {j}) + v(B) ]
    
    If k is provided, then the summation is restricted to subsets B with |B| < k.
    For a 2-additive model, set k = 2 (i.e., only B with |B| = 0 or 1 are included).
    For full multilinear interactions, use k = None.
    
    Parameters:
        v : array-like
            The value function (or capacity) evaluated on each coalition.
            The ordering in v must match the ordering of coalitions in 'all_coalitions'.
        m : int
            Total number of features.
        all_coalitions : list of tuples
            List of coalitions (each a sorted tuple of feature indices), including the empty set.
        k : int or None, optional
            If provided, restricts the summation to subsets B with |B| < k.
            
    Returns:
        interaction_matrix : np.array of shape (m, m)
            A symmetric matrix where entry (i, j) is the Banzhaf interaction index between features i and j.
    """
    interaction_matrix = np.zeros((m, m))
    # Create a lookup dictionary so we can retrieve the value of v for a given coalition quickly.
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    
    # The constant factor in the multilinear model for a pair (i,j)
    constant_factor = 1.0 / (2 ** (m - 2))
    
    # Loop over all distinct pairs (i,j)
    for i in range(m):
        for j in range(i+1, m):
            total = 0.0
            # Define the set of "other" features (excluding i and j)
            others = [k for k in range(m) if k not in (i, j)]
            # Determine the range of subset sizes to iterate over.
            if k is None:
                r_range = range(0, len(others) + 1)
            else:
                r_range = range(0, min(k, len(others) + 1))
            for r in r_range:
                for B in itertools.combinations(others, r):
                    B = tuple(sorted(B))
                    # Get the value for coalition B
                    vB = v[coalition_to_index[B]] if B in coalition_to_index else 0.0
                    # Define the coalitions B ∪ {i}, B ∪ {j}, and B ∪ {i,j}
                    Bi = tuple(sorted(B + (i,)))
                    Bj = tuple(sorted(B + (j,)))
                    Bij = tuple(sorted(B + (i, j)))
                    vBi = v[coalition_to_index[Bi]] if Bi in coalition_to_index else 0.0
                    vBj = v[coalition_to_index[Bj]] if Bj in coalition_to_index else 0.0
                    vBij = v[coalition_to_index[Bij]] if Bij in coalition_to_index else 0.0
                    total += (vBij - vBi - vBj + vB)
            interaction_matrix[i, j] = constant_factor * total
            interaction_matrix[j, i] = interaction_matrix[i, j]  # Ensure symmetry
    return interaction_matrix




def compute_choquet_interaction_matrix(v, m, all_coalitions, k=None):
    """
    Compute the pairwise Shapley interaction indices between features.
    
    For each distinct pair (i, j), the interaction index is computed as:
    
      IS(i,j) = Σ₍B ⊆ M\{i,j}₎ [ ((m - |B| - 2)! * |B|!)/(m - 1)! * ( v(B ∪ {i,j}) - v(B ∪ {i}) - v(B ∪ {j}) + v(B) ) ]
    
    The summation is taken over all subsets B ⊆ M\{i,j}. If k is provided,
    then only subsets with |B| < k are used. For example, to compute the 2-additive
    interaction index (as used in the 2-additive Choquet integral), set k = 2 
    (so that only B with |B| = 0 or 1 are considered). For full Choquet interactions, set 
    k = None (or omit it), which sums over all subsets.
    
    Parameters:
        v : array-like
            Value function evaluated on each coalition. The ordering in v must match that 
            of coalitions in all_coalitions.
        m : int
            Total number of features.
        all_coalitions : list of tuples
            List of coalitions (each is a sorted tuple of feature indices), including the empty set.
        k : int or None, optional
            If provided, restricts the summation to subsets B with |B| < k.
            For 2-additive, use k = 2. For full Choquet, use None.
            
    Returns:
        interaction_matrix : np.array of shape (m, m)
            A symmetric matrix where entry (i, j) is the Shapley interaction index between features i and j.
    """
    interaction_matrix = np.zeros((m, m))
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    
    # Loop over all distinct pairs of features
    for i in range(m):
        for j in range(i+1, m):
            total = 0.0
            # Compute the list of features excluding i and j
            others = [k for k in range(m) if k not in (i, j)]
            # Determine the maximum size for the subsets:
            # If k is provided, sum only over B with |B| < k;
            # Otherwise, sum over all subsets.
            if k is None:
                r_range = range(0, len(others)+1)
            else:
                r_range = range(0, min(k, len(others)+1))
            for r in r_range:
                for B in itertools.combinations(others, r):
                    B = tuple(sorted(B))
                    # Weight: ((m - |B| - 2)! * |B|!)/(m - 1)!
                    weight = factorial(m - r - 2) * factorial(r) / factorial(m - 1)
                    # Retrieve the value function for the coalition and its extensions.
                    vB = v[coalition_to_index[B]] if B in coalition_to_index else 0.0
                    Bi = tuple(sorted(B + (i,)))
                    Bj = tuple(sorted(B + (j,)))
                    Bij = tuple(sorted(B + (i, j)))
                    vBi = v[coalition_to_index[Bi]] if Bi in coalition_to_index else 0.0
                    vBj = v[coalition_to_index[Bj]] if Bj in coalition_to_index else 0.0
                    vBij = v[coalition_to_index[Bij]] if Bij in coalition_to_index else 0.0
                    total += weight * (vBij - vBi - vBj + vB)
            interaction_matrix[i, j] = total
            interaction_matrix[j, i] = total  # Ensure symmetry
    return interaction_matrix


def compute_mlm_banzhaf_interaction_indices(v, m, all_coalitions, k=None):
    """
    Compute Banzhaf interaction indices specifically for MultiLinear Models.
    
    This implementation is optimized for MLM where we know the structure
    of the coalitions in advance, and can use direct lookups instead of
    reconstruction formulas needed for Choquet integrals.
    
    Parameters:
    -----------
    v : array-like
        The learned MLM coefficients
    m : int
        Number of features
    all_coalitions : list of tuples
        All coalitions in the model
    k : int or None
        If provided, limits computation to subsets B with |B| < k
        
    Returns:
    --------
    dict : Contains 'power_indices' (1D array) and 'interaction_matrix' (2D array)
    """
    # Create lookup dictionary for fast coalition value retrieval
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    
    # 1. Compute power indices (singleton interactions)
    power_indices = np.zeros(m)
    norm_power = 2 ** (m - 1)
    
    # 2. Compute interaction matrix (pairwise interactions)
    interaction_matrix = np.zeros((m, m))
    norm_interaction = 2 ** (m - 2)
    
    # Memoization cache for coalition values
    coal_values = {}
    
    def get_value(coalition):
        """Get coalition value with memoization"""
        if not coalition:  # Empty set
            return 0.0
            
        if coalition in coal_values:
            return coal_values[coalition]
            
        try:
            value = v[coalition_to_index[coalition]] if coalition in coalition_to_index else 0.0
        except (KeyError, IndexError):
            value = 0.0
            
        coal_values[coalition] = value
        return value
    
    # For power indices: φ_j^B = (1/2^(m-1)) * ∑_{B⊆M\{j}} [μ(B∪{j}) - μ(B)]
    for j in range(m):
        others = [i for i in range(m) if i != j]
        for r in range(len(others) + 1):
            if k is not None and r >= k:
                break
                
            for B in itertools.combinations(others, r):
                B = tuple(sorted(B))
                Bj = tuple(sorted(B + (j,)))
                power_indices[j] += get_value(Bj) - get_value(B)
    
    power_indices /= norm_power
    
    # For interaction matrix: I_{i,j}^B = (1/2^(m-2)) * ∑_{B⊆M\{i,j}} [μ(B∪{i,j}) - μ(B∪{i}) - μ(B∪{j}) + μ(B)]
    for i in range(m):
        for j in range(i+1, m):
            others = [k for k in range(m) if k not in (i, j)]
            max_r = len(others) if k is None else min(len(others), k)
            
            for r in range(max_r + 1):
                for B in itertools.combinations(others, r):
                    B = tuple(sorted(B))
                    Bi = tuple(sorted(B + (i,)))
                    Bj = tuple(sorted(B + (j,)))
                    Bij = tuple(sorted(B + (i, j)))
                    
                    interaction_matrix[i, j] += get_value(Bij) - get_value(Bi) - get_value(Bj) + get_value(B)
                    
            interaction_matrix[i, j] /= norm_interaction
            interaction_matrix[j, i] = interaction_matrix[i, j]  # Ensure symmetry
    
    return {
        'power_indices': power_indices,
        'interaction_matrix': interaction_matrix
    }


# =============================================================================
# Interaction indices  
# Might be usefull for iterating over all coallitions and seing how much interaction matters
# =============================================================================

from math import factorial
from itertools import chain, combinations

def shapley_interaction_index(A, all_coalitions, v, m):
    """
    Compute the Shapley interaction index I^S(A) for subset A.
    
    Args:
        A (tuple): Target coalition (e.g., (0, 1) for interaction between features 0 and 1).
        all_coalitions (list): List of all nonempty coalitions (as tuples).
        v (np.array): Learned game parameters (μ values for each coalition).
        m (int): Total number of features.
    
    Returns:
        float: Shapley interaction index for coalition A.
    """
    A = set(A)
    M_minus_A = set(range(m)) - A
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    I_S = 0.0
    
    # Iterate over all B ⊆ M \ A
    for B in chain.from_iterable(combinations(M_minus_A, r) for r in range(len(M_minus_A)+1)):
        B = set(B)
        # Iterate over all B' ⊆ A
        sum_term = 0.0
        for B_prime in chain.from_iterable(combinations(A, r) for r in range(len(A)+1)):
            B_prime = set(B_prime)
            # Compute B ∪ B'
            coalition = tuple(sorted(B.union(B_prime)))
            if not coalition:  # Skip empty set (μ(∅) = 0)
                continue
            # Get μ(B ∪ B') from v (0 if coalition not in all_coalitions)
            mu_val = v[coalition_to_index.get(coalition, 0)] if coalition in coalition_to_index else 0.0
            # (-1)^{|A| - |B'|}
            sign = (-1) ** (len(A) - len(B_prime))
            sum_term += sign * mu_val
        
        # Compute weight [(m - |B| - |A|)! |B|!] / (m - |A| + 1)!
        weight_numerator = factorial(m - len(B) - len(A)) * factorial(len(B))
        weight_denominator = factorial(m - len(A) + 1)
        weight = weight_numerator / weight_denominator
        
        I_S += weight * sum_term
    
    return I_S

def banzhaf_interaction_index(A, all_coalitions, v, m):
    """
    Compute the Banzhaf interaction index I^B(A) for subset A.
    
    Args:
        A (tuple): Target coalition (e.g., (0, 1)).
        all_coalitions (list): List of all nonempty coalitions (as tuples).
        v (np.array): Learned game parameters (μ values).
        m (int): Total number of features.
    
    Returns:
        float: Banzhaf interaction index for coalition A.
    """
    A = set(A)
    M_minus_A = set(range(m)) - A
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    I_B = 0.0
    
    # Iterate over all B ⊆ M \ A
    for B in chain.from_iterable(combinations(M_minus_A, r) for r in range(len(M_minus_A)+1)):
        B = set(B)
        # Iterate over all B' ⊆ A
        for B_prime in chain.from_iterable(combinations(A, r) for r in range(len(A)+1)):
            B_prime = set(B_prime)
            # Compute B ∪ B'
            coalition = tuple(sorted(B.union(B_prime)))
            if not coalition:  # Skip empty set
                continue
            # Get μ(B ∪ B') from v
            mu_val = v[coalition_to_index.get(coalition, 0)] if coalition in coalition_to_index else 0.0
            # (-1)^{|A| - |B'|}
            sign = (-1) ** (len(A) - len(B_prime))
            I_B += sign * mu_val
    
    # Normalize by 2^{m - |A|}
    I_B /= 2 ** (m - len(A))
    return I_B

def compute_shapley_interaction_indices(v, m, all_coalitions, target_feature_set):
    """
    Compute the general Shapley interaction indices for any subset A of features.
    
    Parameters:
    -----------
    v : array-like
        Capacity/game values for each coalition
    m : int
        Total number of features
    all_coalitions : list of tuples
        List of all coalitions in the model
    target_feature_set : tuple or list
        The set A of features for which to compute the interaction index
        
    Returns:
    --------
    float : Shapley interaction index I^S(A)
    """
    A = set(target_feature_set)
    A_size = len(A)
    M_minus_A = set(range(m)) - A
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    
    interaction_index = 0.0
    
    # Iterate over all B ⊆ M\A
    for r in range(len(M_minus_A) + 1):
        for B in itertools.combinations(M_minus_A, r):
            B = set(B)
            # Weight calculation for this B
            weight = factorial(m - len(B) - A_size) * factorial(len(B)) / factorial(m - A_size + 1)
            
            # Inner sum over all B' ⊆ A
            inner_sum = 0.0
            for r_prime in range(A_size + 1):
                for B_prime in itertools.combinations(A, r_prime):
                    B_prime = set(B_prime)
                    union = tuple(sorted(B.union(B_prime)))
                    if not union:  # Empty set
                        continue
                        
                    # Get the value, or 0 if not found
                    try:
                        mu_val = v[coalition_to_index[union]]
                    except (KeyError, IndexError):
                        mu_val = 0.0
                        
                    # Apply the sign: (-1)^(|A| - |B'|)
                    sign = (-1) ** (A_size - len(B_prime))
                    inner_sum += sign * mu_val
                    
            interaction_index += weight * inner_sum
            
    return interaction_index

def compute_banzhaf_interaction_index(v, m, all_coalitions, A, k=None):
    """
    Compute the general Banzhaf interaction index I^B(A) for any subset A (Equation 12).
    
    I^B(A) = (1/2^(m-|A|)) * ∑_{B⊆M\A} ∑_{B'⊆A} (-1)^(|A|-|B'|) * μ(B∪B')
    
    Parameters:
    -----------
    v : array-like
        The capacity/game values for each coalition
    m : int
        Total number of features
    all_coalitions : list of tuples
        List of all coalitions in the model
    A : tuple or list
        The subset of features for which to compute the interaction index
    k : int or None, optional
        Additivity limit for k-additive models. If None, full model is assumed.
        
    Returns:
    --------
    float : The Banzhaf interaction index I^B(A)
    """
    A = tuple(sorted(A))
    A_set = set(A)
    M_minus_A = set(range(m)) - A_set
    
    # Create a lookup dictionary for coalition values
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    
    # Memoization cache for coalition values
    coalition_values = {}
    
    def get_coalition_value(coalition):
        """Get or compute the value v(coalition) with memoization"""
        coalition = tuple(sorted(coalition))
        if not coalition:
            return 0.0  # Empty set has value 0
            
        # Check cache first
        if coalition in coalition_values:
            return coalition_values[coalition]
            
        # Calculate and cache the value
        if k is None:
            # For full model, direct lookup or 0
            try:
                value = v[coalition_to_index.get(coalition, -1)] if coalition in coalition_to_index else 0.0
            except (KeyError, IndexError):
                value = 0.0
        else:
            # For k-additive model, reconstruct the value
            value = reconstruct_v_S(coalition, v, all_coalitions, k)
            
        coalition_values[coalition] = value
        return value
    
    interaction_index = 0.0
    
    # Iterate over all B ⊆ M\A
    for B_size in range(len(M_minus_A) + 1):
        for B in itertools.combinations(M_minus_A, B_size):
            B = set(B)
            
            # Iterate over all B' ⊆ A
            for B_prime_size in range(len(A_set) + 1):
                for B_prime in itertools.combinations(A_set, B_prime_size):
                    B_prime = set(B_prime)
                    
                    # (-1)^(|A| - |B'|)
                    sign = (-1) ** (len(A_set) - len(B_prime))
                    
                    # Get μ(B ∪ B')
                    coalition = tuple(sorted(B.union(B_prime)))
                    value = get_coalition_value(coalition)
                    
                    interaction_index += sign * value
    
    # Normalize by 2^(m - |A|)
    interaction_index /= (2 ** (m - len(A_set)))
    
    return interaction_index

# =============================================================================
# Choose Default Implementation
# =============================================================================

#ChoquisticRegression = ChoquisticRegression_Composition
ChoquisticRegression = ChoquisticRegression_Inheritance

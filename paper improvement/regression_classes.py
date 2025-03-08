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
            # For 2-additive model, the coefficient vector is arranged as [main effects | interactions]
            coef = self.classifier_.coef_[0]
            nAttr = self.n_features_in_
            marginal_contrib = coef[:nAttr].copy()
            interactions = coef[nAttr:]
            interaction_matrix = np.zeros((nAttr, nAttr))
            counter = 0
            for i in range(nAttr):
                for j in range(i + 1, nAttr):
                    interaction_matrix[i, j] = interactions[counter]
                    interaction_matrix[j, i] = interactions[counter]
                    counter += 1
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
            return phi

        elif self.method == "choquet_2add":
            coef = self.coef_[0]
            nAttr = self.transformer_.n_features_in_
            marginal_contrib = coef[:nAttr].copy()
            interactions = coef[nAttr:]
            interaction_matrix = np.zeros((nAttr, nAttr))
            counter = 0
            for i in range(nAttr):
                for j in range(i + 1, nAttr):
                    interaction_matrix[i, j] = interactions[counter]
                    interaction_matrix[j, i] = interactions[counter]
                    counter += 1
            shapley_vals = coef[:nAttr] + 0.5 * np.sum(interaction_matrix, axis=1)
            return {"marginal": marginal_contrib, "shapley": shapley_vals}
        else:
            raise ValueError("Shapley value computation is only implemented for 'choquet' and 'choquet_2add'.")


# =============================================================================
# Explicit Interpretability Functions 
# =============================================================================



# =============================================================================
# Power indices 
# =============================================================================
def compute_shapely_values(v, m, all_coalitions):
    """
    Compute Shapley values for a given Choquet model.
    Parameters:
      - v: 1D array of learned coefficients (ordered as in all_coalitions)
      - m: number of original features
      - all_coalitions: list of tuples representing all nonempty coalitions
        (as produced by the choquet_matrix transformation)
    Returns:
      - A 1D numpy array of Shapley values.
    """
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


# =============================================================================
# Interaction matrixes 
# =============================================================================


def compute_banzhaf_interaction_matrix(v, m, all_coalitions):
    """
    Compute the Banzhaf interaction matrix given the learned coefficient vector v,
    the number of original features m, and the list of all nonempty coalitions.

    The Banzhaf interaction index for a pair of features (i, j) is computed as:
      I(i,j) = (1 / 2^(m-2)) * sum_{S ⊆ N\{i,j}} [v(S ∪ {i, j}) - v(S ∪ {i}) - v(S ∪ {j}) + v(S)]

    Parameters:
      - v: 1D array of learned coefficients (ordered as in all_coalitions)
      - m: number of original features
      - all_coalitions: list of tuples representing all nonempty coalitions
        (as produced by the choquet_matrix transformation)

    Returns:
      - A symmetric m x m numpy array containing the Banzhaf interaction indices.
    """
    import itertools

    interaction_matrix = np.zeros((m, m))
    norm = 2 ** (m - 2)  # normalization: number of subsets S of N \ {i, j}
    # Loop over each unique pair of features (i, j)
    for i in range(m):
        for j in range(i + 1, m):
            total = 0.0
            # Compute over all subsets S of the remaining features
            others = [k for k in range(m) if k not in (i, j)]
            for r in range(0, len(others) + 1):
                for S in itertools.combinations(others, r):
                    S = tuple(sorted(S))
                    # v(S): if S is empty, assume 0
                    vS = 0.0 if len(S) == 0 else v[all_coalitions.index(S)]
                    # v(S ∪ {i})
                    Si = tuple(sorted(S + (i,)))
                    try:
                        vSi = v[all_coalitions.index(Si)]
                    except ValueError:
                        vSi = 0.0
                    # v(S ∪ {j})
                    Sj = tuple(sorted(S + (j,)))
                    try:
                        vSj = v[all_coalitions.index(Sj)]
                    except ValueError:
                        vSj = 0.0
                    # v(S ∪ {i, j})
                    Sij = tuple(sorted(S + (i, j)))
                    try:
                        vSij = v[all_coalitions.index(Sij)]
                    except ValueError:
                        vSij = 0.0
                    total += vSij - vSi - vSj + vS
            interaction_matrix[i, j] = total / norm
            interaction_matrix[j, i] = total / norm
    return interaction_matrix



def compute_choquet_interaction_matrix(v, m, all_coalitions):
    """
    Compute the pairwise Shapley interaction indices between features.
    
    For each distinct pair (i, j), the interaction index is computed as:
    
      Iₛᵢ,ⱼ = Σ₍B ⊆ M\{i,j}₎ [((m - |B| - 2)! * |B|!)/(m - 1)! * (μ(B ∪ {i,j}) - μ(B ∪ {i}) - μ(B ∪ {j}) + μ(B))]
    
    Parameters:
        v : array-like
            Value function evaluated on each coalition (μ).
        m : int
            Number of features.
        all_coalitions : list of tuples
            List of coalitions (each is a sorted tuple of feature indices).
            Should include the empty set.
            
    Returns:
        interaction_matrix : np.array of shape (m, m)
            A symmetric matrix where entry (i, j) is the Shapley interaction index between features i and j.
    """
    interaction_matrix = np.zeros((m, m))
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    
    for i in range(m):
        for j in range(i+1, m):
            total = 0.0
            # Consider subsets B of the features excluding both i and j.
            others = [k for k in range(m) if k not in (i, j)]
            for r in range(0, len(others)+1):
                for B in itertools.combinations(others, r):
                    B = tuple(sorted(B))
                    # Weight: ((m - |B| - 2)! * |B|! ) / (m - 1)!
                    weight = factorial(m - r - 2) * factorial(r) / factorial(m - 1)
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


# =============================================================================
# Choose Default Implementation
# =============================================================================

ChoquisticRegression = ChoquisticRegression_Composition
#ChoquisticRegression = ChoquisticRegression_Inheritance

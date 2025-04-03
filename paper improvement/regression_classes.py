import numpy as np
import itertools
from itertools import chain, combinations
from math import comb, factorial
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils.extmath import softmax  
import warnings

# =============================================================================
# Utility functions for k-additive models
# =============================================================================
def nParam_kAdd(kAdd,nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr,ii+1)
    return aux_numb

def powerset(iterable,k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes
    powerset([1,2,..., m],m) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m) ... (1, ..., m)
    powerset([1,2,..., m],2) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add+1))



# =============================================================================
# Choquet Transformation Functions
# =============================================================================



def choquet_matrix_mobius(X_orig,kadd):
    
    nSamp, nAttr = X_orig.shape # Number of samples (train) and attributes
    k_add_numb = nParam_kAdd(kadd,nAttr)
    
    data_opt = np.zeros((nSamp,k_add_numb-1))
    
    for i,s in enumerate(powerset(range(nAttr),kadd)):
        s = list(s)

        if len(s) > 0:
            data_opt[:,i-1] = np.min(X_orig.iloc[:,s],axis=1)
            
    return data_opt


def choquet_matrix(X_orig, all_coalitions=None):
    """
    Compute the full Choquet integral transformation matrix.
    
    This implements the Choquet integral formulation:
    C_μ(x) = Σ_{i=1}^n (x_σ(i) - x_σ(i-1)) * v({σ(i), ..., σ(n)})
    where σ is a permutation that orders features in ascending order.
    
    Parameters:
    -----------
    X_orig : array-like of shape (n_samples, n_features)
        Original feature matrix
    all_coalitions : list of tuples, optional
        Pre-computed list of all nonempty coalitions
        
    Returns:
    --------
    tuple : (transformed feature matrix, list of all coalitions)
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

def choquet_k_additive_game(X_orig, k_add=None):

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
    Compute the 2-additive Choquet integral transformation according to equation (23).
    
    In a 2-additive Choquet integral, the formula is:
    f_CI(v, x_i) = ∑_j x_i,j(φ_j^S - (1/2)∑_{j'≠j} I_{j,j'}^S) + ∑_{j≠j'} (x_i,j ∧ x_i,j') I_{j,j'}^S
    
    This function implements the original formulation including the adjustment terms
    as specified in equation (23) of the paper.
    
    Parameters:
    -----------
    X_orig : array-like
        Original feature matrix
        
    Returns:
    --------
    numpy.ndarray : 2-additive Choquet integral basis transformation
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
    Compute the 2-additive multilinear model transformation according to equation (25).
    
    In a 2-additive MLM, the formula is:
    f_ML(v, x_i) = ∑_j x_i,j(φ_j^B - (1/2)∑_{j'≠j} I_{j,j'}^B) + ∑_{j≠j'} x_i,j x_i,j' I_{j,j'}^B
    
    This function implements the original formulation including the adjustment terms
    as specified in equation (25) of the paper.
    
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


# =============================================================================
# ChoquetTransformer Class
# =============================================================================
class ChoquetTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for Choquet or multilinear based feature transformations.
    
    This transformer implements various fuzzy measure-based transformations:
    - Choquet integral 
        - k-additive Choquet integral with game representation
        - k-additive Choquet integral with Möbius representation
        - 2-additive Choquet integral with shapely representation
    - Full multilinear model 
    - 2-additive multilinear model
    
    Parameters
    ----------
    method : str, default="choquet_2add"
        The transformation method. Options:
        - "choquet": General Choquet integral
        - "choquet_2add": 2-additive Choquet integral shapely representation
        - "mlm": Full multilinear model
        - "mlm_2add": 2-additive multilinear model
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
        
        valid_methods = ["choquet", "choquet_2add", "mlm", "mlm_2add"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        if method == "choquet":
            valid_representations = ["game", "mobius"]
            if representation not in valid_representations:
                raise ValueError(f"For method='choquet', representation must be one of {valid_representations}")
        if k_add is not None:
            if not isinstance(k_add, int) or k_add < 1:
                raise ValueError("k_add must be a positive integer")
            # For methods ending with '_2add' k_add is implicit; otherwise, k_add is used.
    
    def fit(self, X, y=None):
        X = check_array(X, ensure_min_features=1)
        self.n_features_in_ = X.shape[1]
        if self.method == "choquet" and self.representation == "game":
            k = self.k_add if self.k_add is not None else self.n_features_in_
            self.all_coalitions_ = list(powerset(range(self.n_features_in_), k))
        return self

    def transform(self, X):
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
        elif self.method == "mlm":
            if self.k_add is not None:
                raise NotImplementedError("k-additive MLM not yet implemented")
            return mlm_matrix(X)
        elif self.method == "mlm_2add":
            return mlm_matrix_2add(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["n_features_in_"])
        dummy = np.zeros((1, self.n_features_in_))
        transformed = self.transform(dummy)
        n_out = transformed.shape[1]
        return np.array([f"F{i}" for i in range(n_out)], dtype=object)


# =============================================================================
# Implementation 1: Composition-based ChoquisticRegression
# =============================================================================
class ChoquisticRegression_Composition(BaseEstimator, ClassifierMixin):
    """
    Choquistic Regression classifier using composition.

    This estimator combines the Choquet integral (or multilinear model) transformation
    with logistic regression. The model can be written as:
    
    P(Y=1|x) = 1/(1 + exp(-w₀ - C_μ(x)))
    
    where C_μ(x) is the Choquet integral (or multilinear model) transformation of x.
    
    For the 2-additive Choquet model:
    C_μ(x) = ∑ᵢ μᵢxᵢ + ∑_{i<j} Iᵢⱼ min(xᵢ, xⱼ)
    
    Parameters
    ----------
    method : str, default="choquet_2add"
        Transformation method to use. Options:
        - "choquet": Choquet integral
        - "choquet_2add": 2-additive Choquet integral shapely representation
        - "mlm": Full multilinear model
        - "mlm_2add": 2-additive multilinear model
    representation : str, default="game"
        For method="choquet", defines the representation to use:
        - "game": Uses game-based representation
        - "mobius": Uses Möbius representation
        Ignored for other methods.
    k_add : int or None, default=None
        Additivity level for k-additive models. Only used when method is "choquet" or "mlm".
    scale_data : bool, default=True
        Whether to scale data to [0,1] range with MinMaxScaler.
    logistic_params : dict or None, default=None
        Parameters passed to LogisticRegression. If None or incomplete, default values are used.
    **kwargs : dict
        Additional parameters passed to LogisticRegression.
    """

    def __init__(self, method="choquet", representation="game", k_add=None, scale_data=True,
                 logistic_params=None, **kwargs):
        # Store ALL parameters as attributes
        self.method = method
        self.representation = representation
        self.k_add = k_add
        self.scale_data = scale_data
        self.logistic_params = logistic_params
        
        # Default parameters for LogisticRegression
        self.default_logistic_params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'newton-cg',
            'max_iter': 100,
            'tol': 1e-4,
            'class_weight': None,
            'random_state': None,
            'fit_intercept': True
        }
        
        # Additional parameters from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, X, y):
        X = check_array(X)
        self.original_n_features_in_ = X.shape[1] 
        self.n_features_in_ = X.shape[1]
        
        # Initialize scaler if needed
        if self.scale_data:
            self.scaler_ = MinMaxScaler().fit(X)
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        # Create and fit transformer
        self.transformer_ = ChoquetTransformer(
            method=self.method,
            representation=self.representation,  
            k_add=self.k_add
        )
        self.transformer_.fit(X_scaled)
        X_transformed = self.transformer_.transform(X_scaled)
        
        self.transformed_n_features_in_ = X_transformed.shape[1]
        
        # Merge default and user-provided parameters
        params = self.default_logistic_params.copy()
        if self.logistic_params is not None:
            params.update(self.logistic_params)
            
        self.classifier_ = LogisticRegression(**params)
        self.classifier_.fit(X_transformed, y)
        
        # Copy attributes from classifier for convenience
        self.coef_ = self.classifier_.coef_
        self.intercept_ = self.classifier_.intercept_
        self.classes_ = self.classifier_.classes_
        if hasattr(self.classifier_, 'n_iter_'):
            self.n_iter_ = self.classifier_.n_iter_
        
        return self

    def predict(self, X):
        """Predict class labels for X."""
        X = check_array(X)
        X_transformed = self._transform_data(X)
        return self.classifier_.predict(X_transformed)

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        X = check_array(X)
        X_transformed = self._transform_data(X)
        return self.classifier_.predict_proba(X_transformed)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        X = check_array(X)
        X_transformed = self._transform_data(X)
        return self.classifier_.score(X_transformed, y)

    def decision_function(self, X):
        """Return distance of each sample to the decision boundary."""
        X = check_array(X)
        X_transformed = self._transform_data(X)
        return self.classifier_.decision_function(X_transformed)
    
    def _transform_data(self, X):
        """Transform input data through scaling and Choquet transformation."""
        check_is_fitted(self, ["transformer_"])
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but ChoquisticRegression is expecting "
                            f"{self.n_features_in_} features.")
        
        # Apply scaling if configured
        if self.scale_data:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        return self.transformer_.transform(X_scaled)
    
    def get_model_capacity(self):
        """
        Extract the capacity measure from the model as a mapping from transformed
        feature names (as generated by the transformer) to coefficient values.
        """
        check_is_fitted(self, ["coef_"])
        capacity = {}
        # Derive feature names via the transformer.
        feature_names = self.transformer_.get_feature_names_out()
        coef = self.coef_[0]
        for i, name in enumerate(feature_names):
            capacity[name] = coef[i]
        return capacity

    def compute_shapley_values(self):
        check_is_fitted(self, ["coef_"])
        
        if self.method == "choquet":
            m = self.n_features_in_
            all_coalitions = self.transformer_.all_coalitions_
            v = self.coef_[0]
            
            phi = compute_shapley_values(v, m, all_coalitions)
            return phi

        elif self.method == "choquet_2add":
            nAttr = self.n_features_in_
            coef = self.coef_[0]    

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
            
        elif self.method.startswith("mlm"):
            raise NotImplementedError("Shapley value computation for MLM methods not yet implemented")


    def predict_log_proba(self, X):
        X = check_array(X)
        X_transformed = self._transform_data(X)
        return self.classifier_.predict_log_proba(X_transformed)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
            
        if not hasattr(self, "classifier_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}' "
                                "(classifier_ not initialized, call fit() first)")
                                
        return getattr(self.classifier_, name)

# =============================================================================
# Implementation 2: Inheritance-based ChoquisticRegression
# =============================================================================

class ChoquisticRegression_Inheritance(LogisticRegression):
    """
    Choquistic Regression classifier using inheritance from LogisticRegression.

    This estimator combines the Choquet integral (or multilinear model) transformation
    with logistic regression through inheritance. The model can be written as:
    
    P(Y=1|x) = 1/(1 + exp(-w₀ - C_μ(x)))
    
    where C_μ(x) is the Choquet integral (or multilinear model) transformation of x.
    
    Parameters
    ----------
    method : str, default="choquet"
        Transformation method to use. Options:
        - "choquet": Choquet integral
        - "choquet_2add": 2-additive Choquet integral shapely representation
        - "mlm": Full multilinear model
        - "mlm_2add": 2-additive multilinear model
    representation : str, default="game"
        For method="choquet", defines the representation to use:
        - "game": Uses game-based representation
        - "mobius": Uses Möbius representation
        Ignored for other methods.
    k_add : int or None, default=None
        Additivity level for k-additive models. Only used when method is "choquet" or "mlm".
    scale_data : bool, default=True
        Whether to scale data to [0,1] range with MinMaxScaler.
    logistic_params : dict or None, default=None
        Parameters passed to the parent LogisticRegression constructor. If None or incomplete, 
        default values are used.
    **kwargs : dict
        Additional parameters (not used directly but maintained for API compatibility).
    """
    def __init__(self, method="choquet", representation="game", k_add=None, scale_data=True,
                 logistic_params=None, **kwargs):
        # Default parameters for LogisticRegression
        default_params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'newton-cg',
            'max_iter': 100,
            'tol': 1e-4,
            'class_weight': None,
            'random_state': None,
            'fit_intercept': True
        }
        
        # Merge default parameters with user-provided ones
        params = default_params.copy()
        if logistic_params is not None:
            params.update(logistic_params)
            
        # Call parent constructor with merged parameters
        super().__init__(**params)  
        
        # Store our specific parameters
        self.method = method
        self.representation = representation
        self.k_add = k_add
        self.scale_data = scale_data
        self.logistic_params = logistic_params
        
    def fit(self, X, y):
        X = check_array(X)
        self.original_n_features_in_ = X.shape[1] 
        self.n_features_in_ = X.shape[1]
        
        if self.scale_data:
            self.scaler_ = MinMaxScaler().fit(X)
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        self.transformer_ = ChoquetTransformer(
            method=self.method,
            representation=self.representation,  
            k_add=self.k_add
        )
        self.transformer_.fit(X_scaled)
        X_transformed = self.transformer_.transform(X_scaled)
        
        self.transformed_n_features_in_ = X_transformed.shape[1]
        
        return super().fit(X_transformed, y)

    def _transform(self, X):
        check_is_fitted(self, ["original_n_features_in_", "transformer_"])
        X = check_array(X)
        if X.shape[1] != self.original_n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features but expected {self.original_n_features_in_}")
        X_scaled = self.scaler_.transform(X) if self.scale_data else X
        return self.transformer_.transform(X_scaled)

    def predict(self, X):
        return super().predict(self._transform(X))

    def predict_proba(self, X):
        return super().predict_proba(self._transform(X))

    def decision_function(self, X):
        return super().decision_function(self._transform(X))

    def predict_log_proba(self, X):
        return super().predict_log_proba(self._transform(X))

    def score(self, X, y, sample_weight=None):
        return super().score(self._transform(X), y, sample_weight=sample_weight)

    def get_model_capacity(self):
        check_is_fitted(self, ["coef_"])
        capacity = {}
        feature_names = self.transformer_.get_feature_names_out()
        coef = self.coef_[0]
        for i, name in enumerate(feature_names):
            capacity[name] = coef[i]
        return capacity

    def compute_shapley_values(self):
        check_is_fitted(self, ["coef_"])
        if self.method == "choquet":
            n = self.n_features_in_
            # Depending on your implementation, you might need to use self.transformer_.all_coalitions_
            raise NotImplementedError("Shapley value computation for choquet (game) not yet implemented in inheritance version")
        elif self.method == "choquet_2add":
            nAttr = self.n_features_in_
            coef = self.coef_[0]
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
            raise NotImplementedError("Shapley value computation for MLM methods not yet implemented")



# =============================================================================
# Utility functions 
# =============================================================================


def compute_capacity_value(S, v, all_coalitions, k=None):
    """
    Compute or reconstruct the capacity value v(S) for coalition S.
    
    For a full model (k=None), directly looks up v(S) if available.
    For a k-additive model, reconstructs v(S) from its Möbius representation:
    v(S) = ∑_{T⊆S, 0<|T|≤min(|S|,k)} m(T)
    
    Parameters:
    -----------
    S : tuple
        A coalition (sorted tuple of feature indices)
    v : array-like
        Array of capacity/game values (v[0] is assumed to be empty set)
    all_coalitions : list of tuples
        List of all coalitions in the model (NOT including empty set)
    k : int or None, optional
        Additivity limit for k-additive models. If None, attempts direct lookup.
        
    Returns:
    --------
    float : The capacity value v(S)
    """
    if len(S) == 0:
        return 0.0
        
    # Ensure S is a sorted tuple for consistent lookup
    S = tuple(sorted(S))
        
    # If S is in all_coalitions, use direct lookup with +1 offset for empty set
    if S in all_coalitions:
        try:
            idx = all_coalitions.index(S)
            return v[idx + 1] 
        except (ValueError, IndexError):
            pass 
    
    # For k-additive models, reconstruct from smaller coalitions
    if k is not None:
        if len(S) > k:
            total = 0.0
            for r in range(1, k + 1):
                for T in itertools.combinations(S, r):
                    T = tuple(sorted(T))
                    if T in all_coalitions:
                        idx = all_coalitions.index(T)
                        total += v[idx + 1]
            return total
    return 0.0



# =============================================================================
# Explicit Interpretability Functions 
# =============================================================================


# =============================================================================
# Power indices 
# =============================================================================
def compute_shapley_values(v, m, all_coalitions, k=None):
    """
    Compute Shapley power indices for all features.
    
    φ_j^S = ∑_{B⊆M\\{j}} [(m-|B|-1)!|B|!/m!] * [v(B∪{j}) - v(B)]
    
    Parameters:
    -----------
    v : array-like
        Capacity/game values for each coalition (with v[0] representing empty set)
    m : int
        Number of features
    all_coalitions : list of tuples
        List of all coalitions in the model (not including empty set)
    k : int or None, optional
        Additivity limit for k-additive models. If None, full model is assumed.
        
    Returns:
    --------
    numpy.ndarray : Shapley power indices for each feature
    """
    # Preprocess coalition list consistently - use sorted tuples
    processed_coalitions = [tuple(sorted(coal)) for coal in all_coalitions]
    coalition_to_index = {coal: idx for idx, coal in enumerate(processed_coalitions)}
    coalition_values = {}  # Memoization cache
    
    phi = np.zeros(m)
    
    def get_value(coalition):
        if not coalition:
            return 0.0 
        
        if coalition in coalition_values:
            return coalition_values[coalition]
            
        if k is None:
            try:
                value = v[coalition_to_index.get(coalition, -1) + 1] if coalition in coalition_to_index else 0.0
            except (KeyError, IndexError):
                value = 0.0
        else:
            value = compute_capacity_value(coalition, v, all_coalitions, k)
            
        coalition_values[coalition] = value
        return value
    
    for j in range(m):
        others = [i for i in range(m) if i != j]
        
        # Process all subsets of M\{j}
        for r in range(len(others) + 1):
            for B in itertools.combinations(others, r):
                # Calculate marginal contribution
                B_tuple = tuple(sorted(B))
                Bj_tuple = tuple(sorted(B + (j,)))
                
                vB = get_value(B_tuple)
                vBj = get_value(Bj_tuple)
                
                # Calculate weight: (m-|B|-1)!|B|! / m!
                weight = factorial(m - r - 1) * factorial(r) / factorial(m)
                phi[j] += weight * (vBj - vB)
                
    return phi


def compute_banzhaf_power_indices(v, m, all_coalitions, k=None):
    """
    Compute Banzhaf power indices for all features (Equation 13).
    
    φ_j^B = (1/2^(m-1)) * ∑_{B⊆M\\{j}} [v(B∪{j}) - v(B)]
    
    Parameters:
    -----------
    v : array-like
        Capacity/game values for each coalition
    m : int
        Number of features
    all_coalitions : list of tuples
        List of all coalitions in the model
    k : int or None, optional
        Additivity limit for k-additive models. If None, full model is assumed.
        
    Returns:
    --------
    numpy.ndarray : Banzhaf power indices for each feature
    """
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    coalition_values = {} 
    
    phi_b = np.zeros(m)
    norm = 2 ** (m - 1)
    
    def get_value(coalition):
        """Get coalition value with memoization"""
        if not coalition:
            return 0.0
        if coalition in coalition_values:
            return coalition_values[coalition]
        
        if k is None or len(coalition) <= k:
            try:
                if coalition in coalition_to_index:
                    idx = coalition_to_index[coalition]
                    value = v[idx + 1] 
                else:
                    value = 0.0
            except (KeyError, IndexError) as e:
                print(f"Warning: Index error for coalition {coalition}: {e}")
                value = 0.0
        else:
            value = compute_capacity_value(coalition, v, all_coalitions, k)
        coalition_values[coalition] = value
        return value
    
    for j in range(m):
        others = [i for i in range(m) if i != j]
        
        max_r = len(others)
        if k is not None:
            max_r = min(max_r, k)
            
        # Iterate through all subsets B ⊆ M\{j}
        for r in range(max_r + 1):
            for B in itertools.combinations(others, r):
                B = tuple(sorted(B))
                Bj = tuple(sorted(B + (j,)))
                phi_b[j] += get_value(Bj) - get_value(B)
                
    return phi_b / norm


def compute_power_indices(v, m, all_coalitions, model_type="banzhaf", k=None):
    if model_type.lower() in ["banzhaf", "mlm"]:
        return compute_banzhaf_power_indices(v, m, all_coalitions, k)
    elif model_type.lower() in ["shapley", "choquet"]:
        return compute_shapley_values(v, m, all_coalitions, k)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'banzhaf' or 'shapley'.")

# =============================================================================
# Interaction matrices 
# =============================================================================

def compute_banzhaf_interaction_matrix(v, m, all_coalitions, k=None):
    """
    Compute the Banzhaf interaction indices for all pairs of features (Equation 14).
    
    I_{j,j'}^B = (1/2^(m-2)) * ∑_{B⊆M\\{j,j'}} [v(B∪{j,j'}) - v(B∪{j}) - v(B∪{j'}) + v(B)]
    
    Parameters:
    -----------
    v : array-like
        Capacity/game values for each coalition
    m : int
        Number of features
    all_coalitions : list of tuples
        List of all coalitions in the model
    k : int or None, optional
        Additivity limit for k-additive models. If None, full model is assumed.
        
    Returns:
    --------
    numpy.ndarray : A symmetric m×m matrix of Banzhaf interaction indices
    """
    interaction_matrix = np.zeros((m, m))
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    coalition_values = {}
    
    def get_value(coalition):
        if not coalition:
            return 0.0
        if coalition in coalition_values:
            return coalition_values[coalition]
        
        if k is None:
            try:
                if coalition in coalition_to_index:
                    idx = coalition_to_index[coalition] 
                    value = v[idx + 1]
                else:
                    value = 0.0
            except (KeyError, IndexError) as e:
                print(f"Warning: Index error for coalition {coalition}: {e}")
                value = 0.0
        else:
            value = compute_capacity_value(coalition, v, all_coalitions, k)
        coalition_values[coalition] = value
        return value
    
    # Normalization factor: 1/2^(m-2)
    norm = 2 ** (m - 2)
    
    # For every distinct pair of features (i,j)
    for i in range(m):
        for j in range(i+1, m):
            total = 0.0
            others = [feat for feat in range(m) if feat not in (i, j)]
            
            # Determine the range of subset sizes to iterate over
            max_r = len(others)
            if k is not None:
                max_r = min(max_r, k)
                
            for r in range(max_r + 1):
                for B in itertools.combinations(others, r):
                    B = tuple(sorted(B))
                    
                    # Get values for the four required coalitions
                    vB = get_value(B)
                    vBi = get_value(tuple(sorted(B + (i,))))
                    vBj = get_value(tuple(sorted(B + (j,))))
                    vBij = get_value(tuple(sorted(B + (i, j))))
                    
                    total += vBij - vBi - vBj + vB
                    
            # Normalize and store the result
            interaction_matrix[i, j] = total / norm
            interaction_matrix[j, i] = interaction_matrix[i, j]
            
    return interaction_matrix


def compute_choquet_interaction_matrix(v, m, all_coalitions, k=None):
    """
    Compute the Shapley interaction indices for all pairs of features.
    
    I_{j,j'}^S = ∑_{B⊆M\\{j,j'}} [(m-|B|-2)!|B|!/(m-1)!] * [v(B∪{j,j'}) - v(B∪{j}) - v(B∪{j'}) + v(B)]
    
    Parameters:
    -----------
    v : array-like
        Capacity/game values for each coalition (including 0 for empty set at index 0)
    m : int
        Number of features
    all_coalitions : list of tuples
        List of all coalitions in the model (not including empty set)
    k : int or None, optional
        Additivity limit for k-additive models. If None, full model is assumed.
        
    Returns:
    --------
    numpy.ndarray : A symmetric m×m matrix of Shapley interaction indices
    """
    interaction_matrix = np.zeros((m, m))
    
    processed_coalitions = [tuple(sorted(coal)) for coal in all_coalitions]
    coalition_to_index = {coal: idx for idx, coal in enumerate(processed_coalitions)}
    
    def get_value(coalition):
        if not coalition:
            return 0.0
            
        sorted_coal = tuple(sorted(coalition))
        if sorted_coal in coalition_to_index:
            return v[coalition_to_index[sorted_coal] + 1]
        
        return compute_capacity_value(sorted_coal, v, processed_coalitions, k)
    
    for i in range(m):
        for j in range(i+1, m):
            total = 0.0
            others = [feat for feat in range(m) if feat not in (i, j)]
            
            # Process ALL subsets of M\{i,j}
            for r in range(len(others) + 1):
                for B in itertools.combinations(others, r):
                    weight = factorial(m - r - 2) * factorial(r) / factorial(m - 1)

                    vB = get_value(tuple(sorted(B)))
                    vBi = get_value(tuple(sorted(B + (i,))))
                    vBj = get_value(tuple(sorted(B + (j,))))
                    vBij = get_value(tuple(sorted(B + (i, j))))
                    
                    total += weight * (vBij - vBi - vBj + vB)
            
            interaction_matrix[i, j] = total
            interaction_matrix[j, i] = total
            
    return interaction_matrix


def compute_model_interactions(v, m, all_coalitions, model_type="mlm", k=None):
    """
    A unified function to compute both power indices and interaction matrices.
    
    Parameters:
    -----------
    v : array-like
        The game/capacity values for each coalition
    m : int
        Number of features
    all_coalitions : list of tuples
        List of all coalitions in the model
    model_type : str, default="mlm"
        Type of interaction indices to compute: "mlm" or "shapley"
    k : int or None
        Additivity limit for k-additive models. If None, full model is assumed.
        
    Returns:
    --------
    dict : Contains 'power_indices' (feature importance values) and 'interaction_matrix'
    """
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    
    coal_values = {}
    
    def get_value(coalition):
        if not coalition:
            return 0.0
            
        if coalition in coal_values:
            return coal_values[coalition]
            
        if k is None:
            try:
                if coalition in coalition_to_index:
                    idx = coalition_to_index[coalition]
                    value = v[idx + 1] 
                else:
                    value = 0.0
            except (KeyError, IndexError) as e:
                print(f"Warning: Index error for coalition {coalition}: {e}")
                value = 0.0
        else:
            value = compute_capacity_value(coalition, v, all_coalitions, k)
            
        coal_values[coalition] = value
        return value
    
    power_indices = np.zeros(m)
    interaction_matrix = np.zeros((m, m))

    # Compute power indices
    if model_type == "mlm":
        # Banzhaf power indices: φ_j^B = (1/2^(m-1)) * ∑_{B⊆M\{j}} [v(B∪{j}) - v(B)]
        norm_power = 2 ** (m - 1)
        
        for j in range(m):
            others = [i for i in range(m) if i != j]
            max_r = len(others)
            if k is not None:
                max_r = min(max_r, k)
                
            for r in range(max_r + 1):
                for B in itertools.combinations(others, r):
                    B = tuple(sorted(B))
                    Bj = tuple(sorted(B + (j,)))
                    power_indices[j] += get_value(Bj) - get_value(B)
        
        power_indices /= norm_power
        
    elif model_type == "choquet":  
        # Shapley power indices: φ_j^S = ∑_{B⊆M\{j}} [(m-|B|-1)!|B|!/m!] * [v(B∪{j}) - v(B)]
        for j in range(m):
            others = [i for i in range(m) if i != j]
            max_r = len(others)
            if k is not None:
                max_r = min(max_r, k)
                
            for r in range(max_r + 1):
                for B in itertools.combinations(others, r):
                    B = tuple(sorted(B))
                    weight = factorial(m - r - 1) * factorial(r) / factorial(m)
                    power_indices[j] += weight * (get_value(tuple(sorted(B + (j,)))) - get_value(B))
    
    # Compute interaction matrices
    if model_type == "mlm":
        # Banzhaf interaction: I_{i,j}^B = (1/2^(m-2)) * ∑_{B⊆M\{i,j}} [v(B∪{i,j}) - v(B∪{i}) - v(B∪{j}) + v(B)]
        norm_interaction = 2 ** (m - 2)
        
        for i in range(m):
            for j in range(i+1, m):
                others = [k for k in range(m) if k not in (i, j)]
                max_r = len(others)
                if k is not None:
                    max_r = min(max_r, k)
                
                for r in range(max_r + 1):
                    for B in itertools.combinations(others, r):
                        B = tuple(sorted(B))
                        Bi = tuple(sorted(B + (i,)))
                        Bj = tuple(sorted(B + (j,)))
                        Bij = tuple(sorted(B + (i, j)))

                        interaction_matrix[i, j] += get_value(Bij) - get_value(Bi) - get_value(Bj) + get_value(B)
                
                interaction_matrix[i, j] /= norm_interaction
                interaction_matrix[j, i] = interaction_matrix[i, j]  # Ensure symmetry
                
    elif model_type == "choquet":  
        # Shapley interaction I_{i,j}^S = ∑_{B⊆M\{i,j}} [(m-|B|-2)!|B|!/(m-1)!] * [v(B∪{i,j}) - v(B∪{i}) - v(B∪{j}) + v(B)]
        for i in range(m):
            for j in range(i+1, m):
                others = [k for k in range(m) if k not in (i, j)]
                max_r = len(others)
                if k is not None:
                    max_r = min(max_r, k)
                
                for r in range(max_r + 1):
                    for B in itertools.combinations(others, r):
                        B = tuple(sorted(B))
                        weight = factorial(m - r - 2) * factorial(r) / factorial(m - 1)
                        
                        Bi = tuple(sorted(B + (i,)))
                        Bj = tuple(sorted(B + (j,)))
                        Bij = tuple(sorted(B + (i, j)))
                        
                        interaction_matrix[i, j] += weight * (get_value(Bij) - get_value(Bi) - get_value(Bj) + get_value(B))
                
                interaction_matrix[j, i] = interaction_matrix[i, j]
    
    return {
        'power_indices': power_indices,
        'interaction_matrix': interaction_matrix
    }

# =============================================================================
# Interaction indices for arbitrary coalitions
# =============================================================================

def compute_shapley_interaction_index(v, m, all_coalitions, A, k=None):
    """
    Compute the Shapley interaction index I^S(A) for any subset A of features (Equation 5).
    
    I^S(A) = sum_{B ⊆ M\\A} [(m-|B|-|A|)!|B|!/(m-|A|+1)!] * [sum_{B' ⊆ A} (-1)^(|A|-|B'|) * μ(B ∪ B')]
    
    Parameters:
    -----------
    v : array-like
        Capacity/game values for each coalition
    m : int
        Total number of features
    all_coalitions : list of tuples
        List of all coalitions in the model
    A : tuple or list
        The set of features for which to compute the interaction index
    k : int or None, optional
        Additivity limit for k-additive models. If None, full model is assumed.
        
    Returns:
    --------
    float : Shapley interaction index I^S(A)
    """
    A = set(A)
    A_size = len(A)
    M_minus_A = set(range(m)) - A
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}

    coalition_values = {}
    
    def get_coalition_value(coalition):
        """Get or compute the value v(coalition) with memoization"""
        if not coalition:
            return 0.0
            
        if coalition in coalition_values:
            return coalition_values[coalition]
            
        if k is None:
            try:
                if coalition in coalition_to_index:
                    idx = coalition_to_index[coalition]
                    value = v[idx + 1] 
                else:
                    value = 0.0
            except (KeyError, IndexError) as e:
                print(f"Warning: Index error for coalition {coalition}: {e}")
                value = 0.0
        else:
            value = compute_capacity_value(coalition, v, all_coalitions, k)
            
        coalition_values[coalition] = value
        return value
    
    interaction_index = 0.0
    
    # Iterate over all B ⊆ M\A
    for r in range(len(M_minus_A) + 1):
        for B in itertools.combinations(M_minus_A, r):
            B = set(B)
            # Weight calculation: (m-|B|-|A|)!|B|! / (m-|A|+1)!
            weight = factorial(m - len(B) - A_size) * factorial(len(B)) / factorial(m - A_size + 1)
            
            # Inner sum over all B' ⊆ A
            inner_sum = 0.0
            for r_prime in range(A_size + 1):
                for B_prime in itertools.combinations(A, r_prime):
                    B_prime = set(B_prime)
                    coalition = tuple(sorted(B.union(B_prime)))
                    
                    mu_val = get_coalition_value(coalition)

                    sign = (-1) ** (A_size - len(B_prime))
                    inner_sum += sign * mu_val
                    
            interaction_index += weight * inner_sum
            
    return interaction_index


def compute_banzhaf_interaction_index(v, m, all_coalitions, A, k=None):
    """
    Compute the Banzhaf interaction index I^B(A) for any subset A of features (Equation 12).
    
    I^B(A) = (1/2^(m-|A|)) * sum_{B⊆M\\A} sum_{B'⊆A} (-1)^(|A|-|B'|) * μ(B∪B')
    
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
    
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}

    coalition_values = {}
    
    def get_coalition_value(coalition):
        """Get or compute the value v(coalition) with memoization"""
        coalition = tuple(sorted(coalition))
        if not coalition:
            return 0.0 
            
        if coalition in coalition_values:
            return coalition_values[coalition]
            
        if k is None:
            try:
                if coalition in coalition_to_index:
                    idx = coalition_to_index[coalition]
                    value = v[idx + 1]
                else:
                    value = 0.0
            except (KeyError, IndexError) as e:
                print(f"Warning: Index error for coalition {coalition}: {e}")
                value = 0.0
        else:
            value = compute_capacity_value(coalition, v, all_coalitions, k)
            
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

                    sign = (-1) ** (len(A_set) - len(B_prime))
                    
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

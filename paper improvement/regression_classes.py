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
def nParam_kAdd(kAdd, nAttr):
    """
    Return the number of parameters in a k-additive model for nAttr attributes.
    
    In a k-additive model, we only store coefficients for interactions involving 
    up to k features. The total parameter count includes:
    - The empty set (intercept): 1
    - All subsets of size 1 to k: Sum(C(nAttr, r)) for r=1 to k
    
    Parameters:
    -----------
    kAdd : int
        The additivity level of the model
    nAttr : int
        Number of attributes/features
        
    Returns:
    --------
    int : Total number of parameters in the model
    """
    total = 1  # intercept (or empty set)
    for r in range(1, kAdd + 1):
        total += comb(nAttr, r)
    return total


def powerset(iterable, k_add):
    """
    Return the powerset (all subsets up to size k_add) of the given iterable.
    Includes the empty set.
    
    This function generates all possible combinations of elements from the input
    iterable, from size 0 (empty set) up to size k_add.
    
    Parameters:
    -----------
    iterable : iterable
        The input collection of elements
    k_add : int
        Maximum size of subsets to include
        
    Returns:
    --------
    iterable : All subsets of the input up to size k_add
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(0, k_add + 1))


# =============================================================================
# Choquet Transformation Functions
# =============================================================================



def choquet_matrix_mobius(X_orig, kadd):
    """
    Create a feature matrix based on the Möbius representation for k-additive models.
    
    For each subset of features up to size k, computes the minimum value among 
    those features as a new derived feature.
    
    Parameters:
    -----------
    X_orig : DataFrame
        Original feature matrix
    kadd : int
        Additivity level
        
    Returns:
    --------
    numpy.ndarray : Transformed feature matrix
    """
    nSamp, nAttr = X_orig.shape
    # Only count non-empty subsets (exclude the empty set)
    k_add_numb = nParam_kAdd(kadd, nAttr) - 1
    data_opt = np.zeros((nSamp, k_add_numb))
    
    idx = 0
    # Skip empty set by filtering for non-empty subsets
    for s in [subset for subset in powerset(range(nAttr), kadd) if subset]:
        s = list(s)
        data_opt[:, idx] = np.min(X_orig.iloc[:, s], axis=1)
        idx += 1
    
    return data_opt


def choquet_matrix(X_orig, all_coalitions=None):
    """
    Compute the full Choquet integral transformation matrix.
    
    This implements the Choquet integral formulation:
    C_μ(x) = Σ_{i=1}^n (x_σ(i) - x_σ(i-1)) * μ({σ(i), ..., σ(n)})
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


def choquet_matrix_2add(X_orig):
    """
    Compute the 2-additive Choquet integral transformation.
    
    In a 2-additive Choquet model, the capacity μ only has non-zero values
    for singletons and pairs. The Choquet integral simplifies to:
    C_μ(x) = Σ_i μ({i})*x_i + Σ_{i<j} I({i,j})*min(x_i, x_j)
    
    Parameters:
    -----------
    X_orig : array-like
        Original feature matrix
        
    Returns:
    --------
    numpy.ndarray : Transformed feature matrix with:
      - The original features (singletons)
      - For each pair (i,j), the minimum value min(x_i, x_j)
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
    # MLM generates basis functions for all possible non-empty subsets
    subsets = list(chain.from_iterable(combinations(range(nAttr), r) for r in range(1, nAttr + 1)))
    data_opt = np.zeros((nSamp, len(subsets)))
    
    # Assume X_orig is already properly scaled to [0,1] by caller
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
    
    In a 2-additive MLM, we consider:
    - Original features (singletons)
    - Simple pairwise products
    
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
    n_singletons = nAttr
    n_pairs = comb(nAttr, 2)
    data_opt = np.zeros((nSamp, n_singletons + n_pairs))
    
    # Use original features for singletons
    data_opt[:, :n_singletons] = X_orig
    
    # Calculate pairwise terms as simple products
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
    
    This transformer implements various fuzzy measure-based transformations:
    - Full Choquet integral (exponential complexity)
    - 2-additive Choquet integral (quadratic complexity)
    - Full multilinear model (exponential complexity)
    - 2-additive multilinear model (quadratic complexity)
    
    For the full Choquet integral, the transformation computes:
    C_μ(x) = Σ_{i=1}^n (x_σ(i) - x_σ(i-1)) * μ({σ(i), ..., σ(n)})
    where σ orders the features in ascending order.
    
    For the 2-additive Choquet, only singleton and pairwise terms are used:
    C_μ(x) = Σ_i μ({i})*x_i + Σ_{i<j} I({i,j})*min(x_i, x_j)
    
    For the multilinear model (MLM), the transformation computes basis functions:
    MLM(x) = Σ_{T⊆N} m(T) * Π_{i∈T} x_i * Π_{j∉T} (1-x_j)
    
    For MLM methods, input features should be scaled to [0,1] by the caller.

    Parameters
    ----------
    method : str, default="choquet_2add"
        The transformation method. Options:
        - "choquet": Full Choquet integral
        - "choquet_2add": 2-additive Choquet integral
        - "mlm": Full multilinear model
        - "mlm_2add": 2-additive multilinear model
    k_add : int or None, default=None
        Additivity level for k-additive models. If not None and method is 
        "choquet" or "mlm", restricts to k-additive model. Ignored for 
        methods ending with "_2add".
    """

    def __init__(self, method="choquet_2add", k_add=None):
        self.method = method
        self.k_add = k_add
        
        # Validate method
        valid_methods = ["choquet", "choquet_2add", "mlm", "mlm_2add"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        # Validate k_add if specified
        if k_add is not None:
            if not isinstance(k_add, int) or k_add < 1:
                raise ValueError("k_add must be a positive integer")
            if method.endswith("_2add"):
                # Instead of warning, silently ignore since this is a common pattern
                pass
            elif k_add < 1:
                raise ValueError(f"k_add must be at least 1, got {k_add}")
        
        
    def fit(self, X, y=None):
        """
        Fit the transformer by pre-computing necessary structures.
        
        For 'choquet' method, pre-computes all necessary coalitions.
        For other methods, simply stores the feature count.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X = check_array(X, ensure_min_features=1)
        
        # Store number of features for all methods
        self.n_features_in_ = X.shape[1]
        
        # For Choquet, pre-compute coalition structure
        if self.method == "choquet":
            if self.k_add is None:
                # Full Choquet - all coalitions
                _, all_coalitions = choquet_matrix(X)
            else:
                # k-additive Choquet - coalitions up to size k
                all_coalitions = []
                for r in range(1, min(self.k_add, self.n_features_in_) + 1):
                    all_coalitions.extend(list(itertools.combinations(range(self.n_features_in_), r)))
            self.all_coalitions_ = all_coalitions
        
        return self

    def transform(self, X):
        """
        Transform X by applying the selected fuzzy measure-based transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform. For MLM methods, should already be scaled to [0,1].
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        # Check if transformer is fitted
        check_is_fitted(self, ["n_features_in_"])
        
        # Validate input dimensions
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but ChoquetTransformer is expecting "
                            f"{self.n_features_in_} features.")
        
        # Apply the selected transformation
        if self.method == "choquet":
            if not hasattr(self, "all_coalitions_"):
                raise AttributeError("Transformer not properly fitted. Call fit() before transform().")
            X_trans, _ = choquet_matrix(X, all_coalitions=self.all_coalitions_)
            return X_trans
            
        elif self.method == "choquet_2add":
            return choquet_matrix_2add(X)
            
        elif self.method == "mlm":
            if self.k_add is not None:
                raise NotImplementedError("k-additive MLM not yet implemented")
            return mlm_matrix(X)
            
        elif self.method == "mlm_2add":
            return mlm_matrix_2add(X)
            
        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Unknown method: {self.method}")

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. If None, generic names will be generated.
            
        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, ["n_features_in_"])
        
        # Generate default input feature names if none provided
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        else:
            if len(input_features) != self.n_features_in_:
                raise ValueError(
                    f"input_features has length {len(input_features)} but "
                    f"transformer expects {self.n_features_in_} features.")
        
        # Generate output feature names based on method
        if self.method == "choquet":
            # For full Choquet, feature names correspond to coalitions
            feature_names = []
            for coalition in self.all_coalitions_:
                feature_names.append("_AND_".join(input_features[i] for i in coalition))
            return np.array(feature_names, dtype=object)
            
        elif self.method == "choquet_2add":
            feature_names = []
            # Singletons
            feature_names.extend(input_features)
            # Pairs - min(x_i, x_j)
            for i in range(self.n_features_in_):
                for j in range(i+1, self.n_features_in_):
                    feature_names.append(f"min({input_features[i]}, {input_features[j]})")
            return np.array(feature_names, dtype=object)
            
        elif self.method == "mlm":
            # Full MLM feature names
            if not hasattr(self, "all_coalitions_"):
                # Generate all coalitions for MLM
                all_coalitions = []
                for r in range(1, self.n_features_in_ + 1):
                    all_coalitions.extend(list(itertools.combinations(range(self.n_features_in_), r)))
                
            feature_names = []
            for coalition in all_coalitions:
                terms = []
                for i in range(self.n_features_in_):
                    if i in coalition:
                        terms.append(input_features[i])
                    else:
                        terms.append(f"(1-{input_features[i]})")
                feature_names.append("*".join(terms))
            return np.array(feature_names, dtype=object)
            
        elif self.method == "mlm_2add":
            feature_names = []
            # Singletons
            feature_names.extend(input_features)
            # Pairs - x_i * x_j
            for i in range(self.n_features_in_):
                for j in range(i+1, self.n_features_in_):
                    feature_names.append(f"{input_features[i]}*{input_features[j]}")
            return np.array(feature_names, dtype=object)


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
        - "choquet": Full Choquet integral
        - "choquet_2add": 2-additive Choquet integral
        - "mlm": Full multilinear model
        - "mlm_2add": 2-additive multilinear model
    k_add : int or None, default=None
        Additivity level for k-additive models. Only used when method is "choquet" or "mlm".
    scale_data : bool, default=True
        Whether to scale data to [0,1] range with MinMaxScaler.
    **kwargs : dict
        Additional parameters passed to LogisticRegression.
    """

    def __init__(self, method="choquet_2add", k_add=None, scale_data=True, **kwargs):
        self.method = method
        self.k_add = k_add
        self.scale_data = scale_data
        
        # Set default parameters if not provided
        default_params = {
            "penalty": None,
            "max_iter": 1000,
            "solver": "newton-cg",
            "random_state": 0
        }
        
        # Use provided kwargs, with defaults as fallbacks
        self.logistic_params = default_params.copy()
        self.logistic_params.update(kwargs)
        
        # Validate method
        valid_methods = ["choquet", "choquet_2add", "mlm", "mlm_2add"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def fit(self, X, y):
        """
        Fit the Choquistic Regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        # Store original feature count to match the inheritance version
        self.original_n_features_in_ = X.shape[1] 
        self.n_features_in_ = X.shape[1]
        
        # Initialize scaler if needed
        if self.scale_data:
            self.scaler_ = MinMaxScaler().fit(X)
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        # Create transformer with correct parameters (no scale_data parameter)
        self.transformer_ = ChoquetTransformer(
            method=self.method, 
            k_add=self.k_add
        )
        X_transformed = self.transformer_.fit_transform(X_scaled)
        
        # Store the transformed feature count to match the inheritance version
        self.transformed_n_features_in_ = X_transformed.shape[1]
        
        self.classifier_ = LogisticRegression(**self.logistic_params)
        self.classifier_.fit(X_transformed, y)
        
        # Copy key attributes from the classifier for compatibility
        self.coef_ = self.classifier_.coef_
        self.intercept_ = self.classifier_.intercept_
        self.classes_ = self.classifier_.classes_
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
        Extract the capacity/fuzzy measure from the model.
        
        Returns
        -------
        dict : A mapping from coalitions to capacity values
        """
        check_is_fitted(self, ["coef_"])
        capacity = {}
        
        if self.method == "choquet":
            all_coalitions = self.transformer_.all_coalitions_
            coef = self.coef_[0]
            for i, coalition in enumerate(all_coalitions):
                capacity[coalition] = coef[i]
                
        elif self.method == "choquet_2add":
            n_attr = self.n_features_in_
            coef = self.coef_[0]
            
            # Singletons
            for i in range(n_attr):
                capacity[(i,)] = coef[i]
                
            # Add interaction terms (corresponding to pairs)
            idx = n_attr
            for i in range(n_attr):
                for j in range(i + 1, n_attr):
                    pair = (i, j)
                    # In 2-additive model, we store interaction values separately
                    # but the actual capacity is computed differently
                    capacity[pair] = coef[idx]
                    idx += 1
                    
        return capacity

    def compute_shapley_values(self):
        """
        Compute Shapley values for the model.
        
        The Shapley value for feature j is the average marginal contribution of j
        across all possible feature coalitions:
        
        φ_j = ∑_{S⊆M\\{j}} [(|S|!(m-|S|-1)!)/m!] * [v(S∪{j}) - v(S)]
        
        For 2-additive Choquet models, this simplifies to:
        φ_j = v({j}) + 0.5 * ∑_{i≠j} I({i,j})
        
        Returns
        -------
        numpy.ndarray or dict : Shapley values for each feature
        """
        check_is_fitted(self, ["coef_"])
        
        if self.method == "choquet":
            m = self.n_features_in_
            all_coalitions = self.transformer_.all_coalitions_
            v = self.coef_[0]
            
            # Use our utility function from power indices section
            phi = compute_shapley_values(v, m, all_coalitions)
            return phi

        elif self.method == "choquet_2add":
            # For 2-additive models, Shapley values have a simplified formula:
            # φj = μ({j}) + 0.5 * Σi≠j I({i,j})
            nAttr = self.n_features_in_
            coef = self.coef_[0]
            
            # First nAttr coefficients are singleton capacities μ({j})
            marginal_contrib = coef[:nAttr].copy()  
            
            # The rest are interaction terms I({i,j})
            interactions = coef[nAttr:]             

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
            
        elif self.method.startswith("mlm"):
            # For multilinear models, use Banzhaf power indices
            raise NotImplementedError("Shapley value computation for MLM methods not yet implemented")
            
        else:
            raise ValueError("Shapley value computation is only implemented for 'choquet' and 'choquet_2add' methods.")

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            The predicted log-probabilities of the sample for each class.
        """
        X = check_array(X)
        X_transformed = self._transform_data(X)
        return self.classifier_.predict_log_proba(X_transformed)

    # Forward missing methods from LogisticRegression to the internal classifier
    def __getattr__(self, name):
        """
        Forward any attributes/methods not found in this class to the underlying classifier.
        This ensures both implementations remain fully interchangeable.
        """
        if name.startswith("__") and name.endswith("__"):
            # Don't forward special Python methods
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
    Choquistic Regression classifier using inheritance.
    
    This class extends LogisticRegression and adds the Choquet integral or
    multilinear model transformation as a preprocessing step. The model represents:
    
    P(Y=1|x) = 1/(1 + exp(-w₀ - C_μ(x)))
    
    where C_μ(x) is the Choquet integral (or multilinear model) transformation of x.
    
    Parameters
    ----------
    method : str, default="choquet_2add"
        Transformation method to use. Options:
        - "choquet": Full Choquet integral
        - "choquet_2add": 2-additive Choquet integral
        - "mlm": Full multilinear model
        - "mlm_2add": 2-additive multilinear model
    k_add : int or None, default=None
        Additivity level for k-additive models. Only used when method is "choquet" or "mlm".
    scale_data : bool, default=True
        Whether to scale data to [0,1] range with MinMaxScaler.
    **kwargs : dict
        Additional parameters passed to LogisticRegression.
    """

    def __init__(self, method="choquet_2add", k_add=None, scale_data=True, **kwargs):
        self.method = method
        self.k_add = k_add
        self.scale_data = scale_data
        
        # Set default parameters if not provided
        default_params = {
            "penalty": None,
            "max_iter": 1000,
            "solver": "newton-cg",
            "random_state": 0
        }
        
        for key, value in default_params.items():
            if key not in kwargs:
                kwargs[key] = value
        
        # Initialize LogisticRegression with given parameters
        super().__init__(**kwargs)
        
        # Create transformer (no scale_data parameter)
        self.transformer_ = ChoquetTransformer(
            method=self.method, 
            k_add=self.k_add
        )
        
        # Validate method
        valid_methods = ["choquet", "choquet_2add", "mlm", "mlm_2add"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def fit(self, X, y, **fit_params):
        """
        Fit the Choquistic Regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        **fit_params : dict
            Additional parameters passed to LogisticRegression.fit().
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        # Store original feature count separately to avoid conflicts with LogisticRegression
        self.original_n_features_in_ = X.shape[1]
        
        # Scale input data if configured
        if self.scale_data:
            self.scaler_ = MinMaxScaler().fit(X)
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        # Transform data using the Choquet transformer
        X_transformed = self.transformer_.fit_transform(X_scaled)
        
        # Store the transformed feature count
        self.transformed_n_features_in_ = X_transformed.shape[1]
                
        # Let LogisticRegression set n_features_in_ to the transformed size
        # and fit the logistic regression model with transformed data
        return super().fit(X_transformed, y, **fit_params)

    def _transform(self, X):
        """Transform input data through scaling and Choquet transformation."""
        check_is_fitted(self, ["original_n_features_in_", "transformed_n_features_in_"])
        X = check_array(X)
        
        # Check if the features match the original or transformed dimensions
        if X.shape[1] == self.original_n_features_in_:
            # Original features - need transformation
            if self.scale_data:
                X_scaled = self.scaler_.transform(X)
            else:
                X_scaled = X
                
            return self.transformer_.transform(X_scaled)
        elif X.shape[1] == self.transformed_n_features_in_:
            # Already transformed features
            return X
        else:
            # Incorrect dimensions
            raise ValueError(f"X has {X.shape[1]} features, but ChoquisticRegression was fitted with "
                           f"{self.original_n_features_in_} original features "
                           f"(transformed into {self.transformed_n_features_in_} features).")
            
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        X_transformed = self._transform(X)
        return super().predict(X_transformed)

    def predict_proba(self, X):
        """
        Probability estimates for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        check_is_fitted(self)
        X_transformed = self._transform(X)
        return super().predict_proba(X_transformed)

    def predict_log_proba(self, X):
        """
        Predict log probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            The log of prediction probabilities.
        """
        X_transformed = self._transform(X)
        return super().predict_log_proba(X_transformed)

    def decision_function(self, X):
        """
        Decision function for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        T : ndarray of shape (n_samples,) or (n_samples, n_classes)
            The decision function values.
        """
        check_is_fitted(self)
        X_transformed = self._transform(X)
        return super().decision_function(X_transformed)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X_transformed = self._transform(X)
        return super().score(X_transformed, y, sample_weight=sample_weight)

    def get_model_capacity(self):
        """
        Extract the capacity/fuzzy measure from the model.
        
        Returns
        -------
        dict : A mapping from coalitions to capacity values
        """
        check_is_fitted(self, ["coef_"])
        capacity = {}
        
        if self.method == "choquet":
            all_coalitions = self.transformer_.all_coalitions_
            coef = self.coef_[0]
            for i, coalition in enumerate(all_coalitions):
                capacity[coalition] = coef[i]
                
        elif self.method == "choquet_2add":
            n_attr = self.original_n_features_in_
            coef = self.coef_[0]
            
            # Singletons
            for i in range(n_attr):
                capacity[(i,)] = coef[i]
                
            # Add interaction terms (corresponding to pairs)
            idx = n_attr
            for i in range(n_attr):
                for j in range(i + 1, n_attr):
                    pair = (i, j)
                    capacity[pair] = coef[idx]
                    idx += 1
                    
        return capacity

    def compute_shapley_values(self):
        """
        Compute Shapley values for the model.
        
        The Shapley value for feature j is the average marginal contribution of j
        across all possible feature coalitions:
        
        φ_j = ∑_{S⊆M\\{j}} [(|S|!(m-|S|-1)!)/m!] * [v(S∪{j}) - v(S)]
        
        For 2-additive Choquet models, this simplifies to:
        φ_j = v({j}) + 0.5 * ∑_{i≠j} I({i,j})
        
        Returns
        -------
        numpy.ndarray or dict : Shapley values for each feature
        """
        check_is_fitted(self, ["coef_"])
        
        if self.method == "choquet":
            m = self.original_n_features_in_
            all_coalitions = self.transformer_.all_coalitions_
            v = self.coef_[0]
            
            # Use the utility function from power indices section
            phi = compute_shapley_values(v, m, all_coalitions)
            return phi

        elif self.method == "choquet_2add":
            # For 2-additive models, Shapley values have a simplified formula:
            # φj = μ({j}) + 0.5 * Σi≠j I({i,j})
            nAttr = self.original_n_features_in_
            coef = self.coef_[0]
            
            marginal_contrib = coef[:nAttr].copy()  # singleton capacities: μ({j})
            interactions = coef[nAttr:]             # interaction indices: I({i,j})

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
        
        elif self.method.startswith("mlm"):
            # For multilinear models, use Banzhaf power indices
            raise NotImplementedError("Shapley value computation for MLM methods not yet implemented")
            
        else:
            raise ValueError("Shapley value computation is only implemented for 'choquet' and 'choquet_2add' methods.")

# =============================================================================
# Utility functions 
# =============================================================================

def compute_mobius_transform(T, v, all_coalitions, k=None):
    """
    Compute the Möbius transform m(T) for coalition T.
    
    The Möbius transform maps between a capacity function v and its Möbius representation:
    - For a singleton T = {i}: m({i}) = v({i})
    - For larger T: m(T) = v(T) - ∑_{S⊂T, S≠∅} m(S)
    
    In a k-additive model, m(T) = 0 for all |T| > k.
    
    Parameters:
    -----------
    T : tuple
        A coalition (sorted tuple of feature indices)
    v : array-like
        Array of capacity/game values for each coalition
    all_coalitions : list of tuples
        List of all coalitions in the model
    k : int or None, optional
        Additivity limit. If specified and |T| > k, returns 0.
        
    Returns:
    --------
    float : The Möbius transform value for coalition T
    """
    # In k-additive models, m(T) = 0 for |T| > k
    if k is not None and len(T) > k:
        return 0.0
    
    # Base case: empty set
    if len(T) == 0:
        return 0.0
        
    # Base case: singleton
    if len(T) == 1:
        try:
            return v[all_coalitions.index(T)]
        except ValueError:
            return 0.0
    
    # General case: compute m(T) recursively
    try:
        v_T = v[all_coalitions.index(T)]
    except ValueError:
        v_T = 0.0
        
    # Subtract the Möbius values of all proper non-empty subsets
    total = 0.0
    for r in range(1, len(T)):
        for S in itertools.combinations(T, r):
            S = tuple(sorted(S))
            total += compute_mobius_transform(S, v, all_coalitions, k)
            
    return v_T - total


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
    # Base case: empty set has value 0
    if len(S) == 0:
        return 0.0
        
    # Ensure S is a sorted tuple for consistent lookup
    S = tuple(sorted(S))
        
    # If S is in all_coalitions, use direct lookup with +1 offset for empty set
    if S in all_coalitions:
        try:
            idx = all_coalitions.index(S)
            return v[idx + 1]  # +1 because v[0] is for empty set
        except (ValueError, IndexError):
            pass  # Fall back to reconstruction
    
    # For k-additive models, reconstruct from smaller coalitions
    if k is not None:
        # For coalitions larger than k, reconstruct using Möbius coefficients
        if len(S) > k:
            total = 0.0
            # Sum over all subsets T ⊆ S with |T| ≤ k
            for r in range(1, k + 1):
                for T in itertools.combinations(S, r):
                    T = tuple(sorted(T))
                    if T in all_coalitions:
                        idx = all_coalitions.index(T)
                        # Use direct coefficient for existing coalitions
                        total += v[idx + 1]  # +1 for empty set
            return total
            
    # If we can't compute a value, return 0.0
    return 0.0


def reconstruct_interaction_value(S, v, all_coalitions, k=None, memo=None):
    """
    Efficiently reconstruct interaction values for any coalition S.
    
    This is a memoized version that avoids redundant calculations.
    
    Parameters:
    -----------
    S : tuple
        A sorted tuple representing the coalition
    v : array-like
        Capacity/game values for each coalition
    all_coalitions : list of tuples
        List of all coalitions in the model
    k : int or None, optional
        Additivity limit for k-additive models
    memo : dict, optional
        Memoization dictionary to cache results
        
    Returns:
    --------
    float : The reconstructed interaction value
    """
    # Initialize memoization dictionary if not provided
    if memo is None:
        memo = {}
        
    # Check if already computed
    if S in memo:
        return memo[S]
        
    # Direct lookup if possible
    if S in all_coalitions:
        try:
            result = v[all_coalitions.index(S)]
            memo[S] = result
            return result
        except (ValueError, IndexError):
            pass
    
    # For k-additive models, reconstruct using Möbius basis
    if k is not None:
        total = 0.0
        max_r = min(len(S), k)
        
        for r in range(1, max_r + 1):
            for T in itertools.combinations(S, r):
                T = tuple(sorted(T))
                # Use memoization for nested calls
                if T not in memo:
                    if T in all_coalitions:
                        memo[T] = compute_mobius_transform(T, v, all_coalitions, k)
                    else:
                        memo[T] = 0.0
                total += memo[T]
                
        memo[S] = total
        return total
        
    # If k is None and S is not in all_coalitions
    memo[S] = 0.0
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
        """Get coalition value with memoization"""
        if not coalition:
            return 0.0  # Empty set
        
        # Check cache first
        if coalition in coalition_values:
            return coalition_values[coalition]
            
        # Calculate and cache the value
        if k is None:
            try:
                # Add +1 to match the interaction matrix function
                value = v[coalition_to_index.get(coalition, -1) + 1] if coalition in coalition_to_index else 0.0
            except (KeyError, IndexError):
                value = 0.0
        else:
            value = compute_capacity_value(coalition, v, all_coalitions, k)
            
        coalition_values[coalition] = value
        return value
    
    # Compute Shapley values for each feature
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
    coalition_values = {}  # Memoization cache
    
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
                # Simplified and fixed lookup pattern
                if coalition in coalition_to_index:
                    idx = coalition_to_index[coalition]
                    value = v[idx + 1]  # +1 for empty set at v[0]
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
        
        # Determine maximum subset size for iteration
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
    """
    A unified function to compute power indices for either Banzhaf or Shapley methods.
    
    Parameters:
    -----------
    v : array-like
        Game/capacity values for each coalition
    m : int
        Number of features
    all_coalitions : list of tuples
        List of all coalitions in the model
    model_type : str, default="banzhaf"
        Type of power indices to compute: "banzhaf" or "shapley"
    k : int or None, optional
        Additivity limit for k-additive models. If None, full model is assumed.
        
    Returns:
    --------
    numpy.ndarray : Power indices for each feature
    """
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
    coalition_values = {}  # Memoization cache
    
    def get_value(coalition):
        """Get coalition value with memoization"""
        if not coalition:
            return 0.0
        if coalition in coalition_values:
            return coalition_values[coalition]
        
        if k is None:
            try:
                # Simplified and fixed lookup pattern
                if coalition in coalition_to_index:
                    idx = coalition_to_index[coalition] 
                    value = v[idx + 1]  # +1 for empty set at v[0]
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
            # Define the set of "other" features (excluding i and j)
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
                    
                    # Apply the interaction formula
                    total += vBij - vBi - vBj + vB
                    
            # Normalize and store the result (ensure matrix symmetry)
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
    
    # Preprocess coalition list consistently - use sorted tuples
    processed_coalitions = [tuple(sorted(coal)) for coal in all_coalitions]
    coalition_to_index = {coal: idx for idx, coal in enumerate(processed_coalitions)}
    
    # In compute_choquet_interaction_matrix
    def get_value(coalition):
        """Get coalition value with consistent processing"""
        if not coalition:
            return 0.0
            
        sorted_coal = tuple(sorted(coalition))
        if sorted_coal in coalition_to_index:
            # +1 because v[0] is empty set
            return v[coalition_to_index[sorted_coal] + 1]
        
        # For k-additive models or missing coalitions
        return compute_capacity_value(sorted_coal, v, processed_coalitions, k)
    
    # Loop over all distinct pairs
    for i in range(m):
        for j in range(i+1, m):
            total = 0.0
            others = [feat for feat in range(m) if feat not in (i, j)]
            
            # Process ALL subsets of M\{i,j} - correct for any k-additive model
            for r in range(len(others) + 1):
                for B in itertools.combinations(others, r):
                    # Calculate weight according to Shapley interaction formula
                    weight = factorial(m - r - 2) * factorial(r) / factorial(m - 1)
                    
                    # Critical fix: ensure consistent handling of coalition values
                    B_tuple = tuple(sorted(B))
                    Bi_tuple = tuple(sorted(B + (i,)))
                    Bj_tuple = tuple(sorted(B + (j,)))
                    Bij_tuple = tuple(sorted(B + (i, j)))
                    
                    vB = get_value(B_tuple)
                    vBi = get_value(Bi_tuple)
                    vBj = get_value(Bj_tuple)
                    vBij = get_value(Bij_tuple)
                    
                    # Accumulate weighted interaction
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
    # Create lookup dictionary for fast coalition value retrieval
    coalition_to_index = {coal: idx for idx, coal in enumerate(all_coalitions)}
    
    # Memoization cache for coalition values
    coal_values = {}
    
    def get_value(coalition):
        """Get coalition value with memoization"""
        if not coalition:
            return 0.0
            
        if coalition in coal_values:
            return coal_values[coalition]
            
        if k is None:
            try:
                # Simplified and fixed lookup pattern
                if coalition in coalition_to_index:
                    idx = coalition_to_index[coalition]
                    value = v[idx + 1]  # +1 for empty set at v[0]
                else:
                    value = 0.0
            except (KeyError, IndexError) as e:
                print(f"Warning: Index error for coalition {coalition}: {e}")
                value = 0.0
        else:
            value = compute_capacity_value(coalition, v, all_coalitions, k)
            
        coal_values[coalition] = value
        return value
    
    # Initialize output structures
    power_indices = np.zeros(m)
    interaction_matrix = np.zeros((m, m))
    
    # 1. Compute power indices (feature importance)
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
        
    elif model_type == "choquet":  # Shapley power indices
        # φ_j^S = ∑_{B⊆M\{j}} [(m-|B|-1)!|B|!/m!] * [v(B∪{j}) - v(B)]
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
    
    # 2. Compute interaction matrix
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
                
    elif model_type == "choquet":  # Shapley interaction
        # I_{i,j}^S = ∑_{B⊆M\{i,j}} [(m-|B|-2)!|B|!/(m-1)!] * [v(B∪{i,j}) - v(B∪{i}) - v(B∪{j}) + v(B)]
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
                
                interaction_matrix[j, i] = interaction_matrix[i, j]  # Ensure symmetry
    
    # Remove all scaling factors - return raw mathematical values
    
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
    
    # Memoization cache for coalition values
    coalition_values = {}
    
    def get_coalition_value(coalition):
        """Get or compute the value v(coalition) with memoization"""
        if not coalition:  # Empty set
            return 0.0
            
        # Check cache first
        if coalition in coalition_values:
            return coalition_values[coalition]
            
        # Calculate and cache the value
        if k is None:
            try:
                # Simplified and fixed lookup pattern
                if coalition in coalition_to_index:
                    idx = coalition_to_index[coalition]
                    value = v[idx + 1]  # +1 for empty set at v[0]
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
                    
                    # Get μ(B ∪ B'), with support for k-additive models
                    mu_val = get_coalition_value(coalition)
                    
                    # Apply the sign: (-1) ** (A_size - len(B_prime))
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
            try:
                # Simplified and fixed lookup pattern
                if coalition in coalition_to_index:
                    idx = coalition_to_index[coalition]
                    value = v[idx + 1]  # +1 for empty set at v[0]
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

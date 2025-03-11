import numpy as np
import itertools
from scipy.special import comb
from itertools import chain, combinations

def powerset(iterable, k_add):
    """
    Return the powerset (all subsets up to size k_add) of the given iterable.
    Includes the empty set.
    
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


def nParam_kAdd(k_add, nAttr):
    """
    Calculate the number of parameters in a k-additive model for nAttr attributes.
    
    Parameters:
    -----------
    k_add : int
        The additivity level of the model
    nAttr : int
        Number of attributes/features
        
    Returns:
    --------
    int : Total number of parameters in the model (including empty set)
    """
    total = 1  # empty set
    for r in range(1, k_add + 1):
        total += comb(nAttr, r)
    return int(total)


def choquet_matrix(X, k_add=None):
    """
    Unified implementation of the Choquet integral transformation matrix 
    supporting k-additivity (Equation 22 in the paper).
    
    The function implements the Choquet integral:
    f_CI(v, x_i) = Σ_{j=1}^{m} (x_{i,(j)} - x_{i,(j-1)}) * v({(j), ..., (m)})
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Original feature matrix
    k_add : int, optional
        Maximum size of coalitions to consider. If None, full model is used.
        
    Returns:
    --------
    tuple : (transformed feature matrix, list of all coalitions)
    """
    X = np.array(X)
    nSamp, nAttr = X.shape
    
    # Determine the maximum coalition size to consider
    max_size = k_add if k_add is not None else nAttr
    
    # Generate all coalitions up to max_size, excluding empty set
    all_coalitions = []
    for r in range(1, min(max_size, nAttr) + 1):
        all_coalitions.extend(list(itertools.combinations(range(nAttr), r)))
    
    # Create mapping from coalition to index
    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    data_opt = np.zeros((nSamp, len(all_coalitions)))
    
    for i in range(nSamp):
        # Sort features in ascending order as required by Choquet integral definition
        order = np.argsort(X[i])
        sorted_vals = np.sort(X[i])
        prev = 0.0  # x_{i,(0)} is defined as 0
        
        for j in range(nAttr):
            # Calculate the difference (x_{i,(j)} - x_{i,(j-1)})
            diff = sorted_vals[j] - prev
            prev = sorted_vals[j]
            
            # The coalition {(j), ..., (m)} in the formula
            full_coalition = tuple(sorted(order[j:]))
            
            # Handle k-additive limitation:
            if len(full_coalition) <= max_size:
                # Direct assignment for coalitions within size limit
                idx = coalition_to_index.get(full_coalition)
                if idx is not None:
                    data_opt[i, idx] = diff
            else:
                # For coalitions larger than k, distribute the value to k-sized subcoalitions
                current_feature = order[j]
                remaining_features = order[j+1:]
                
                if max_size > 0:  # Ensure we're not dealing with trivial case
                    if len(remaining_features) >= max_size - 1:
                        # Form max_size-sized coalitions containing the current feature
                        for subset in itertools.combinations(remaining_features, max_size - 1):
                            coalition = tuple(sorted((current_feature,) + subset))
                            idx = coalition_to_index.get(coalition)
                            if idx is not None:
                                # Distribute with equal weights
                                weight = 1.0 / comb(len(remaining_features), max_size - 1)
                                data_opt[i, idx] += diff * weight
                    else:
                        # Use the largest possible coalition
                        coalition = full_coalition
                        idx = coalition_to_index.get(coalition)
                        if idx is not None:
                            data_opt[i, idx] += diff
    
    return data_opt, all_coalitions


def choquet_shapley(X, k_add=2):
    """
    Implementation of the 2-additive Choquet integral using the Shapley 
    and interaction indices formulation (Equation 23 in the paper).

    In a 2-additive Choquet integral, the formula is:
    f_CI(v, x_i) = ∑_j x_i,j(φ_j^S - (1/2)∑_{j'≠j} I_{j,j'}^S) + ∑_{j≠j'} (x_i,j ∧ x_i,j') I_{j,j'}^S
    
    Parameters:
    -----------
    X : array-like
        Original feature matrix
    k_add : int, default=2
        Additivity level (currently only k=2 is supported for this implementation)
        
    Returns:
    --------
    numpy.ndarray : Transformed feature matrix using Shapley/interaction formulation
    """
    if k_add != 2:
        raise ValueError("The Shapley/interaction formulation only supports k_add=2")
        
    X = np.array(X)
    nSamp, nAttr = X.shape
    
    # Count parameters in 2-additive model (including empty set)
    k_add_numb = int(nParam_kAdd(k_add, nAttr))
    
    # Create coalition indicator matrix
    coalit = np.zeros((k_add_numb, nAttr))
    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1
    
    # Initialize output matrix
    data_opt = np.zeros((nSamp, k_add_numb))
    
    # Direct feature terms
    for i in range(nAttr):
        data_opt[:, i+1] += X[:, i]
    
    # Pairwise minimum terms
    for i in range(nAttr):
        for i2 in range(i+1, nAttr):
            # Find the column where both features appear (the pair interaction)
            pair_col = (coalit[:, [i, i2]]==1).all(axis=1)
            # Set this column to the minimum of the two features
            data_opt[:, pair_col] = np.minimum(X[:, i], X[:, i2]).reshape(nSamp, 1)
    
    # Adjustment terms
    for i in range(nAttr):
        for ii in range(nAttr+1, len(coalit)):
            if coalit[ii, i] == 1:
                data_opt[:, ii] += (-1/2) * X[:, i]
    
    # Exclude the empty set from output
    return data_opt[:, 1:]


def convert_representations(X, from_standard=True, k_add=2):
    """
    Convert between standard Choquet representation (Eq. 22) and 
    Shapley/interaction representation (Eq. 23).
    
    Parameters:
    -----------
    X : array-like
        Input data matrix
    from_standard : bool, default=True
        If True, converts from standard (Eq. 22) to Shapley (Eq. 23)
        If False, converts from Shapley to standard
    k_add : int, default=2
        The additivity level (currently only k=2 is supported)
        
    Returns:
    --------
    tuple : (converted matrix, transformation matrix, coalitions)
    """
    if k_add != 2:
        raise ValueError("Currently conversion is only supported for k_add=2")
        
    # Get both representations
    standard_matrix, coalitions = choquet_matrix(X, k_add=k_add)
    shapley_matrix = choquet_shapley(X, k_add=k_add)
    
    if from_standard:
        # Convert from standard to Shapley representation
        M = np.linalg.lstsq(standard_matrix, shapley_matrix, rcond=None)[0]
        converted = standard_matrix @ M
        return converted, M, coalitions
    else:
        # Convert from Shapley to standard representation
        M = np.linalg.lstsq(shapley_matrix, standard_matrix, rcond=None)[0]
        converted = shapley_matrix @ M
        return converted, M, coalitions


def convert_coefficients(coeffs, transform_matrix, from_standard=True):
    """
    Convert model coefficients between standard and Shapley representations.
    
    Parameters:
    -----------
    coeffs : array-like
        Coefficients in the source representation
    transform_matrix : array-like
        Transformation matrix between representations
    from_standard : bool, default=True
        If True, converts from standard to Shapley coefficients
        If False, converts from Shapley to standard coefficients
        
    Returns:
    --------
    array : Converted coefficients
    """
    if from_standard:
        # If standard_matrix @ transform_matrix = shapley_matrix
        # Then for predictions to match:
        # standard_matrix @ standard_coeffs = shapley_matrix @ shapley_coeffs
        # This means: standard_coeffs = transform_matrix @ shapley_coeffs
        return transform_matrix @ coeffs
    else:
        # Going the other way requires a pseudo-inverse
        # shapley_coeffs = np.linalg.pinv(transform_matrix) @ standard_coeffs
        # Or use least squares for a more stable solution:
        return np.linalg.lstsq(transform_matrix, coeffs, rcond=None)[0]


class ChoquetIntegral:
    """
    A class for creating and using Choquet integral transformations.
    
    This class provides methods for both the standard formulation (Eq. 22)
    and the Shapley/interaction formulation (Eq. 23).
    
    Parameters:
    -----------
    k_add : int, optional
        The additivity level. If None, full model is used.
    representation : str, default='standard'
        The representation to use ('standard' or 'shapley').
    """
    
    def __init__(self, k_add=None, representation='standard'):
        self.k_add = k_add
        
        if representation not in ['standard', 'shapley']:
            raise ValueError("representation must be 'standard' or 'shapley'")
        
        if representation == 'shapley' and (k_add != 2 and k_add is not None):
            raise ValueError("Shapley representation only supports k_add=2")
            
        self.representation = representation
        self.coalitions_ = None
        self.transform_matrix_ = None
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : ignored
            
        Returns:
        --------
        self
        """
        X = np.array(X)
        
        if self.representation == 'standard':
            _, self.coalitions_ = choquet_matrix(X, k_add=self.k_add)
        elif self.representation == 'shapley':
            # Shapley just computes the transformation
            if self.k_add is None:
                self.k_add = 2  # Default for Shapley representation
            self.shapley_output_ = choquet_shapley(X, k_add=self.k_add)
            _, self.coalitions_ = choquet_matrix(X, k_add=self.k_add)
            
            # Compute transformation matrix for later use
            std_matrix, _ = choquet_matrix(X, k_add=self.k_add)
            self.transform_matrix_ = np.linalg.lstsq(std_matrix, self.shapley_output_, rcond=None)[0]
            
        return self
    
    def transform(self, X):
        """
        Transform X using the Choquet integral.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_coalitions)
            Transformed output
        """
        X = np.array(X)
        
        if self.representation == 'standard':
            transformed, _ = choquet_matrix(X, k_add=self.k_add)
            return transformed
        elif self.representation == 'shapley':
            return choquet_shapley(X, k_add=self.k_add)
        
    def convert_coefficients(self, coeffs, to_representation='standard'):
        """
        Convert coefficients between representations.
        
        Parameters:
        -----------
        coeffs : array-like
            Coefficients to convert
        to_representation : str, default='standard'
            Target representation ('standard' or 'shapley')
            
        Returns:
        --------
        array : Converted coefficients
        """
        if self.transform_matrix_ is None:
            raise ValueError("Must fit the transformer before converting coefficients")
            
        if to_representation not in ['standard', 'shapley']:
            raise ValueError("to_representation must be 'standard' or 'shapley'")
            
        from_standard = self.representation == 'standard'
        to_standard = to_representation == 'standard'
        
        if from_standard == to_standard:
            return coeffs  # No conversion needed
        
        if from_standard:
            # Convert standard to Shapley
            return convert_coefficients(coeffs, self.transform_matrix_, from_standard=True)
        else:
            # Convert Shapley to standard
            return convert_coefficients(coeffs, self.transform_matrix_, from_standard=False)


# Example usage
def example():
    """Example usage of the Choquet integral implementations."""
    import matplotlib.pyplot as plt
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(100, 3)
    
    # 1. Basic transformation
    print("===== Basic transformation =====")
    X_choquet, coalitions = choquet_matrix(X, k_add=2)
    print(f"Original data shape: {X.shape}")
    print(f"Transformed data shape: {X_choquet.shape}")
    print(f"Coalitions: {coalitions}")
    
    # 2. Compare standard and Shapley representations
    print("\n===== Comparing representations =====")
    X_standard, _ = choquet_matrix(X, k_add=2)
    X_shapley = choquet_shapley(X, k_add=2)
    print(f"Standard representation shape: {X_standard.shape}")
    print(f"Shapley representation shape: {X_shapley.shape}")
    
    # 3. Convert between representations
    print("\n===== Converting between representations =====")
    X_converted, transform_matrix, _ = convert_representations(X, from_standard=True)
    conversion_error = np.linalg.norm(X_converted - X_shapley) / np.linalg.norm(X_shapley)
    print(f"Conversion error: {conversion_error:.10f}")
    
    # 4. Use the class-based interface
    print("\n===== Using the ChoquetIntegral class =====")
    standard_model = ChoquetIntegral(k_add=2, representation='standard')
    shapley_model = ChoquetIntegral(k_add=2, representation='shapley')
    
    standard_model.fit(X)
    shapley_model.fit(X)
    
    X_std = standard_model.transform(X)
    X_shap = shapley_model.transform(X)
    
    print(f"Standard class output shape: {X_std.shape}")
    print(f"Shapley class output shape: {X_shap.shape}")
    
    # 5. Visualize the transformed data
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_std[:, 0], X_std[:, 1], alpha=0.6)
    plt.title("Standard Representation")
    plt.xlabel("Coalition (0)")
    plt.ylabel("Coalition (1)")
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_shap[:, 0], X_shap[:, 1], alpha=0.6)
    plt.title("Shapley Representation")
    plt.xlabel("Shapley value for feature 0")
    plt.ylabel("Shapley value for feature 1")
    
    plt.tight_layout()
    plt.savefig("choquet_representations.png")
    plt.close()
    
    print("\nVisualization saved to choquet_representations.png")


if __name__ == "__main__":
    example()

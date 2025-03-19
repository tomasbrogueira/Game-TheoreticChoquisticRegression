import numpy as np
import itertools
from scipy.special import comb
from itertools import chain, combinations

def nParam_kAdd(nAttr, kAdd):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr,ii+1)
    return aux_numb

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
    return chain.from_iterable(combinations(s, r) for r in range(k_add + 1))


def choquet_matrix(X_orig, k_add=None):
    """
    Unified implementation of the Choquet integral transformation matrix supporting k-additivity.
    
    This function strictly implements the Choquet integral as per Equation (22) in the paper:
    f_CI(v, x_i) = Σ_{j=1}^{m} (x_{i,(j)} - x_{i,(j-1)}) * v({(j), ..., (m)})
    
    For k-additive models, it limits the coalition sizes to k and distributes values from larger
    coalitions appropriately among smaller ones.
    
    Parameters:
      X_orig : array-like, shape (nSamp, nAttr)
          The original data.
      k_add : int
          The maximum coalition size (k–additivity).
          
    Returns:
      data_opt : np.ndarray, shape (nSamp, n_coalitions)
    """
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape
    
    max_size = k_add if k_add is not None else nAttr
    

    all_coalitions = []
    for r in range(1, min(max_size, nAttr) + 1):
        all_coalitions.extend(list(itertools.combinations(range(nAttr), r)))
    

    coalition_to_index = {coalition: idx for idx, coalition in enumerate(all_coalitions)}
    data_opt = np.zeros((nSamp, len(all_coalitions)))
    
    # f_CI(v, x_i) = Σ_{j=1}^{m} (x_{i,(j)} - x_{i,(j-1)}) * v({(j), ..., (m)})
    for i in range(nSamp):
        order = np.argsort(X_orig[i])
        sorted_vals = np.sort(X_orig[i])
        prev = 0.0
        
        for j in range(nAttr):
            # Calculate the difference (x_{i,(j)} - x_{i,(j-1)})
            diff = sorted_vals[j] - prev
            prev = sorted_vals[j]
            
            full_coalition = tuple(sorted(order[j:]))
            
            if len(full_coalition) <= max_size:
                idx = coalition_to_index.get(full_coalition)
                if idx is not None:
                    data_opt[i, idx] = diff
            else:
                current_feature = order[j]
                remaining_features = order[j+1:]
                
                if max_size > 0:
                    if len(remaining_features) >= max_size - 1:
                        for subset in itertools.combinations(remaining_features, max_size - 1):
                            coalition = tuple(sorted((current_feature,) + subset))
                            idx = coalition_to_index.get(coalition)
                            if idx is not None:
                                weight = 1.0 / comb(len(remaining_features), max_size - 1)
                                data_opt[i, idx] += diff * weight
                    else:
                        coalition = full_coalition
                        idx = coalition_to_index.get(coalition)
                        if idx is not None:
                            data_opt[i, idx] += diff
    
    return data_opt, all_coalitions

def choquet_matrix_kadd_guilherme(X_orig, kadd):
    """
    Compute the Choquet integral transformation matrix restricted to coalitions
    of maximum size kadd.

    Parameters
    ----------
    X_orig : array-like, shape (n_samples, n_attributes)
        The input data.
    kadd : int
        Maximum coalition size allowed.

    Returns
    -------
    data_opt : ndarray, shape (n_samples, num_allowed_coalitions)
        The Choquet integral matrix using allowed coalitions.

    Raises
    ------
    ValueError
        If kadd is greater than the number of features in X_orig.
    """
    nSamp, nAttr = X_orig.shape
    if kadd > nAttr:
        raise ValueError(f"kadd ({kadd}) cannot be greater than the number of features ({nAttr}).")

    # Sort data row-wise and pad with zeros to compute successive differences
    X_orig_sort = np.sort(X_orig, axis=1)
    X_orig_sort_ind = np.argsort(X_orig, axis=1)
    X_orig_sort_ext = np.concatenate((np.zeros((nSamp, 1)), X_orig_sort), axis=1)

    max_coal_size = kadd
    num_coalitions = sum(comb(nAttr, r) for r in range(1, max_coal_size + 1))

    sequence = np.arange(nAttr)
    combin = 99 * np.ones((num_coalitions, nAttr), dtype=int)
    count = 0
    for r in range(1, max_coal_size + 1):
        combos = list(itertools.combinations(sequence, r))
        for i, combo in enumerate(combos):
            combin[count + i, :r] = combo
        count += len(combos)

    data_opt = np.zeros((nSamp, num_coalitions))
    # Only compute differences corresponding to allowed coalition sizes
    start_idx = nAttr - max_coal_size
    for jj in range(nSamp):
        for ii in range(start_idx, nAttr):
            coalition = np.concatenate((np.sort(X_orig_sort_ind[jj, ii:]), 99 * np.ones(ii, dtype=int)), axis=0).tolist()
            try:
                aux = combin.tolist().index(coalition)
            except ValueError:
                continue
            data_opt[jj, aux] = X_orig_sort_ext[jj, ii + 1] - X_orig_sort_ext[jj, ii]

    return data_opt

def choquet_matrix_2add(X_orig):
    
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape # Number of samples (train) and attrbiutes
    
    k_add = 2
    k_add_numb = int(nParam_kAdd(nAttr,k_add))
    
    coalit = np.zeros((k_add_numb,nAttr))
    
    for i,s in enumerate(powerset(range(nAttr),k_add)):
        s = list(s)
        coalit[i,s] = 1
        
    data_opt = np.zeros((nSamp,k_add_numb))
    for i in range(nAttr):
        data_opt[:,i+1] = data_opt[:,i+1] + X_orig[:,i]
        
        for i2 in range(i+1,nAttr):
            data_opt[:,(coalit[:,[i,i2]]==1).all(axis=1)] = (np.min([X_orig[:,i],X_orig[:,i2]],axis=0)).reshape(nSamp,1)
            
        for ii in range(nAttr+1,len(coalit)):
            if coalit[ii,i]==1:
                data_opt[:,ii] = data_opt[:,ii] + (-1/2)*X_orig[:,i]

    return data_opt[:,1:]



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
    standard_matrix, coalitions = choquet_matrix_kadd_guilherme(X, k_add=k_add)
    shapley_matrix = choquet_matrix_2add(X)
    
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



def differences_between_matrixes():
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Generate random data
    x = np.random.rand(100, 3)
    
    # Calculate both matrices
    eq22_matrix, coalitions = choquet_matrix(x, k_add=2)
    eq23_matrix = choquet_matrix_2add(x)
    
    # Calculate difference matrix
    diff_matrix = eq22_matrix - eq23_matrix
    
    # Print the relative norm of the difference
    relative_norm = np.linalg.norm(diff_matrix) / np.linalg.norm(eq22_matrix)
    print(f"Relative norm of difference: {relative_norm:.6f}")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap of differences
    ax = sns.heatmap(diff_matrix, cmap='coolwarm', center=0, 
                     annot=False, cbar=True)
    
    # Add labels and title
    plt.title('Differences Between Choquet Matrix Representations', fontsize=14)
    plt.xlabel('Coalition Index', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    
    # Create a second plot showing the average absolute difference per coalition
    plt.figure(figsize=(10, 6))
    mean_abs_diff = np.mean(np.abs(diff_matrix), axis=0)
    coalition_labels = [str(c) for c in coalitions]
    
    plt.bar(range(len(mean_abs_diff)), mean_abs_diff)
    plt.title('Average Absolute Difference per Coalition', fontsize=14)
    plt.xlabel('Coalition', fontsize=12)
    plt.ylabel('Mean Absolute Difference', fontsize=12)
    
    # Only add coalition labels if not too many
    if len(coalitions) <= 10:
        plt.xticks(range(len(mean_abs_diff)), coalition_labels, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return diff_matrix

differences_between_matrixes()
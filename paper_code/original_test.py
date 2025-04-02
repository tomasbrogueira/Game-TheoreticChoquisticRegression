import os
os.environ["SCIPY_ARRAY_API"] = "1"

import itertools
import numpy as np
from math import factorial, comb
from itertools import chain, combinations

import matplotlib.pyplot as plt
import mod_GenFuzzyRegression as mGFR

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Define functions from original.py here instead of importing
def powerset(iterable,k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes
    powerset([1,2,..., m],m) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m) ... (1, ..., m)
    powerset([1,2,..., m],2) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add+1))

def nParam_kAdd(kAdd,nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr,ii+1)
    return aux_numb

#def choquet_matrix(X):
    """
    Generate the full Choquet matrix for the given dataset
    """
    n_samples, n_features = X.shape
    feature_indices = list(range(n_features))
    
    # Get all possible subsets except the empty set
    all_subsets = list(powerset(feature_indices))[1:]
    n_subsets = len(all_subsets)
    
    choquet_X = np.zeros((n_samples, n_subsets))
    
    # For each subset, compute the minimum value of features in that subset
    for i, subset in enumerate(all_subsets):
        if len(subset) > 0:
            choquet_X[:, i] = np.min(X[:, subset], axis=1)
    
    return choquet_X

#def choquet_matrix_2add(X):
    """
    Generate the Choquet matrix with 2-additions for the given dataset
    (only considering interactions between at most 2 features)
    """
    n_samples, n_features = X.shape
    feature_indices = list(range(n_features))
    
    # Get subsets with at most 2 elements (excluding empty set)
    subsets_2add = [s for s in powerset(feature_indices) if 0 < len(s) <= 2]
    n_subsets = len(subsets_2add)
    
    choquet_X = np.zeros((n_samples, n_subsets))
    
    # For each subset, compute the minimum value of features in that subset
    for i, subset in enumerate(subsets_2add):
        choquet_X[:, i] = np.min(X[:, subset], axis=1)
    
    return choquet_X

def choquet_matrix(X_orig):
    
    X_orig_sort = np.sort(X_orig)
    X_orig_sort_ind = np.array(np.argsort(X_orig))
    nSamp, nAttr = X_orig.shape # Number of samples (train) and attrbiutes
    X_orig_sort_ext = np.concatenate((np.zeros((nSamp,1)),X_orig_sort),axis=1)
    
    
    sequence = np.arange(nAttr)
    
    combin = (99)*np.ones((2**nAttr-1,nAttr))
    count = 0
    for ii in range(nAttr):
        combin[count:count+comb(nAttr,ii+1),0:ii+1] = np.array(list(itertools.combinations(sequence, ii+1)))
        count += comb(nAttr,ii+1)
    
    data_opt = np.zeros((nSamp,2**nAttr-1))
    for ii in range(nAttr):
        for jj in range(nSamp):
            list1 = combin.tolist()
            aux = list1.index(np.concatenate((np.sort(X_orig_sort_ind[jj,ii:]),99*np.ones((ii,))),axis=0).tolist())
            data_opt[jj,aux] = X_orig_sort_ext[jj,ii+1] - X_orig_sort_ext[jj,ii]
            
    return data_opt

def choquet_matrix_2add(X_orig):
    
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape # Number of samples (train) and attrbiutes
    
    k_add = 2
    k_add_numb = nParam_kAdd(k_add,nAttr)
    
    coalit = np.zeros((k_add_numb,nAttr))
    
    # Store the actual coalitions for printing
    coalition_list = []
    
    for i,s in enumerate(powerset(range(nAttr),k_add)):
        s = list(s)
        coalition_list.append(s)
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


def choquet_k_additive_game(X_orig, k_add=2):

    X_orig = np.asarray(X_orig)
    nSamp, nAttr = X_orig.shape

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

def choquet_k_additive_unordered(X_orig, k_add=2):
    """Refined implementation of k-additive Choquet integral transformation."""
    X_orig = np.asarray(X_orig)
    nSamp, nAttr = X_orig.shape
    
    # Generate all valid coalitions up to size k_add
    all_coalitions = []
    for r in range(1, k_add+1):
        all_coalitions.extend(list(combinations(range(nAttr), r)))
    
    # Calculate number of features in the transformed space
    n_transformed = len(all_coalitions)
    
    # Initialize output matrix
    transformed = np.zeros((nSamp, n_transformed))
    
    # Performance optimization: Pre-compute coalition membership tests
    # This significantly reduces the computational burden of the inner loop
    coalition_members = {}
    for coal_idx, coalition in enumerate(all_coalitions):
        coalition_set = set(coalition)
        coalition_members[coal_idx] = coalition_set
    
    # For each sample
    for i in range(nSamp):
        x = X_orig[i]
        
        # Sort feature indices by their values
        sorted_indices = np.argsort(x)
        sorted_values = x[sorted_indices]
        
        # Add a sentinel value for the first difference
        sorted_values_ext = np.concatenate([[0], sorted_values])
        
        # For each position in the sorted feature list
        for j in range(nAttr):
            # Current feature index and value
            feat_idx = sorted_indices[j]
            # Difference with previous value
            diff = sorted_values_ext[j+1] - sorted_values_ext[j]
            
            # All features from this position onward
            higher_features = set(sorted_indices[j:])
            
            # Find all valid coalitions containing this feature and higher features
            for coal_idx, coalition_set in coalition_members.items():
                # Check if coalition is valid (optimized set operations)
                if feat_idx in coalition_set and coalition_set.issubset(higher_features):
                    transformed[i, coal_idx] += diff
    
    return transformed

def choquet_k_additive_mobius(X_orig, k_add=2):

    X_orig = np.asarray(X_orig)
    nSamp, nAttr = X_orig.shape

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

# Mappin functions for full choquet
def create_coalition_mapping(nAttr):
    """Creates a mapping between refined and original coalition indices"""
    # Generate coalitions in order used by choquet_matrix
    sequence = np.arange(nAttr)
    combin = (99)*np.ones((2**nAttr-1,nAttr))
    count = 0
    for ii in range(nAttr):
        combin[count:count+comb(nAttr,ii+1),0:ii+1] = np.array(list(itertools.combinations(sequence, ii+1)))
        count += comb(nAttr,ii+1)
    
    # Convert to tuple format for comparison
    original_coalitions = []
    for i in range(combin.shape[0]):
        coal = tuple(sorted([int(j) for j in combin[i] if j != 99]))
        original_coalitions.append(coal)
    
    # Generate refined coalitions
    refined_coalitions = []
    for r in range(1, nAttr+1):
        refined_coalitions.extend(list(combinations(range(nAttr), r)))
    
    # Create mapping from refined to original index
    mapping = {}
    for refined_idx, refined_coal in enumerate(refined_coalitions):
        orig_idx = original_coalitions.index(refined_coal)
        mapping[refined_idx] = orig_idx
        
    return mapping, original_coalitions, refined_coalitions

def compare_reordered_matrices(X_orig, nAttr):
    # Get original and refined matrices
    original = choquet_matrix(X_orig)
    refined = choquet_k_additive_game(X_orig, k_add=nAttr)
    
    # Get mapping
    mapping, orig_coalitions, refined_coalitions = create_coalition_mapping(nAttr)
    
    # Create reordered refined matrix
    reordered_refined = np.zeros_like(refined)
    for refined_idx, orig_idx in mapping.items():
        reordered_refined[:, orig_idx] = refined[:, refined_idx]
    
    # Compare
    are_equal = np.allclose(original, reordered_refined, rtol=1e-5, atol=1e-5)
    return are_equal, original, reordered_refined

def compare_reordered_matrices_2add(X_orig, nAttr):
    # Get original and refined matrices
    original = choquet_matrix_2add(X_orig)
    refined = choquet_k_additive_game(X_orig, k_add=2)
    
    # Get coalitions from original implementation
    coalition_list = []
    for i,s in enumerate(powerset(range(nAttr),2)):
        if s:  # Skip empty set
            coalition_list.append(tuple(sorted(s)))
    
    # Get coalitions from refined implementation
    refined_coalitions = []
    for r in range(1, min(2, nAttr)+1):
        refined_coalitions.extend(list(combinations(range(nAttr), r)))
    
    # Create mapping
    mapping = {}
    for refined_idx, refined_coal in enumerate(refined_coalitions):
        if refined_coal in coalition_list:
            orig_idx = coalition_list.index(refined_coal)
            mapping[refined_idx] = orig_idx
    
    # Reorder refined matrix
    reordered_refined = np.zeros_like(original)
    for refined_idx, orig_idx in mapping.items():
        reordered_refined[:, orig_idx] = refined[:, refined_idx]
    
    # Compare
    are_equal = np.allclose(original, reordered_refined, rtol=1e-5, atol=1e-5)
    return are_equal, original, reordered_refined

# game to shap transformation
from scipy.special import bernoulli

def tr_shap2game(nAttr, k_add):
    '''Return the transformation matrix from Shapley interaction indices, given a k_additive model, to game'''
    nBern = bernoulli(k_add) #Números de Bernoulli
    k_add_numb = nParam_kAdd(k_add,nAttr)
    
    coalit = np.zeros((k_add_numb,nAttr))
    
    for i,s in enumerate(powerset(range(nAttr),k_add)):
        s = list(s)
        coalit[i,s] = 1
        
    matrix_shap2game = np.zeros((k_add_numb,k_add_numb))
    for i in range(coalit.shape[0]):
        for i2 in range(k_add_numb):
            aux2 = int(sum(coalit[i2,:]))
            aux3 = int(sum(coalit[i,:] * coalit[i2,:]))
            aux4 = 0
            for i3 in range(int(aux3+1)):
                aux4 += comb(aux3, i3) * nBern[aux2-i3]
            matrix_shap2game[i,i2] = aux4
    return matrix_shap2game

def tr_shap2game_general(nAttr, k_add):
    """Transformation matrix from Shapley indices to game parameters for any k-additive model."""
    # Generate all subsets A, B with |A|, |B| <= k_add
    subsets = list(powerset(range(nAttr), k_add))
    n_params = len(subsets)
    
    # Bernoulli numbers (alpha_0 to alpha_{k_add})
    alpha = bernoulli(k_add + 1)  # Includes alpha_0 to alpha_{k_add}
    alpha[1] = -0.5  # Correct sign for alpha_1
    
    # Initialize transformation matrix
    T = np.zeros((n_params, n_params))
    
    # Populate T[A][B] = gamma^{|B|}_{|A ∩ B|}
    for A_idx, A in enumerate(subsets):
        A_set = set(A)
        for B_idx, B in enumerate(subsets):
            B_size = len(B)
            intersect_size = len(A_set.intersection(B))
            
            # Compute gamma^{B_size}_{intersect_size}
            gamma = 0.0
            for q in range(intersect_size + 1):
                alpha_idx = B_size - q
                if alpha_idx >= 0 and alpha_idx < len(alpha):
                    gamma += comb(intersect_size, q) * alpha[alpha_idx]
            T[A_idx, B_idx] = gamma
    
    return T

def shapley_transform(X_game, nAttr, k_add):
    """Transform from game basis to Shapley basis.
       Note: We use the inverse of the Shapley-to-game transformation matrix.
    """
    nSamp = X_game.shape[0]
    full_k_add_numb = nParam_kAdd(k_add, nAttr)  # includes empty set
    
    # Get transformation matrix (Shapley to game)
    transform_matrix_full = tr_shap2game_general(nAttr, k_add)
    print(f"Transform matrix shape: {transform_matrix_full.shape}")
    
    # Check if X_game includes the empty set or not
    expected_cols_with_empty = transform_matrix_full.shape[1]
    expected_cols_without_empty = transform_matrix_full.shape[1] - 1
    
    # Determine if we need to remove empty set from transformation matrix
    if X_game.shape[1] == expected_cols_without_empty:
        # Input doesn't include empty set, so remove it from transformation
        transform_matrix = transform_matrix_full[1:, 1:]
        print(f"X_game has {X_game.shape[1]} columns, assuming empty set is excluded")
    elif X_game.shape[1] == expected_cols_with_empty:
        # Input includes empty set, use full matrix
        transform_matrix = transform_matrix_full
        print(f"X_game has {X_game.shape[1]} columns, assuming empty set is included")
    else:
        raise ValueError(f"X_game has {X_game.shape[1]} columns, but expected either {expected_cols_with_empty} (with empty set) or {expected_cols_without_empty} (without empty set)")
    
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    
    # Apply transformation
    X_shapley = np.zeros((nSamp, transform_matrix.shape[0]))
    for i in range(nSamp):
        X_shapley[i] = np.dot(inv_transform_matrix, X_game[i])
    
    return X_shapley


# data_imp = list(['dados_covid_sbpo_atual'])
data_imp = list(['banknotes'])

for ll in range(len(data_imp)):
    X, y = mGFR.func_read_data(data_imp[ll])

    # Data parameters
    nSamp,nAttr = X.shape
    print("Verifying Choquet model equivalence across different domains")
    print("=" * 70)

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale data to [0,1] range (important for both approaches)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    print(f"First X vs X_scaled: {X_train[0]} vs {X_train_scaled[0]}")

    # check coallitions being considered in each choquet matrix
    X_choquet = choquet_matrix(X_train_scaled)
    X_choquet_refined = choquet_k_additive_game(X_train_scaled, k_add=5)
    X_choquet_2add = choquet_matrix_2add(X_train_scaled)
    X_choquet_refined_2add = choquet_k_additive_game(X_train_scaled, k_add=2)
    print(f"first choquet matrix: {X_choquet[0]}")
    print(f"first choquet refined matrix: {X_choquet_refined[0]}")

    print(f"equality choquet matrices: {np.allclose(X_choquet, X_choquet_refined, rtol=1e-5, atol=1e-5)}")
    print("equality choquet 2-additive matrices: ", np.allclose(X_choquet_2add, X_choquet_refined_2add, rtol=1e-5, atol=1e-5))

    X_shapley = shapley_transform(X_choquet_refined_2add, nAttr, k_add=2)
    print(f"first shapley matrix: {X_shapley[0]}")
    print(f"first choquet 2-additive matrix: {X_choquet_2add[0]}")
    diff_2 = np.abs(X_choquet_2add - X_shapley)
    print(f"Max difference after transformation: {np.max(diff_2)}")
   
   # Create a small synthetic dataset for debugging:
    # Create a small synthetic dataset for thorough testing
    X_small = np.array([[0.2, 0.5, 0.8],
                        [0.1, 0.4, 0.9]])
    y_small = np.array([1, 0])  # Dummy labels
    nSamp_small, nAttr_small = X_small.shape

    # Use k_add = 2 for testing
    k_add = 1

    X_choquet_small = choquet_matrix(X_small)
    X_choquet_refined_small = choquet_k_additive_game(X_small, k_add=3)
    print("equality choquet matrices: ", np.allclose(X_choquet_small, X_choquet_refined_small, rtol=1e-5, atol=1e-5))
    print(f"first choquet matrix: {X_choquet_small}")


    # Test 1: Original implementation vs. choquet_k_additive with fixed padding
    print("\n=== Test 1: Comparing original implementation vs choquet_k_additive ===")
    X_choquet_orig = choquet_matrix_2add(X_small)
    X_choquet_refined = choquet_k_additive_game(X_small, k_add=k_add)
    X_choquet = choquet_matrix(X_small)
    X_choquet_mobius = choquet_k_additive_mobius(X_small, k_add=k_add)
    x_choquet_unordered = choquet_k_additive_unordered(X_small, k_add=k_add)
    print("Original 2-add Choquet:")
    print(X_choquet_orig)
    print("Refined Choquet without empty set:")
    print(X_choquet_refined)
    print("Game Choquet without empty set:")
    print(X_choquet_mobius)
    print("Full Choquet without empty set:")
    print(X_choquet)
    print("Refined Choquet with unordered features:")
    print(x_choquet_unordered)


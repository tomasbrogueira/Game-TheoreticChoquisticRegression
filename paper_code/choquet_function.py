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


def choquet_matrix_2add(X_orig):
    
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape # Number of samples (train) and attrbiutes
    
    k_add = 2
    k_add_numb = nParam_kAdd(k_add,nAttr)
    
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


def choquet_matrix_2add_fixed(X):
    """
    Improved implementation of the 2-additive Choquet integral transformation using correct coalition structure.
    This version properly represents singleton and pair coalitions, making it theoretically consistent
    with k-additivity theory.
    
    Args:
        X: Input matrix of shape (n_samples, n_features)
        
    Returns:
        Transformed matrix using proper 2-additive coalition structure
    """
    # Get shape info
    n_samples, n_features = X.shape
    
    # Calculate the number of expected columns for 2-additivity
    n_singletons = n_features  # All size-1 subsets
    n_pairs = n_features * (n_features - 1) // 2  # All size-2 subsets
    total_columns = n_singletons + n_pairs
    
    # Create output matrix
    output = np.zeros((n_samples, total_columns))
    
    # Fill in singleton (size 1) coalitions
    for i in range(n_features):
        output[:, i] = X[:, i]
    
    # Fill in pair (size 2) coalitions
    col_idx = n_features
    for i in range(n_features):
        for j in range(i+1, n_features):
            # Create interaction term using minimum (standard for Choquet integral)
            output[:, col_idx] = np.minimum(X[:, i], X[:, j])
            col_idx += 1
    
    return output


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


def strict_kadd_choquet(X, k_add=2):
    """
    Strict k-additive Choquet integral implementation that correctly
    represents exactly the theoretical coalitions and nothing more.
    
    Args:
        X: Input data matrix of shape (n_samples, n_features)
        k_add: Maximum coalition size (k-additivity level)
        
    Returns:
        Transformed matrix with exactly the right coalitions for k-additivity
    """
    n_samples, n_features = X.shape
    
    # Calculate number of coalitions for each size from 1 to k_add
    num_coalitions = sum(comb(n_features, r) for r in range(1, k_add+1))
    
    # Initialize output matrix
    output = np.zeros((n_samples, num_coalitions))
    
    # Current column index
    col_idx = 0
    
    # Add singleton coalitions (size 1)
    for i in range(n_features):
        output[:, col_idx] = X[:, i]
        col_idx += 1
    
    # Add coalitions of sizes 2 to k_add
    for size in range(2, k_add+1):
        for combo in combinations(range(n_features), size):
            # Use minimum operator for interactions (Choquet standard)
            output[:, col_idx] = np.min(X[:, combo], axis=1)
            col_idx += 1
    
    return output


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

    # 1. Calculate both transformation matrices
    print("Computing transformation matrices...")

    # Game domain (k=2)
    X_game_train = choquet_matrix_kadd_guilherme(X_train_scaled, kadd=2)
    X_game_test = choquet_matrix_kadd_guilherme(X_test_scaled, kadd=2)

    # Shapley domain (2-additive)
    X_shapley_train = choquet_matrix_2add(X_train_scaled)
    X_shapley_test = choquet_matrix_2add(X_test_scaled)

    print(f"Game domain matrix shape: {X_game_train.shape}")
    print(f"Shapley domain matrix shape: {X_shapley_train.shape}")

    # 2. Compare matrix structure
    print("\nMatrix Statistics:")
    print(f"Game domain mean: {np.mean(X_game_train):.6f}, std: {np.std(X_game_train):.6f}")
    print(f"Shapley domain mean: {np.mean(X_shapley_train):.6f}, std: {np.std(X_shapley_train):.6f}")

    # 3. Train logistic regression models on both transformations
    print("\nTraining models...")

    # Game domain model
    game_model = LogisticRegression(max_iter=1000)
    game_model.fit(X_game_train, y_train)

    # Shapley domain model
    shapley_model = LogisticRegression(max_iter=1000)
    shapley_model.fit(X_shapley_train, y_train)

    # 4. Compare predictions and performance
    game_preds = game_model.predict(X_game_test)
    shapley_preds = shapley_model.predict(X_shapley_test)

    game_proba = game_model.predict_proba(X_game_test)[:, 1]
    shapley_proba = shapley_model.predict_proba(X_shapley_test)[:, 1]

    # Calculate metrics
    game_acc = accuracy_score(y_test, game_preds)
    shapley_acc = accuracy_score(y_test, shapley_preds)

    game_auc = roc_auc_score(y_test, game_proba)
    shapley_auc = roc_auc_score(y_test, shapley_proba)

    print("\nPerformance Comparison:")
    print(f"Game domain    - Accuracy: {game_acc:.4f}, AUC: {game_auc:.4f}")
    print(f"Shapley domain - Accuracy: {shapley_acc:.4f}, AUC: {shapley_auc:.4f}")

    # Calculate agreement between models
    agreement = np.mean(game_preds == shapley_preds)
    correlation = np.corrcoef(game_proba, shapley_proba)[0, 1]

    print(f"\nModel Agreement:")
    print(f"Prediction agreement: {agreement:.4f} (fraction of identical predictions)")
    print(f"Probability correlation: {correlation:.4f}")

    # 5. Optional: Visualize probability predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(game_proba, shapley_proba, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Game Domain Probabilities')
    plt.ylabel('Shapley Domain Probabilities')
    plt.title('Comparison of Prediction Probabilities')
    plt.grid(True)
    plt.savefig('choquet_domain_comparison.png')
    print("\nPlot saved as 'choquet_domain_comparison.png'")

    # 6. Compare feature importance
    print("\nFeature Importance Comparison:")
    print("Note: Coefficients will differ due to different domain representations")
    print("Game domain top coefficients:")
    game_coef = game_model.coef_[0]
    top_game = np.argsort(np.abs(game_coef))[::-1][:10]
    for i, idx in enumerate(top_game):
        print(f"  {i+1}. Feature {idx}: {game_coef[idx]:.6f}")

    print("\nShapley domain top coefficients:")
    shapley_coef = shapley_model.coef_[0]
    top_shapley = np.argsort(np.abs(shapley_coef))[::-1][:10]
    for i, idx in enumerate(top_shapley):
        print(f"  {i+1}. Feature {idx}: {shapley_coef[idx]:.6f}")
import numpy as np
import math
from itertools import combinations, chain

def nParam_kAdd(kAdd, nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += math.comb(nAttr, ii+1)
    return aux_numb

def powerset(iterable, k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes'''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add+1))

def choquet_k_additive_shapley(X_orig, k_add=None):
    """
    k‑additive ‘Shapley‑basis’ transform up to order k_add.
    
    For each sample x in X_orig, we build
      v_x(∅)=0,  v_x(S)=min(x[j] for j in S) for S≠∅
    and then for every coalition A with 1 ≤ |A| ≤ k_add compute
      g_A(x) = ∑_{C⊆A} [ (-1)^{|A|-|C|} / (|A|-|C|+1) ] · v_x(C).
    
    The output array has columns ordered first by |A|=1 (singletons),
    then all |A|=2 pairs, … up to |A|=k_add, in lex order within each size.
    """
    X = np.asarray(X_orig, dtype=float)
    n_samp, n = X.shape

    # default to full n‑way if not specified
    if k_add is None:
        k_add = n
    elif k_add > n:
        raise ValueError("k_add cannot exceed number of features")

    # 1) build list of all “v‑coalitions” up to size k_add, including ∅
    v_coalitions = [()]  # index 0 is the empty set
    for r in range(1, k_add+1):
        v_coalitions += list(combinations(range(n), r))
    idx_of = {coal: i for i, coal in enumerate(v_coalitions)}

    # 2) compute v_x(C) for every sample x and every coalition C
    m = len(v_coalitions)
    V = np.zeros((n_samp, m))
    for i, C in enumerate(v_coalitions[1:], start=1):
        if len(C) == 1:
            V[:, i] = X[:, C[0]]
        else:
            V[:, i] = X[:, C].min(axis=1)

    # 3) build the list of output coalitions (those of size 1..k_add)
    out_coals = []
    for r in range(1, k_add+1):
        out_coals += list(combinations(range(n), r))

    # 4) allocate output using the formula
    T = np.zeros((n_samp, len(out_coals)))
    for j, A in enumerate(out_coals):
        s = len(A)
        # sum over all C ⊆ A
        for r in range(s+1):
            coeff = (-1)**(s-r) / (s-r + 1)
            for C in combinations(A, r):
                T[:, j] += coeff * V[:, idx_of[C]]

    return T

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

def choquet_k_additive_game(X_orig, k_add=None, full=False):
    """
    Game‐basis feature map: raw suffix‐differences (g_j) ready for logistic regression.

    Features g_j(x) = x_(j) - x_(j-1), for the nested suffix sets S_j = {indices of top n-j+1 values}.
    k_add (optional) restricts to suffixes of size <= k_add.

    Parameters
    ----------
    X_orig : array‐like, shape (n_samples, n_features)
    k_add   : int or None  (default None)
        If None, include all n suffixes; if k, only include suffixes of cardinality <= k.
    full    : bool  (default False)
        If False, return compact matrix shape (n_samples, m) with m = # kept suffixes.
        If True, return full 2^n-1 matrix, scattering each g_j into its coalition column.

    Returns
    -------
    T : ndarray
        Feature matrix for logistic‐regression training:
        - full=False: (n_samples, m)
        - full=True : (n_samples, 2^n - 1)
    """
    X = np.asarray(X_orig, dtype=float)
    n_samp, n = X.shape

    # 1) sort and compute raw diffs g_j = x_(j) - x_(j-1)
    order = np.argsort(X, axis=1)
    X_srt = np.take_along_axis(X, order, axis=1)
    X_ext = np.concatenate((np.zeros((n_samp,1)), X_srt), axis=1)
    diffs = X_ext[:,1:] - X_ext[:,:-1]    # shape (n_samp, n)

    # 2) decide which suffixes to include under k_add
    if k_add is None:
        include = [True]*n
    else:
        include = [ (n-j) <= k_add for j in range(n) ]

    # 3) if compact form: return only diffs for included suffixes
    if not full:
        cols = [j for j,inc in enumerate(include) if inc]
        return diffs[:, cols]

    # 4) full form: scatter each g_j into the 2^n-1 coalition vector
    #    build global list of all nonempty subsets, in size→lex order
    all_coals = []
    for r in range(1, n+1):
        all_coals.extend(combinations(range(n), r))
    coal_idx = {coal:i for i,coal in enumerate(all_coals)}

    T_full = np.zeros((n_samp, len(all_coals)))
    for i in range(n_samp):
        for j in range(n):
            if include[j]:
                S_j = tuple(sorted(order[i,j:]))
                T_full[i, coal_idx[S_j]] = diffs[i, j]
    return T_full

def choquet_k_additive_game_len(X_orig, k_add=None, full=True):
    """
    Game-basis Choquet transform, full or properly k-additive.

    Parameters
    ----------
    X_orig : array-like, shape (n_samples, n_features)
        Input data.
    k_add : int or None
        If None: use the full capacity v_x(S)=min(x[S]) on suffix-sets.
        If integer k: enforce true k-additivity by truncating Möbius terms m(A)
        for |A|>k, then recomputing the capacity on each suffix-set.
    full : bool, default=False
        If False: return the n-dimensional suffix-difference vector.
        If True: return the 2^n−1 vector that places each suffix diff
        into its matching subset-column (zeros elsewhere).

    Returns
    -------
    T : ndarray
        - If full=False: shape (n_samples, n_features)
        - If full=True:  shape (n_samples, 2^n − 1)
    """
    X = np.asarray(X_orig, dtype=float)
    n_samp, n = X.shape

    # 1) sort and compute differences x_(j) - x_(j-1)
    order = np.argsort(X, axis=1)
    X_srt = np.take_along_axis(X, order, axis=1)
    X_ext = np.concatenate((np.zeros((n_samp,1)), X_srt), axis=1)
    diffs = X_ext[:,1:] - X_ext[:,:-1]   # (n_samp, n)

    # 2) if k_add specified, build & truncate Möbius coefficients
    if k_add is not None:
        # list all subsets A of {0..n-1}
        full_coals = [()]
        for r in range(1, n+1):
            full_coals += list(combinations(range(n), r))
        idx_full = {coal: idx for idx, coal in enumerate(full_coals)}

        # compute full Möbius m_full(A)=v_x(A) for all nonempty A
        M_full = np.zeros((n_samp, len(full_coals)-1))
        for j, A in enumerate(full_coals[1:], start=0):
            if len(A) == 1:
                M_full[:, j] = X[:, A[0]]
            else:
                M_full[:, j] = X[:, A].min(axis=1)

        # truncate high-order terms
        m_trunc = np.zeros_like(M_full)
        for A, idx in idx_full.items():
            if A and len(A) <= k_add:
                m_trunc[:, idx-1] = M_full[:, idx-1]

    # 3) compute the n suffix-difference features
    G = np.zeros((n_samp, n))
    for i in range(n_samp):
        for j in range(n):
            S_j = tuple(sorted(order[i, j:]))
            if k_add is None:
                # capacity v_x(S_j) = min over x[S_j]
                mu = X[i, list(S_j)].min() if S_j else 0.0
            else:
                # truncated capacity: mu_k(S_j)=sum_{A⊆S_j} m_trunc(A)
                mu = 0.0
                for r in range(1, len(S_j)+1):
                    for A in combinations(S_j, r):
                        mu += m_trunc[i, idx_full[A]-1]
            G[i, j] = diffs[i, j] * mu

    if not full:
        return G

    # 4) build the full 2^n-1 vector placing each G[:,j] into the column for suffix S_j
    all_coals = []
    for r in range(1, n+1):
        all_coals += list(combinations(range(n), r))
    T_full = np.zeros((n_samp, len(all_coals)))
    for i in range(n_samp):
        for j in range(n):
            S_j = tuple(sorted(order[i, j:]))
            col = all_coals.index(S_j)
            T_full[i, col] = G[i, j]
    return T_full

def choquet_k_additive_game_old(X_orig, k_add=None):

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

def choquet_matrix(X_orig):
    X_orig_sort = np.sort(X_orig)
    X_orig_sort_ind = np.array(np.argsort(X_orig))
    nSamp, nAttr = X_orig.shape
    X_orig_sort_ext = np.concatenate((np.zeros((nSamp, 1)), X_orig_sort), axis=1)
    
    sequence = np.arange(nAttr)
    combin = (99)*np.ones((2**nAttr-1, nAttr))
    count = 0
    for ii in range(nAttr):
        combin[count:count+math.comb(nAttr, ii+1), 0:ii+1] = np.array(list(combinations(sequence, ii+1)))
        count += math.comb(nAttr, ii+1)
    
    data_opt = np.zeros((nSamp, 2**nAttr-1))
    for ii in range(nAttr):
        for jj in range(nSamp):
            list1 = combin.tolist()
            aux = list1.index(np.concatenate((np.sort(X_orig_sort_ind[jj, ii:]), 99*np.ones((ii,))), axis=0).tolist())
            data_opt[jj, aux] = X_orig_sort_ext[jj, ii+1] - X_orig_sort_ext[jj, ii]
    return data_opt

def choquet_matrix_2add(X_orig):
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

def choquet_k_additive_game_idk(X_orig, k_add=None, full=False):
    """
    Proper k-additive game-basis transform via Möbius truncation.

    For each sample x, computes sorted differences g_j(x)=x_(j)-x_(j-1),
    reconstructs a k-additive capacity by truncating Möbius coefficients,
    then multiplies g_j by the truncated capacity on each suffix set.
    Supports compact (n features) or full (2^n-1 features) output.
    """
    X = np.asarray(X_orig, dtype=float)
    n_samp, n = X.shape

    # 1) compute sorted differences g_j(x)
    order = np.argsort(X, axis=1)
    X_srt = np.take_along_axis(X, order, axis=1)
    X_ext = np.concatenate([np.zeros((n_samp,1)), X_srt], axis=1)
    diffs = X_ext[:,1:] - X_ext[:,:-1]  # shape (n_samp, n)

    # 2) build all coalitions and full Möbius coefficients
    full_coals = [()]
    for r in range(1, n+1):
        full_coals += list(combinations(range(n), r))
    idx_full = {coal: idx for idx, coal in enumerate(full_coals)}

    # compute v_x(A) = min(x[A]) for all A≠∅
    M_full = np.zeros((n_samp, len(full_coals)-1))
    for j, A in enumerate(full_coals[1:], start=0):
        if len(A) == 1:
            M_full[:, j] = X[:, A[0]]
        else:
            M_full[:, j] = X[:, A].min(axis=1)

    # 3) truncate Möbius for |A|>k_add
    if k_add is None or k_add >= n:
        m_trunc = M_full.copy()
    else:
        m_trunc = np.zeros_like(M_full)
        for A, idx in idx_full.items():
            if A and len(A) <= k_add:
                m_trunc[:, idx-1] = M_full[:, idx-1]

    # 4) build truncated capacity mu_k on each suffix S_j
    suffixes = [tuple(range(j, n)) for j in range(n)]
    suffix2mob = []
    for S in suffixes:
        cols = []
        for r in range(1, len(S)+1):
            for A in combinations(S, r):
                cols.append(idx_full[A] - 1)
        suffix2mob.append(cols)

    # compute mu_k for each sample and suffix j
    mu_k = np.zeros((n_samp, n))
    for j, cols in enumerate(suffix2mob):
        if cols:
            mu_k[:, j] = m_trunc[:, cols].sum(axis=1)

    # 5) multiply diffs by mu_k
    G = diffs * mu_k

    # compact output: n features
    if not full:
        return G

    # full output: scatter into 2^n-1 vector
    all_coals = full_coals[1:]
    T_full = np.zeros((n_samp, len(all_coals)))
    for i in range(n_samp):
        for j, S in enumerate(suffixes):
            idx = all_coals.index(S)
            T_full[i, idx] = G[i, j]
    return T_full


def compare_choquet_matrix(X_orig):
    choquet_mat = choquet_matrix(X_orig)
    choquet_mat_2add = choquet_matrix_2add(X_orig)
    
    choquet_k = choquet_k_additive_game_idk(X_orig)
    choquet_k_2add = choquet_k_additive_shapley(X_orig, k_add=2)

    # Check if the two matrices are equal
    full = np.allclose(choquet_mat, choquet_k)
    add = np.allclose(choquet_mat_2add, choquet_k_2add)
    print("choquet_mat == choquet_k: ", full)
    print("choquet_mat_2add == choquet_k_2add: ", add)

def compare_game_choquet(X_orig):
    for k_add in range(len(X_orig[0])+1):
        choquet_k = choquet_k_additive_game_old(X_orig, k_add=k_add)
        len_choquet = choquet_k_additive_game_len(X_orig, k_add=k_add, full=True)
        choquet_new = choquet_k_additive_game_len(X_orig, k_add=k_add, full=False)
        print(choquet_k.shape)
        print(choquet_new)
    



if __name__ == "__main__":
    # Example usage
    X_orig = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    choquet_mat      = choquet_matrix(   X_orig)           # shape (3,7)
    choquet_k_full   = choquet_k_additive_game(
                        X_orig,
                        k_add=None,
                        full=True           # <-- ask for full 2^n-1 vector
                    )                        # shape (3,7)

    print(choquet_mat.shape, choquet_k_full.shape)
    # (3, 7) (3, 7)
    print(choquet_mat)
    print(choquet_k_full)

    print("Equal?", np.allclose(choquet_mat, choquet_k_full))
    # should now be True

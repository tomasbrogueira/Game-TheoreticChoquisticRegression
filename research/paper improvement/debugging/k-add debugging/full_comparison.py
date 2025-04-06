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
    return chain.from_iterable(combinations(s, r) for r in range(0, k_add + 1))


def choquet_matrix(X_orig):
    
    X_orig_sort = np.sort(X_orig)
    X_orig_sort_ind = np.array(np.argsort(X_orig))
    nSamp, nAttr = X_orig.shape # Number of samples (train) and attrbiutes
    X_orig_sort_ext = np.concatenate((np.zeros((nSamp,1)),X_orig_sort),axis=1)
    
    
    sequence = np.arange(nAttr)
    
    combin = (99)*np.ones((2**nAttr-1,nAttr))
    count = 0
    for ii in range(nAttr):
        combin[count:count+int(comb(nAttr,ii+1)),0:ii+1] = np.array(list(itertools.combinations(sequence, ii+1)))
        count += int(comb(nAttr,ii+1))
    
    data_opt = np.zeros((nSamp,2**nAttr-1))
    for ii in range(nAttr):
        for jj in range(nSamp):
            list1 = combin.tolist()
            aux = list1.index(np.concatenate((np.sort(X_orig_sort_ind[jj,ii:]),99*np.ones((ii,))),axis=0).tolist())
            data_opt[jj,aux] = X_orig_sort_ext[jj,ii+1] - X_orig_sort_ext[jj,ii]
            
    return data_opt



def choquet_matrix_new(X_orig, k_add=None):
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

def choquet_matrix_2add(X_orig, k_add=None):
    
    X_orig = np.array(X_orig)
    nSamp, nAttr = X_orig.shape # Number of samples (train) and attrbiutes
    
    k_add = k_add if k_add is not None else 2
    k_add_numb = int(nParam_kAdd(k_add,nAttr))
    
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

def compare_choquet_matrix():
    X_orig = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    data_opt_old = choquet_matrix(X_orig)
    data_opt_new, coalitions = choquet_matrix_new(X_orig)
    data_opt_2add = choquet_matrix_2add(X_orig, k_add=3)
    print(np.allclose(data_opt_old, data_opt_new))
    print(np.allclose(data_opt_old, data_opt_2add))

compare_choquet_matrix()
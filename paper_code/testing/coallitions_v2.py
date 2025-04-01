import numpy as np
from itertools import combinations
from choquet_function import choquet_matrix_kadd_guilherme, choquet_matrix_2add

def test_coalition_restrictions():
    """Test if the implementations properly restrict coalition sizes"""
    
    print("=== Testing Coalition Size Restrictions ===")
    
    # Create test data with 4 features
    n_features = 4
    X_test = np.random.rand(2, n_features)
    
    # Test different k_add values
    for k_add in [1, 2, 3, 4]:
        print(f"\nTesting k_add = {k_add}")
        
        # Expected coalition structure
        expected_coalitions = []
        for r in range(1, k_add + 1):
            expected_coalitions.extend(list(combinations(range(n_features), r)))
        
        print(f"Expected max coalition size: {k_add}")
        print(f"Expected number of coalitions (excluding empty set): {len(expected_coalitions)}")
        
        # Apply Game domain transformation
        game_result = choquet_matrix_kadd_guilherme(X_test, kadd=k_add)
        
        print(f"\nGame domain output shape: {game_result.shape}")
        print(f"Game domain output non-zero elements: {np.count_nonzero(game_result)}")
        
        # Apply prime number test to detect coalitions
        prime_test = np.zeros((1, n_features))
        primes = [2, 3, 5, 7]
        for i in range(n_features):
            prime_test[0, i] = primes[i]
        
        game_result_prime = choquet_matrix_kadd_guilherme(prime_test, kadd=k_add)
        
        # Analyze coalition sizes by checking products
        max_coalition_size = 0
        for val in game_result_prime[0]:
            if val > 0:
                # Identify which primes are in this product
                coalition = []
                test_val = val
                for i, p in enumerate(primes):
                    if test_val % p == 0:
                        coalition.append(i)
                max_coalition_size = max(max_coalition_size, len(coalition))
        
        print(f"Detected max coalition size: {max_coalition_size}")
        
        if max_coalition_size > k_add:
            print(f"ERROR: Coalition size {max_coalition_size} exceeds k_add={k_add}")
        else:
            print(f"Coalition size restriction correctly implemented")

if __name__ == "__main__":
    test_coalition_restrictions()
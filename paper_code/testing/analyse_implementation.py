import numpy as np
from choquet_function import choquet_matrix_kadd_guilherme, choquet_matrix_2add

def analyze_implementations():
    """Analyze and compare the implementation approaches of both Choquet functions"""
    
    print("=== Choquet Implementation Analysis ===")
    
    # Game Domain (choquet_matrix_kadd_guilherme)
    print("\n1. Game Domain Implementation:")
    print("- Sorts features by value for each sample")
    print("- Uses the classic Choquet integral formula:")
    print("  C_μ(x) = Σ (x_σ(i) - x_σ(i-1)) * μ({σ(i), ..., σ(n)})")
    print("- Coalition structure is permutation-dependent")
    print("- Handles feature importance through differences in sorted values")
    
    # Shapley Domain (choquet_matrix_2add)
    print("\n2. Shapley Domain Implementation:")
    print("- Directly models individual feature contributions")
    print("- Uses a 2-additive formula based on Shapley values:")
    print("  C_μ(x) = Σ φ_j * x_j + Σ I_{i,j} * min(x_i, x_j)")
    print("- Coalition structure focuses on individual features and pairs")
    print("- Different mathematical basis than the game domain")
    
    print("\nKEY FINDING: These are different mathematical formulations that")
    print("achieve similar goals but aren't directly transformable without")
    print("accounting for the implementation differences.")

# Test with simple cases
def test_simple_cases():
    """Test both implementations with simple cases to verify behavior"""
    
    print("\n=== Simple Test Cases ===\n")
    
    # Create simple test cases with known values
    # Case 1: Unit matrix (each feature = 1.0)
    X_unit = np.ones((1, 4))
    
    # Case 2: Identity matrix-like (diagonal = 1.0, rest = 0.0)
    X_identity = np.eye(4)
    
    # Case 3: Increasing values
    X_increasing = np.array([[0.2, 0.4, 0.6, 0.8]])
    
    # Test cases with both implementations
    test_cases = [
        ("Unit Matrix", X_unit),
        ("Identity Matrix", X_identity),
        ("Increasing Values", X_increasing)
    ]
    
    for name, X in test_cases:
        print(f"\nTest Case: {name}")
        print(f"Input shape: {X.shape}")
        print("Input values:")
        print(X)
        
        # Apply both transformations
        game_result = choquet_matrix_kadd_guilherme(X, kadd=2)
        shapley_result = choquet_matrix_2add(X)
        
        print("\nGame Domain Output:")
        print(f"Shape: {game_result.shape}")
        print("Non-zero elements and their indices:")
        non_zero_game = np.where(np.abs(game_result) > 1e-10)
        for i, j in zip(*non_zero_game):
            print(f"  [{i},{j}]: {game_result[i,j]:.6f}")
        
        print("\nShapley Domain Output:")
        print(f"Shape: {shapley_result.shape}")
        print("Non-zero elements and their indices:")
        non_zero_shapley = np.where(np.abs(shapley_result) > 1e-10)
        for i, j in zip(*non_zero_shapley):
            print(f"  [{i},{j}]: {shapley_result[i,j]:.6f}")
        
        print("\nOutput comparison:")
        print(f"Number of non-zero elements (Game): {len(non_zero_game[0])}")
        print(f"Number of non-zero elements (Shapley): {len(non_zero_shapley[0])}")

if __name__ == "__main__":
    analyze_implementations()
    test_simple_cases()
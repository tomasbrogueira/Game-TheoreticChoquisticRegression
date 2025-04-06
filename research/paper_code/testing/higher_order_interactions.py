import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from choquet_function import choquet_matrix_kadd_guilherme, choquet_matrix_2add

def create_higher_order_dataset(n_samples=1000):
    """
    Create a dataset where prediction requires higher-order interactions.
    
    Uses an XOR-like pattern where individual features and pairs have limited
    predictive power, but combinations of 3+ features are highly predictive.
    """
    print("Creating dataset with higher-order interactions...")
    
    # Generate random features
    X = np.random.rand(n_samples, 4)
    
    # Create target that depends on a 3-way interaction
    # y = 1 when (x1 > 0.5 XOR x2 > 0.5 XOR x3 > 0.5), with some noise
    y = (X[:, 0] > 0.5) ^ (X[:, 1] > 0.5) ^ (X[:, 2] > 0.5)
    
    # Add some 2-way interaction effect
    mask_2way = (X[:, 0] > 0.7) & (X[:, 1] > 0.7)
    y[mask_2way] = ~y[mask_2way]  # Flip predictions for some 2-way interactions
    
    # Add noise
    noise = np.random.rand(n_samples) < 0.1
    y = np.logical_xor(y, noise).astype(int)
    
    print(f"Dataset created with {n_samples} samples and 4 features")
    print("Target depends primarily on 3-way interactions")
    
    return X, y

def test_implementation_interactions():
    """Test hypothesis that original implementation captures higher-order interactions"""
    print("=== Testing Higher-Order Interactions Hypothesis ===\n")
    
    # Create dataset with higher-order interactions
    X, y = create_higher_order_dataset()
    
    # Split into train/test
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    results = []
    
    # Define implementations to test
    implementations = [
        ("Original Game", lambda X, k: choquet_matrix_kadd_guilherme(X, kadd=k)),
        ("Refined Game", lambda X, k: refined_choquet_k_additive(X, k_add=k)[0]),
        ("Shapley", lambda X, k: choquet_matrix_2add(X))
    ]
    
    # Test with different k values
    k_values = [1, 2, 3]
    
    for name, implementation in implementations:
        if name == "Shapley" and len(k_values) > 1:
            # Shapley implementation doesn't use k parameter
            k_values_to_test = [k_values[0]]
        else:
            k_values_to_test = k_values
            
        for k in k_values_to_test:
            print(f"\nTesting {name} implementation with k={k}...")
            
            # Transform the data
            X_train_trans = implementation(X_train, k)
            X_test_trans = implementation(X_test, k)
            
            # Train logistic regression
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_trans, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_trans)
            accuracy = accuracy_score(y_test, y_pred)
            
            try:
                y_proba = model.predict_proba(X_test_trans)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = np.nan
            
            # Store results
            results.append({
                'Implementation': name,
                'k': k,
                'Accuracy': accuracy,
                'AUC': auc,
                'Transformed Shape': X_train_trans.shape[1]
            })
            
            print(f"  Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            print(f"  Transformed features: {X_train_trans.shape[1]}")
            
            # Analyze model coefficients
            if name == "Original Game":
                coef = model.coef_[0]
                top_indices = np.argsort(np.abs(coef))[::-1][:5]
                print("  Top 5 coefficients:")
                for i in top_indices:
                    print(f"    Feature {i}: {coef[i]:.6f}")
    
    # Create summary table
    print("\n=== Summary of Results ===")
    print(f"{'Implementation':<15} {'k':<3} {'Accuracy':<10} {'AUC':<10} {'#Features':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['Implementation']:<15} {r['k']:<3} {r['Accuracy']:<10.4f} {r['AUC']:<10.4f} {r['Transformed Shape']:<10}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    
    # Group by implementation and k value
    implementations = sorted(set(r['Implementation'] for r in results))
    k_values = sorted(set(r['k'] for r in results))
    
    x = np.arange(len(implementations))
    width = 0.25
    
    for i, k in enumerate(k_values):
        accuracies = [next((r['Accuracy'] for r in results if r['Implementation'] == impl and r['k'] == k), 0) 
                     for impl in implementations]
        
        plt.bar(x + i*width, accuracies, width, label=f'k={k}')
    
    plt.xlabel('Implementation')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Implementations on Higher-Order Interaction Dataset')
    plt.xticks(x + width, implementations)
    plt.legend()
    plt.ylim(0.5, 1.0)  # Start from chance level
    plt.grid(axis='y')
    plt.savefig('higher_order_comparison.png')
    print("\nPlot saved as 'higher_order_comparison.png'")

def refined_choquet_k_additive(X_orig, k_add=2):
    """
    Refined implementation of k-additive Choquet integral transformation.
    """
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
            # Difference with previous value
            diff = sorted_values_ext[j+1] - sorted_values_ext[j]
            
            # All features from this position onward
            higher_features = sorted_indices[j:]
            
            # Find all valid coalitions containing this feature and higher features
            for coal_idx, coalition in enumerate(all_coalitions):
                # Check if coalition is valid: contains current feature and only higher features
                if sorted_indices[j] in coalition and all(f in higher_features for f in coalition):
                    transformed[i, coal_idx] += diff
    
    return transformed, all_coalitions

if __name__ == "__main__":
    test_implementation_interactions()
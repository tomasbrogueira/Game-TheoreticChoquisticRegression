import pickle
import numpy as np
import matplotlib.pyplot as plt
from regression_classes import compute_shapley_values, compute_choquet_interaction_matrix

# Load the debug data
debug_file = 'plots/dados_covid_sbpo_atual/model_debug.pkl'
try:
    with open(debug_file, 'rb') as f:
        debug_data = pickle.load(f)
    print(f"Successfully loaded debug data from {debug_file}")
except:
    print(f"Could not load {debug_file}, trying alternative location...")
    debug_file = 'model_debug.pkl'
    with open(debug_file, 'rb') as f:
        debug_data = pickle.load(f)

# Extract components
v = debug_data['v']
all_coalitions = debug_data['all_coalitions']
feature_names = debug_data['feature_names']
m = len(feature_names)

# Recompute the Shapley values
shapley_values = compute_shapley_values(v, m, all_coalitions)

# Extract singleton values directly
marginal_values = np.zeros(m)
for i in range(m):
    singleton = (i,)
    if singleton in all_coalitions:
        idx = all_coalitions.index(singleton)
        marginal_values[i] = v[idx + 1]  # +1 for empty set

# Recompute the interaction matrix
interaction_matrix = compute_choquet_interaction_matrix(v, m, all_coalitions)

# Calculate overall interactions by different methods
method1 = 0.5 * np.sum(interaction_matrix, axis=1)  # Standard matrix method
method1_neg = -0.5 * np.sum(interaction_matrix, axis=1)  # Negated matrix method
method2 = shapley_values - marginal_values  # Direct Shapley - marginal

print("=== INTERACTION CALCULATION DEBUG ===")

# Compare all methods
print("\nFeature-by-feature comparison:")
print(f"{'Feature':<15} {'Shapley':<10} {'Marginal':<10} {'Shap-Marg':<10} {'MatrixMethod':<12} {'NegMatrix':<12} {'BetterMatch'}")
print("-" * 85)

for i in range(m):
    name = feature_names[i][:15] if len(feature_names[i]) > 15 else feature_names[i]
    shapley = shapley_values[i]
    marginal = marginal_values[i]
    diff_direct = method2[i]
    matrix_method = method1[i]
    neg_matrix = method1_neg[i]
    
    diff1 = abs(diff_direct - matrix_method)
    diff2 = abs(diff_direct - neg_matrix)
    better = "NEGATED" if diff2 < diff1 else "STANDARD"
    
    print(f"{name:<15} {shapley:<10.6f} {marginal:<10.6f} {diff_direct:<10.6f} {matrix_method:<12.6f} {neg_matrix:<12.6f} {better}")

# Calculate overall statistics
diff_std = np.mean(np.abs(method2 - method1))
diff_neg = np.mean(np.abs(method2 - method1_neg))

print(f"\nAverage difference with standard method: {diff_std:.6f}")
print(f"Average difference with negated method: {diff_neg:.6f}")
print(f"Better overall match: {'NEGATED' if diff_neg < diff_std else 'STANDARD'}")

# Let's also try a scaled version
scale_factor = np.linalg.norm(method2) / np.linalg.norm(method1) if np.linalg.norm(method1) > 0 else 1
method1_scaled = method1 * scale_factor

print(f"\nOptimal scale factor for standard method: {scale_factor:.6f}")
print(f"Avg diff with scaled standard method: {np.mean(np.abs(method2 - method1_scaled)):.6f}")

# Visualize the relationships
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.scatter(method2, method1)
plt.plot([-1, 1], [-1, 1], 'r--')  # Identity line
plt.xlabel('Shapley - Marginal')
plt.ylabel('0.5 * Σ I(i,j)')
plt.title('Standard Matrix Method vs Ground Truth')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)  
plt.scatter(method2, method1_neg)
plt.plot([-1, 1], [-1, 1], 'r--')  # Identity line
plt.xlabel('Shapley - Marginal')
plt.ylabel('-0.5 * Σ I(i,j)')
plt.title('Negated Matrix Method vs Ground Truth')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
bar_width = 0.35
x = np.arange(m)
plt.bar(x - bar_width/2, method2, bar_width, label='Shapley - Marginal')
plt.bar(x + bar_width/2, method1, bar_width, label='0.5 * Σ I(i,j)')
plt.xlabel('Features')
plt.ylabel('Overall Interaction Effect')
plt.title('Comparison of Calculation Methods')
plt.xticks(x, [f'F{i}' for i in range(m)], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
bar_width = 0.35
plt.bar(x - bar_width/2, method2, bar_width, label='Shapley - Marginal')
plt.bar(x + bar_width/2, method1_neg, bar_width, label='-0.5 * Σ I(i,j)')
plt.xlabel('Features')
plt.ylabel('Overall Interaction Effect')
plt.title('Using Negated Matrix Method')
plt.xticks(x, [f'F{i}' for i in range(m)], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('interaction_method_debug.png')
plt.close()

print(f"\nDebug visualization saved to 'interaction_method_debug.png'")

# Try to derive the correct formula from first principles
print("\n=== MATHEMATICAL ANALYSIS ===")

# Test if the matrix calculation has sign issues
if np.corrcoef(method2, -method1)[0, 1] > np.corrcoef(method2, method1)[0, 1]:
    print("FINDING: The negated matrix method correlates better with ground truth!")
    correct_formula = "φᵢ = v({i}) - 0.5 * Σⱼ≠ᵢ I({i,j})"
else:
    print("FINDING: The standard matrix method correlates better with ground truth!")
    correct_formula = "φᵢ = v({i}) + 0.5 * Σⱼ≠ᵢ I({i,j})"
    
print(f"Correct formula appears to be: {correct_formula}")

# For more advanced analysis, use the verify_shapley_decomposition_relationship function
if 'verify_choquet_relationships' in globals() or 'verify_shapley_decomposition_relationship' in globals():
    print("\nRunning deeper verification...")
    from verify_choquet_relationships import verify_shapley_decomposition_relationship
    verify_shapley_decomposition_relationship(v, all_coalitions, feature_names)

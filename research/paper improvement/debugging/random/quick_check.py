import pickle
import numpy as np
from regression_classes import compute_shapley_values, compute_choquet_interaction_matrix

# Load debug data
with open('plots/dados_covid_sbpo_atual/model_debug.pkl', 'rb') as f:
    debug_data = pickle.load(f)

v = debug_data['v']
all_coalitions = debug_data['all_coalitions']
feature_names = debug_data['feature_names']

# Get number of features
m = max(max(c) for c in all_coalitions if c) + 1

# Compute values for verification
shapley = compute_shapley_values(v, m, all_coalitions)
interaction = compute_choquet_interaction_matrix(v, m, all_coalitions)

# Extract marginal values
marginal = np.zeros(m)
for i in range(m):
    singleton = (i,)
    if singleton in all_coalitions:
        idx = all_coalitions.index(singleton)
        marginal[i] = v[idx + 1]

# Calculate using both formulations
standard = marginal + 0.5 * np.sum(interaction, axis=1)
negated = marginal - 0.5 * np.sum(interaction, axis=1)

# Print feature-by-feature results
print(f"{'Feature':<15} {'Shapley':<10} {'Marginal':<10} {'Standard':<10} {'Std Diff':<10} {'Negated':<10} {'Neg Diff':<10}")
print("-" * 80)

for i in range(m):
    name = feature_names[i]
    print(f"{name:<15} {shapley[i]:<10.6f} {marginal[i]:<10.6f} {standard[i]:<10.6f} "
          f"{shapley[i]-standard[i]:<10.6f} {negated[i]:<10.6f} {shapley[i]-negated[i]:<10.6f}")

print(f"\nStandard formula - average error: {np.mean(np.abs(shapley - standard)):.6f}")
print(f"Negated formula - average error: {np.mean(np.abs(shapley - negated)):.6f}")

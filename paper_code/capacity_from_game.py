import numpy as np
from scipy.optimize import minimize

# Data
X = np.array([[0.2, 0.5, 0.8], [0.1, 0.4, 0.9]])
y = np.array([1, 0])

# Obtained coefficients [[0.  0.  0.3 0.  0.  0.3 0.2] [0.  0.  0.5 0.  0.  0.3 0.1]]
#f1 = np.array([0, 0, 0.3, 0, 0, 0.3])
#f2 = np.array([0, 0, 0.5, 0, 0, 0.3])
#matrix = np.array([f1, f2])


f1 = np.array([0.2,0.5,0.8,0.2,0.2,0.5])
f2 = np.array([0.1,0.4,0.9,0.1,0.1,0.4])
matrix = np.array([f1, f2])

# Define the loss function
def loss(params, matrix=matrix, y=y):
    v1, v2, v3, v4, v5, v6 = params
    param_vector = np.array([v1, v2, v3, v4, v5, v6])
    
    # Compute f_CI for each sample by multiplying the matrix with the parameters one by one
    f_CI = matrix @ param_vector

    # Sigmoid probabilities
    sigmoid_outputs = 1 / (1 + np.exp(-f_CI))
    
    loss_values = y * np.log(sigmoid_outputs) + (1 - y) * np.log(1 - sigmoid_outputs)
    # Cross-entropy loss
    return -np.sum(loss_values)

def compute_capacities(result_params):
    v1, v2, v3, v4, v5, v6 = result_params
    
    # Initialize capacities
    mu = {}
    mu[frozenset()] = 0  # Empty set has capacity 0
    
    # Individual features (singletons)
    mu[frozenset([1])] = v1
    mu[frozenset([2])] = v2
    mu[frozenset([3])] = v3
    
    # For pairs, using both phi values and interaction indices
    mu[frozenset([1, 2])] = v1 + v2 + v4
    mu[frozenset([1, 3])] = v1 + v3 + v5
    mu[frozenset([2, 3])] = v2 + v3 + v6
    
    # For the full set
    mu[frozenset([1, 2, 3])] = v1 + v2 + v3 + v4 + v5 + v6
    
    return mu

# Initial guess for parameters
initial_guess = np.zeros(6)

# Minimize the loss
result = minimize(loss, initial_guess, method='BFGS')
print("Optimal parameters:", result.x)

optimal_params = result.x
capacities = compute_capacities(optimal_params)
# Print the capacities for each subset
for subset, value in capacities.items():
    print(f"μ({list(subset) if subset else '∅'}) = {value:.4f}")
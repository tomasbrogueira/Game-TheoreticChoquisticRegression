import numpy as np
from scipy.optimize import minimize

# Data
X = np.array([[0.2, 0.5, 0.8], [0.1, 0.4, 0.9]])
y = np.array([1, 0])

# Obtained coefficients 
#f1 = np.array([0.2, 0.5, 0.8, 0.2, 0.2, 0.5])
#f2 = np.array([0.1, 0.4, 0.9, 0.1, 0.1, 0.4])
#matrix = np.array([f1, f2])

f1 = np.array([0.2, 0.5, 0.8, -0.15, -0.3, -0.15])
f2 = np.array([0.1, 0.4, 0.9, -0.15, -0.4, -0.25])
matrix = np.array([f1, f2])

# Define the loss function
def loss(params, matrix=matrix, y=y):
    phi1, phi2, phi3, I12, I13, I23 = params
    
    # Compute f_CI for each sample by multiplying the matrix with the parameters one by one
    f1 = matrix[0] @ np.array([phi1, phi2, phi3, I12, I13, I23])
    f2 = matrix[1] @ np.array([phi1, phi2, phi3, I12, I13, I23])

    # Sigmoid probabilities
    p1 = 1 / (1 + np.exp(-f1))
    p2 = 1 / (1 + np.exp(-f2))
    
    # Cross-entropy loss
    return - (y[0]*np.log(p1) + (1-y[0])*np.log(1-p1) + y[1]*np.log(p2) + (1-y[1])*np.log(1-p2))

def compute_capacities(result_params):
    phi1, phi2, phi3, I12, I13, I23 = result_params
    
    # Initialize capacities
    mu = {}
    mu[frozenset()] = 0  # Empty set has capacity 0
    
    # Individual features (singletons)
    mu[frozenset([1])] = phi1
    mu[frozenset([2])] = phi2
    mu[frozenset([3])] = phi3
    
    # For pairs, using both phi values and interaction indices
    mu[frozenset([1, 2])] = phi1 + phi2 + I12
    mu[frozenset([1, 3])] = phi1 + phi3 + I13
    mu[frozenset([2, 3])] = phi2 + phi3 + I23
    
    # For the full set
    mu[frozenset([1, 2, 3])] = phi1 + phi2 + phi3 + I12 + I13 + I23
    
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
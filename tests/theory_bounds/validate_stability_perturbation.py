import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os
import json

# Adjust path to locate the core model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from core.models.regression import ChoquisticRegression
except ImportError:
    pass

RESULTS_FILE = 'theory_validation_results.json'

def update_results_json(experiment_key, new_data):
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            try:
                full_data = json.load(f)
            except: full_data = {}
    else:
        full_data = {}
    full_data[experiment_key] = new_data
    with open(RESULTS_FILE, 'w') as f:
        json.dump(full_data, f, indent=4)
    print(f"  -> Data saved to {RESULTS_FILE} under key '{experiment_key}'")

def run_stability(seed=42):
    np.random.seed(seed)
    n = 10; k = 1 
    N = 100
    # Range covering linear and saturation
    C_values = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5] 
    repeats = 40
    
    print(f"--- Stability Test: Linear Regime vs Saturation (seed={seed}) ---")
    
    json_data = {
        'description': "Stability test measuring parameter sensitivity to single label perturbation.",
        'seed': seed,
        'N': N,
        'n': n,
        'results': []
    }
    plot_C, plot_diff, plot_err = [], [], []

    for C in C_values:
        print(f"Testing C={C}...")
        diffs = []
        for rep in range(repeats):
            np.random.seed(seed + int(C * 1000) + rep)  # Deterministic seed for each iteration
            X = np.random.rand(N, n)
            y = np.random.randint(0, 2, N) 
            
            # Train Model 1
            m1 = ChoquisticRegression(k_add=k, penalty='l2', C=C, tol=1e-6)
            m1.fit(X, y)
            w1 = m1.model_.coef_.flatten()

            # Train Model 2 (1 flipped label)
            y_flip = y.copy()
            y_flip[0] = 1 - y_flip[0]
            m2 = ChoquisticRegression(k_add=k, penalty='l2', C=C, tol=1e-6)
            m2.fit(X, y_flip)
            w2 = m2.model_.coef_.flatten()
            
            diffs.append(np.linalg.norm(w1 - w2))
        
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        
        plot_C.append(C)
        plot_diff.append(mean_diff)
        plot_err.append(std_diff)
        
        json_data['results'].append({'C': C, 'mean': mean_diff, 'std': std_diff})

    update_results_json('stability_test', json_data)

    # --- PLOTTING ---
    plt.figure(figsize=(8, 6))
    
    # 1. Empirical Data
    plt.errorbar(plot_C, plot_diff, yerr=plot_err, fmt='o', 
                 color='#6a0dad', ecolor='gray', capsize=4, 
                 label='Empirical Sensitivity')
    
    # 2. Linear Regime Fit (The "Actual Value" of the Linear Bound)
    # Fit line to the first few points (C <= 0.75) where behavior is strictly linear
    linear_points = 5
    slope_emp = np.mean([plot_diff[i]/plot_C[i] for i in range(linear_points)])
    
    x_range = np.linspace(0, max(plot_C)*1.05, 100)
    linear_fit = slope_emp * x_range
    
    plt.plot(x_range, linear_fit, 'k--', linewidth=1.5, 
             label=r'Theoretical Linear Trend ($\beta \propto C$)')

    plt.xlabel(r'Inverse Regularisation ($C$)', fontsize=12)
    plt.ylabel(r'Parameter Sensitivity $\|\Delta \theta\|_2$', fontsize=12)
    plt.title(r'Stability: Linear Regime vs Saturation', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('stability_proof_clean.png', dpi=300)
    print("Saved stability_proof_clean.png")

if __name__ == "__main__":
    run_stability()
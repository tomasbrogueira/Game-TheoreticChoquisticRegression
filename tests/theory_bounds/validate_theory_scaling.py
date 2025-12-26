import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import comb
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from core.models.regression import ChoquisticRegression
except ImportError:
    pass

RESULTS_FILE = 'theory_validation_results.json'

def update_results_json(experiment_key, new_data):
    """Updates the shared JSON file with new experiment data."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            try:
                full_data = json.load(f)
            except json.JSONDecodeError:
                full_data = {}
    else:
        full_data = {}
    
    full_data[experiment_key] = new_data
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(full_data, f, indent=4)
    print(f"  -> Data saved to {RESULTS_FILE} under key '{experiment_key}'")

def vc_bound_formula(D, N, delta=0.05):
    """Vapnik-Chervonenkis Bound Formula"""
    if D > N: return 1.0 
    return np.sqrt((D * (np.log(2 * N / D) + 1) + np.log(4 / delta)) / N)

def run_scaling(seed=42):
    np.random.seed(seed)
    n_features = 10
    N = 1000
    repeats = 10
    k_vals = range(1, 9) 
    
    print(f"--- Scaling Test: Dynamic Bound Calculation (seed={seed}) ---")
    
    # Structure for JSON
    json_data = {
        'description': "Worst-case noise memorization test. Compares empirical gap vs theoretical VC bound.",
        'seed': seed,
        'N': N,
        'n_features': n_features,
        'results': []
    }
    
    # Temp lists for plotting
    plot_D = []
    plot_vc = []
    plot_none = []
    plot_l1 = []
    plot_l2 = []
    
    for k in k_vals:
        D = int(sum([comb(n_features, i) for i in range(1, k+1)]))
        theo_val = vc_bound_formula(D, N)
        
        print(f"Testing k={k} (D={D})... Bound={theo_val:.3f}")
        
        g_none, g_l1, g_l2 = [], [], []
        
        for rep in range(repeats):
            np.random.seed(seed + k * 100 + rep)  # Deterministic seed for each iteration
            X = np.random.rand(N, n_features)
            y = np.random.randint(0, 2, N)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
            
            # 1. Unregularized
            m = ChoquisticRegression(k_add=k, penalty='l2', C=1e4, max_iter=2000)
            m.fit(X_tr, y_tr)
            g_none.append(accuracy_score(y_tr, m.predict(X_tr)) - accuracy_score(y_te, m.predict(X_te)))
            
            # 2. L1
            m = ChoquisticRegression(k_add=k, penalty='l1', C=0.1, solver='liblinear')
            m.fit(X_tr, y_tr)
            g_l1.append(accuracy_score(y_tr, m.predict(X_tr)) - accuracy_score(y_te, m.predict(X_te)))
            
            # 3. L2
            m = ChoquisticRegression(k_add=k, penalty='l2', C=0.1)
            m.fit(X_tr, y_tr)
            g_l2.append(accuracy_score(y_tr, m.predict(X_tr)) - accuracy_score(y_te, m.predict(X_te)))

        # Save stats for this k
        step_data = {
            'k': k,
            'D': D,
            'theoretical_vc_bound': float(theo_val),
            'gap_unregularized_mean': float(np.mean(g_none)),
            'gap_unregularized_std': float(np.std(g_none)),
            'gap_l1_mean': float(np.mean(g_l1)),
            'gap_l2_mean': float(np.mean(g_l2))
        }
        json_data['results'].append(step_data)
        
        plot_D.append(D)
        plot_vc.append(theo_val)
        plot_none.append(np.mean(g_none))
        plot_l1.append(np.mean(g_l1))
        plot_l2.append(np.mean(g_l2))

    # Save to JSON
    update_results_json('scaling_test', json_data)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(plot_D, plot_vc, 'k--', linewidth=2, label=r'Theoretical VC Bound ($\sqrt{D/N}$)')
    plt.plot(plot_D, plot_none, 'ro-', alpha=0.7, label='Unregularized Gap')
    plt.plot(plot_D, plot_l1, 'g^-', alpha=0.7, label='L1 Gap')
    plt.plot(plot_D, plot_l2, 'bs-', alpha=0.7, label='L2 Gap')
    plt.fill_between(plot_D, 0, plot_vc, color='gray', alpha=0.1)
    plt.xlabel('Model Dimension ($D_k$)')
    plt.ylabel('Generalization Gap')
    plt.title(f'Empirical Gap vs. Theoretical VC Bound (N={N})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('theoretical_validation_avg.png')
    print("Saved theoretical_validation_avg.png")

if __name__ == "__main__":
    run_scaling()
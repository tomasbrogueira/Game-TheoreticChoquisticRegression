import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

def run_spectrum(seed=42):
    np.random.seed(seed)
    n = 12; k = 2; N = 200
    repeats = 10
    sparsity_levels = [2, 5, 10, 20, 40, 60]
    
    print(f"--- Spectrum Test: Dynamic Bound Values (seed={seed}) ---")
    
    json_data = {
        'description': "Impact of feature correlation/density on L1 vs L2 accuracy and learned complexity norms.",
        'seed': seed,
        'N': N,
        'n': n,
        'results': []
    }
    
    # Temp lists for plotting
    s_vals = []
    l1_acc_mean = []
    l2_acc_mean = []
    l1_norm_mean = []
    l2_norm_mean = []

    for s in sparsity_levels:
        print(f"Testing Sparsity S={s}...")
        l1_scores, l2_scores = [], []
        l1_bounds, l2_bounds = [], []
        
        for rep in range(repeats):
            np.random.seed(seed + s * 100 + rep)  # Deterministic seed for each iteration
            # 1. Correlated Data Generation
            X_base = np.random.rand(N, n)
            for i in range(4): # Correlation
                X_base[:, n-1-i] = X_base[:, i] + np.random.normal(0, 0.05, N)
            
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            active_idx = np.random.choice(len(all_pairs), size=s, replace=False)
            
            logit = np.zeros(N)
            for idx in active_idx:
                i, j = all_pairs[idx]
                term = np.minimum(X_base[:, i], X_base[:, j])
                logit += np.random.choice([-1, 1]) * term
            
            logit = (logit - np.mean(logit)) / (np.std(logit) + 1e-9) * 3.0
            y = np.where(1/(1+np.exp(-logit)) > 0.5, 1, 0)
            flip = np.random.choice(N, int(0.1*N), replace=False)
            y[flip] = 1 - y[flip]
            
            X_tr, X_te, y_tr, y_te = train_test_split(X_base, y, test_size=0.3, random_state=seed)
            
            # 2. L1 Model & Norm Calc
            m1 = ChoquisticRegression(k_add=k, penalty='l1', C=0.5, solver='liblinear')
            m1.fit(X_tr, y_tr)
            l1_scores.append(accuracy_score(y_te, m1.predict(X_te)))
            
            # Calculate Dynamic L1 Bound Proxy: ||theta||_1
            w1 = m1.model_.coef_.flatten() if hasattr(m1, 'model_') else np.zeros(1)
            l1_bounds.append(np.sum(np.abs(w1)))
            
            # 3. L2 Model & Norm Calc
            m2 = ChoquisticRegression(k_add=k, penalty='l2', C=0.5)
            m2.fit(X_tr, y_tr)
            l2_scores.append(accuracy_score(y_te, m2.predict(X_te)))
            
            # Calculate Dynamic L2 Bound Proxy: ||theta||_2
            w2 = m2.model_.coef_.flatten() if hasattr(m2, 'model_') else np.zeros(1)
            l2_bounds.append(np.linalg.norm(w2))
            
        step_res = {
            'sparsity': s,
            'l1_accuracy_mean': float(np.mean(l1_scores)),
            'l2_accuracy_mean': float(np.mean(l2_scores)),
            'l1_norm_mean': float(np.mean(l1_bounds)),
            'l2_norm_mean': float(np.mean(l2_bounds))
        }
        json_data['results'].append(step_res)
        
        s_vals.append(s)
        l1_acc_mean.append(np.mean(l1_scores))
        l2_acc_mean.append(np.mean(l2_scores))
        l1_norm_mean.append(np.mean(l1_bounds))
        l2_norm_mean.append(np.mean(l2_bounds))

    update_results_json('spectrum_test', json_data)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Empirical Accuracy
    ax1.plot(s_vals, l1_acc_mean, 'g-s', label='L1 Accuracy')
    ax1.plot(s_vals, l2_acc_mean, 'b-^', label='L2 Accuracy')
    ax1.set_xlabel('Problem Complexity (S)')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Empirical Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: The Calculated Theoretical "Cost"
    ax2.plot(s_vals, l1_norm_mean, 'g--', label=r'Learned Complexity ($\|\hat{\theta}\|_1$)')
    ax2.plot(s_vals, l2_norm_mean, 'b--', label=r'Learned Energy ($\|\hat{\theta}\|_2$)')
    ax2.set_xlabel('Problem Complexity (S)')
    ax2.set_ylabel('Norm of Learned Weights')
    ax2.set_title('Complexity Cost Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bound_comparison_spectrum.png')
    print("Saved bound_comparison_spectrum.png")

if __name__ == "__main__":
    run_spectrum()
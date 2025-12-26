import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import comb
import sys
import os
import json

# Add path to your project root to import core models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from core.models.regression import ChoquisticRegression
    from core.models.choquet import ChoquetTransformer
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

def calculate_effective_dimension(Phi):
    """Calculates Stable Rank: (Tr(Sigma))^2 / Tr(Sigma^2)"""
    # Center data to get covariance
    Phi_centered = Phi - np.mean(Phi, axis=0)
    try:
        # Singular values s. Eigenvalues of Sigma are s^2
        _, s, _ = np.linalg.svd(Phi_centered, full_matrices=False)
        eig_vals = s**2
        if np.sum(eig_vals**2) == 0: return 0
        d_eff = (np.sum(eig_vals))**2 / np.sum(eig_vals**2)
        return d_eff
    except:
        return 0

def run_effective_dimension_test(seed=42):
    np.random.seed(seed)
    n_features = 8; N = 1000; repeats = 10
    # Test deeper interactions to see the divergence
    k_vals = range(1, 9) 
    
    print(f"--- Validating The 'True' Dimension (Effective Rank) (seed={seed}) ---")
    
    json_data = {'seed': seed, 'results': []}
    plot_k, plot_D, plot_deff = [], [], []
    plot_gap_unreg, plot_gap_l2 = [], []

    for k in k_vals:
        D_raw = int(sum([comb(n_features, i) for i in range(1, k+1)]))
        print(f"Testing k={k} (D={D_raw})...")
        
        gaps_unreg, gaps_l2 = [], []
        eff_dims = []
        
        # Use transformer to get the exact basis used by the model
        try:
            transformer = ChoquetTransformer(representation='mobius', k_add=k)
        except:
             print("Transformer not found, skipping...")
             continue

        for rep in range(repeats):
            np.random.seed(seed + k * 100 + rep)  # Deterministic seed for each iteration
            X = np.random.rand(N, n_features)
            y = np.random.randint(0, 2, N)
            
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
            
            # 1. Unregularized (High C) -> Should follow D_k
            m_unreg = ChoquisticRegression(k_add=k, penalty='l2', C=1e5, max_iter=2000)
            m_unreg.fit(X_tr, y_tr)
            gap_u = accuracy_score(y_tr, m_unreg.predict(X_tr)) - accuracy_score(y_te, m_unreg.predict(X_te))
            gaps_unreg.append(gap_u)
            
            # 2. L2 Regularized (Standard C=1) -> Should follow d_eff
            m_l2 = ChoquisticRegression(k_add=k, penalty='l2', C=1.0, max_iter=2000)
            m_l2.fit(X_tr, y_tr)
            gap_r = accuracy_score(y_tr, m_l2.predict(X_tr)) - accuracy_score(y_te, m_l2.predict(X_te))
            gaps_l2.append(gap_r)
            
            # Measure Effective Dimension
            Phi = transformer.fit_transform(X)
            eff_dims.append(calculate_effective_dimension(Phi))
            
        mean_deff = float(np.mean(eff_dims))
        mean_gap_unreg = float(np.mean(gaps_unreg))
        mean_gap_l2 = float(np.mean(gaps_l2))
        
        step_res = {
            'k': k, 'D': D_raw, 'd_eff': mean_deff,
            'gap_unreg': mean_gap_unreg, 'gap_l2': mean_gap_l2
        }
        json_data['results'].append(step_res)
        
        plot_k.append(k); plot_D.append(D_raw); plot_deff.append(mean_deff)
        plot_gap_unreg.append(mean_gap_unreg); plot_gap_l2.append(mean_gap_l2)

    update_results_json('effective_dimension_test', json_data)

    # --- PLOT (Matches LaTeX Figure) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Axis 1: Dimensions (Log Scale for D_k to show explosion)
    ax1.plot(plot_k, plot_D, 'k:', label='Combinatorial Dimension $D_k$')
    ax1.plot(plot_k, plot_deff, 'b-', linewidth=3, alpha=0.3, label='Effective Dimension $d_{eff}$')
    ax1.set_xlabel('Interaction Order (k)')
    ax1.set_ylabel('Dimension / Capacity', color='black')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Axis 2: Empirical Gaps (Linear Scale)
    ax2 = ax1.twinx()
    ax2.plot(plot_k, plot_gap_unreg, 'r--', linewidth=2, label='Unregularized Gap (Follows $D_k$)')
    ax2.plot(plot_k, plot_gap_l2, 'b-o', linewidth=2, label='L2 Regularized Gap (Follows $d_{eff}$)')
    
    ax2.set_ylabel('Generalization Gap', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Combine Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'L2 Regularization vs Effective Dimension')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_effective.png')
    print("Saved plot_effective.png")

if __name__ == "__main__":
    run_effective_dimension_test()
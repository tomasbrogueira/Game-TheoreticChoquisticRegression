import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb, factorial
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# IMPORTANT: set these before any other imports!
os.environ["SCIPY_ARRAY_API"] = "1"
pd.set_option('future.no_silent_downcasting', True)

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# Import our Choquistic module and the data reader module.
from root_cr_new import ChoquisticRegression
import mod_GenFuzzyRegression as modGF

def run_experiment(
    data_imp='dados_covid_sbpo_atual',
    test_size=0.2,
    random_state=0,
    n_simulations=1,
    solver_lr=('newton-cg', 'sag'),
    baseline_max_iter=1000,
    baseline_logistic_params=None,
    choq_logistic_params=None,
    methods=["choquet_2add", "choquet", "mlm", "mlm_2add"],
    scale_data=True,
    plot_folder="plots",
    results_filename="results.pkl",
    log_odds_bins=30
):
    """
    Runs an experiment comparing baseline Logistic Regression with ChoquisticRegression models.
    - Loads a dataset using modGF.func_read_data
    - Normalizes the data to [0,1]
    - Runs both baseline Logistic Regression and ChoquisticRegression models for each method
    - Collects accuracy, Shapley values, interaction matrices, and log-odds for analysis
    - Generates and saves plots

    Parameters:
      - data_imp: dataset identifier
      - test_size: fraction of data used for testing
      - random_state: seed for reproducibility
      - n_simulations: number of independent simulation runs
      - solver_lr: tuple of solver names for baseline LR
      - baseline_max_iter: max iterations for baseline LR
      - baseline_logistic_params: additional params for baseline LR
      - choq_logistic_params: additional params for ChoquisticRegression
      - methods: list of methods to evaluate
      - scale_data: whether to standardize input features before transformation
      - plot_folder: directory to save plots
      - results_filename: file to save experiment results
      - log_odds_bins: number of bins in log-odds histogram

    Returns:
      - final_results: dictionary with accuracy scores, coefficients, and other relevant data
    """
    # Set defaults if parameters are None
    if baseline_logistic_params is None:
        baseline_logistic_params = {'penalty': None, 'solver': 'newton-cg', 'max_iter': baseline_max_iter, 'random_state': random_state}
    if choq_logistic_params is None:
        choq_logistic_params = {'penalty': None, 'solver': 'newton-cg', 'max_iter': baseline_max_iter, 'random_state': random_state}
    
    ensure_folder(plot_folder)
    
    # 1. Load and normalize the data (raw data expected)
    X, y = modGF.func_read_data(data_imp)
    if isinstance(X, pd.DataFrame):
        X = (X - X.min())/(X.max()-X.min())
    else:
        X = (X - X.min())/(X.max()-X.min())
    
    # Containers for aggregated results
    all_sim_results = []
    all_shapley = []           # for choquet / choquet_2add (only for full choquet)
    all_interaction_matrices = []  # for choquet_2add
    all_log_odds = []          # concatenated log-odds for test set
    all_probs = []             # concatenated predicted probabilities for test set
    
    for sim in range(n_simulations):
        sim_results = {}
        print(f"\nSimulation {sim+1}/{n_simulations}")
        sim_seed = random_state + sim
        
        # 2. Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=sim_seed
        )
        
        # 3. Baseline Logistic Regression on raw features
        lr_baseline = LogisticRegression(random_state=sim_seed, max_iter=baseline_max_iter, **baseline_logistic_params)
        lr_baseline.fit(X_train, y_train)
        baseline_train_acc = lr_baseline.score(X_train, y_train)
        baseline_test_acc  = lr_baseline.score(X_test, y_test)
        print("Baseline LR Train Acc: {:.2%}, Test Acc: {:.2%}".format(baseline_train_acc, baseline_test_acc))
        sim_results['LR'] = {
            'train_acc': baseline_train_acc,
            'test_acc': baseline_test_acc,
            'coef': lr_baseline.coef_
        }
        
        # 4. Run Choquistic Models using various transformations
        for method in methods:
            print("Processing method:", method)
            # For mlm methods, adjust logistic parameters for convergence.
            if method in ["mlm", "mlm_2add"]:
                method_logistic_params = choq_logistic_params.copy()
            else:
                method_logistic_params = choq_logistic_params
            model = ChoquisticRegression(method=method, 
                                         logistic_params=method_logistic_params,
                                         scale_data=scale_data,
                                         random_state=sim_seed)
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc  = model.score(X_test, y_test)
            # For composition-based models, coefficients are stored in classifier_.coef_;
            # for inheritance-based, they are in model.coef_.
            coef = model.classifier_.coef_ if hasattr(model, "classifier_") else model.coef_
            sim_results[method] = {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'coef': coef
            }
            # Compute Shapley values only for full choquet method.
            if method in ["choquet", "choquet_2add"]:
                try:
                    shapley_vals = model.compute_shapley_values()
                    sim_results[method]['shapley'] = shapley_vals
                    all_shapley.append(shapley_vals)
                    print("Shapley values for {}: {}".format(method, shapley_vals))
                except Exception as e:
                    print("Could not compute shapley values for {}: {}".format(method, e))
            print("Method: {:12s} | Train Acc: {:.2%} | Test Acc: {:.2%}".format(method, train_acc, test_acc))
        
        # 5. Collect Interaction Effects for choquet_2add (from the model coefficients)
        model_ch2add = ChoquisticRegression(method="choquet_2add", 
                                            logistic_params=choq_logistic_params, 
                                            scale_data=scale_data, 
                                            random_state=sim_seed)
        model_ch2add.fit(X_train, y_train)
        if hasattr(model_ch2add, "classifier_"):
            coef_ch = model_ch2add.classifier_.coef_[0]
        else:
            coef_ch = model_ch2add.coef_[0]
        nAttr = X_train.shape[1]
        interaction_coef = coef_ch[nAttr:]
        interaction_matrix = np.zeros((nAttr, nAttr))
        idx = 0
        for i in range(nAttr):
            for j in range(i+1, nAttr):
                interaction_matrix[i, j] = interaction_coef[idx]
                interaction_matrix[j, i] = interaction_coef[idx]
                idx += 1
        all_interaction_matrices.append(interaction_matrix)
        
        # 6. Collect log-odds and probabilities using the model’s public API (raw X_test)
        log_odds_test = model_ch2add.decision_function(X_test)
        probs_test = model_ch2add.predict_proba(X_test)
        all_log_odds.append(log_odds_test)
        all_probs.append(probs_test)
        sim_results['choquet_2add_extra'] = {
            'log_odds_test': log_odds_test,
            'predicted_probabilities_test': probs_test
        }
        
        all_sim_results.append(sim_results)
    
    # 7. Produce Aggregate Plots
    # (A) Marginal Contributions (if any Shapley values were computed)
    if all_shapley:
        all_shapley_arr = np.vstack(all_shapley)
        mean_shapley = np.mean(all_shapley_arr, axis=0)
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            nAttr = X.shape[1]
            feature_names = [f"Feature {i}" for i in range(nAttr)]
        ordered_indices = np.argsort(mean_shapley)[::-1]
        ordered_names = np.array(feature_names)[ordered_indices]
        ordered_values = mean_shapley[ordered_indices]
        plt.figure(figsize=(10, 8))
        plt.barh(ordered_names, ordered_values, color='skyblue', edgecolor='black')
        plt.xlabel("Mean Shapley Value (Marginal Contribution)", fontsize=16)
        plt.title("Aggregate Marginal Contributions (Shapley Values)", fontsize=18)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        marginal_plot_path = os.path.join(plot_folder, "aggregate_marginal_contribution_shapley.png")
        plt.savefig(marginal_plot_path)
        plt.close()
        print("Saved aggregate marginal contribution histogram to:", marginal_plot_path)
    else:
        print("No Shapley values computed across simulations; skipping aggregate marginal contributions plot.")
    
    # (B) Interaction Effects: average interaction matrix from choquet_2add.
    if all_interaction_matrices:
        mean_interaction_matrix = np.mean(np.array(all_interaction_matrices), axis=0)
        plt.figure(figsize=(8, 6))
        plt.imshow(mean_interaction_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(orientation="vertical")
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            nAttr = X.shape[1]
            feature_names = [f"Feature {i}" for i in range(nAttr)]
        plt.xticks(range(nAttr), feature_names, rotation=90, fontsize=12)
        plt.yticks(range(nAttr), feature_names, fontsize=12)
        plt.title("Aggregate Interaction Effects (Choquet 2-additive)", fontsize=16)
        plt.tight_layout()
        interaction_plot_path = os.path.join(plot_folder, "aggregate_interaction_effects.png")
        plt.savefig(interaction_plot_path)
        plt.close()
        print("Saved aggregate interaction effects plot to:", interaction_plot_path)
    
    # (C) Log-Odds Histogram
    if all_log_odds:
        all_log_odds_concat = np.concatenate(all_log_odds)
        plt.figure(figsize=(10, 6))
        plt.hist(all_log_odds_concat, bins=log_odds_bins, color='lightgreen', edgecolor='black')
        plt.xlabel("Log-Odds", fontsize=16)
        plt.ylabel("Frequency", fontsize=16)
        plt.title("Aggregate Log-Odds Histogram (Choquet 2-additive)", fontsize=18)
        plt.tight_layout()
        log_odds_plot_path = os.path.join(plot_folder, "aggregate_log_odds_histogram.png")
        plt.savefig(log_odds_plot_path)
        plt.close()
        print("Saved aggregate log-odds histogram to:", log_odds_plot_path)
    
    # (D) Log-Odds vs. Predicted Probability Scatter
    if all_log_odds and all_probs:
        all_log_odds_concat = np.concatenate(all_log_odds)
        all_probs_concat = np.concatenate(all_probs, axis=0)
        plt.figure(figsize=(10, 6))
        plt.scatter(all_log_odds_concat, all_probs_concat[:, 1], alpha=0.7, color='coral', edgecolor='k')
        plt.xlabel("Log-Odds", fontsize=16)
        plt.ylabel("Predicted Probability (Positive Class)", fontsize=16)
        plt.title("Aggregate Log-Odds vs. Predicted Probability", fontsize=18)
        plt.tight_layout()
        log_odds_prob_plot_path = os.path.join(plot_folder, "aggregate_log_odds_vs_prob.png")
        plt.savefig(log_odds_prob_plot_path)
        plt.close()
        print("Saved aggregate log-odds vs predicted probability plot to:", log_odds_prob_plot_path)
    
    # (E) Optional: Decision Boundary Plot for 2D data (if applicable)
    def plot_decision_boundary(X, y, model, filename):
        from matplotlib.colors import ListedColormap
        X = np.array(X)
        y = np.array(y)
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
        if X.shape[1] != 2:
            print("Decision boundary plot only works for 2D data.")
            return
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Decision Boundary")
        plt.savefig(filename)
        plt.close()
        print("Saved decision boundary plot to:", filename)
    
    if np.array(X).shape[1] == 2:
        plot_decision_boundary(X_train, y_train, lr_baseline, os.path.join(plot_folder, "aggregate_decision_boundary_lr.png"))
        model_ch2add_last = ChoquisticRegression(method="choquet_2add", logistic_params=choq_logistic_params, scale_data=scale_data, random_state=random_state)
        model_ch2add_last.fit(X_train, y_train)
        plot_decision_boundary(X_train, y_train, model_ch2add_last, os.path.join(plot_folder, "aggregate_decision_boundary_choquistic.png"))
    
    final_results = {'simulations': all_sim_results}
    with open(results_filename, "wb") as f:
        pickle.dump(final_results, f)
    print("Saved overall results to", results_filename)
    
    return final_results

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle
import itertools
from sklearn.preprocessing import StandardScaler

# necessary for shapley values
os.environ["SCIPY_ARRAY_API"] = "1"

from regression_classes import ChoquisticRegression
import mod_GenFuzzyRegression as modGF
from simulation_helper_functions import get_feature_names, ensure_folder


def simulation(
    data_imp="dados_covid_sbpo_atual",
    test_size=0.2,
    random_state=0,
    n_simulations=2,
    solver_lr=None,
    penalty_lr=None,
    baseline_max_iter=None,
    baseline_logistic_params={
        "penalty": None,
        "max_iter": 1000,
        "random_state": 0,
        "solver": "newton-cg",
    },
    choq_logistic_params={
        "penalty": None,
        "max_iter": 1000,
        "random_state": 0,
        "solver": "newton-cg",
    },
    methods=["choquet_2add", "choquet", "mlm", "mlm_2add"],
    scale_data=True,
    plot_folder="plots",
    results_filename="results.pkl",
    log_odds_bins=30,
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
        # Update logistic parameters if provided
    if solver_lr is not None:
        baseline_logistic_params["solver"] = solver_lr
        choq_logistic_params["solver"] = solver_lr
    if penalty_lr is not None:
        baseline_logistic_params["penalty"] = penalty_lr
        choq_logistic_params["penalty"] = penalty_lr
    if baseline_max_iter is not None:
        baseline_logistic_params["max_iter"] = baseline_max_iter
        choq_logistic_params["max_iter"] = baseline_max_iter

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Load and (optionally) scale the entire dataset
    X, y = modGF.func_read_data(data_imp)
    if scale_data:
        X_co = StandardScaler().fit_transform(X)
    else:
        X_co = X

    # Compute the coalitions only once for the full Choquet model.
    from regression_classes import choquet_matrix
    _, coalitions_full = choquet_matrix(X_co)

    # Compute the 2-additive coalitions only once.
    nAttr = X.shape[1]
    coalitions_2add = (
    [(i,) for i in range(nAttr)] +
    [tuple(sorted(pair)) for pair in itertools.combinations(range(nAttr), 2)]
)


    # Containers for results
    all_sim_results = {"LR": [], "choquet": [], "choquet_2add": [], "mlm": [], "mlm_2add": []}
    interaction_matrices_dict = {"choquet": [], "choquet_2add": []}
    choquet_coalitions = coalitions_full  # Reuse the precomputed coalitions
    choquet_2add_coalitions = coalitions_2add  # Reuse the precomputed 2-add coalitions

    # Initialize the interaction matrices dictionary for all methods
    interaction_matrices_dict = {
        "choquet": [], 
        "choquet_2add": [], 
        "mlm": [], 
        "mlm_2add": []
    }

    # Initialize dictionaries to store Banzhaf power indices
    banzhaf_indices = {
        "mlm": [],
        "mlm_2add": []
    }

    # Containers for plotting values
    shapley_full = []
    shapley_2add = []
    marginal_full = []
    marginal_2add = []
    log_odds_decision = []
    all_probs = []

    for sim in range(n_simulations):
        sim_results = {}
        print(f"\nSimulation {sim+1}/{n_simulations}")
        sim_seed = random_state + sim

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=sim_seed
        )

        if scale_data:
            scaler = StandardScaler()
            X_train_base = scaler.fit_transform(X_train)
            X_test_base = scaler.transform(X_test)
        else:
            X_train_base = X_train
            X_test_base = X_test

        # Baseline LR
        baseline_params = baseline_logistic_params.copy()
        baseline_params["random_state"] = sim_seed
        lr_baseline = LogisticRegression(**baseline_params)
        lr_baseline.fit(X_train_base, y_train)
        baseline_train_acc = lr_baseline.score(X_train_base, y_train)
        baseline_test_acc = lr_baseline.score(X_test_base, y_test)
        print("Baseline LR Train Acc: {:.2%}, Test Acc: {:.2%}, n_iter: {}".format(
            baseline_train_acc, baseline_test_acc, lr_baseline.n_iter_
        ))
        sim_results["LR"] = {"train_acc": baseline_train_acc, "test_acc": baseline_test_acc,
                             "coef": lr_baseline.coef_, "n_iter": lr_baseline.n_iter_}

        # Process each method
        for method in methods:
            print("Processing method:", method)
            model = ChoquisticRegression(
                method=method,
                logistic_params=choq_logistic_params,
                scale_data=scale_data,
            )
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            coef = model.classifier_.coef_ if hasattr(model, "classifier_") else model.coef_
            n_iter = model.classifier_.n_iter_ if hasattr(model, "classifier_") else model.n_iter_
            sim_results[method] = {"train_acc": train_acc, "test_acc": test_acc,
                                "coef": coef, "n_iter": n_iter}
            


            # Fix the coalition printing loop by adding a safety check
            for coalition in choquet_coalitions:
                try:
                    idx = choquet_coalitions.index(coalition)
                    if idx < len(coef[0]):
                        print(f"{method} Coalition: {coalition}, coefficient: {coef[0][idx]}")
                except IndexError:
                    print(f"{method} Coalition: {coalition} - Index out of bounds")



            if method == "choquet":
                # Use the precomputed 'choquet_coalitions' directly instead of recomputing.
                try:
                    shapley_vals = model.compute_shapley_values()
                    sim_results[method]["shapley"] = shapley_vals
                    shapley_full.append(np.atleast_1d(shapley_vals))
                    marginal_vals = coef[0][:nAttr]
                    sim_results[method]["marginal"] = marginal_vals
                    marginal_full.append(np.atleast_1d(marginal_vals))
                    
                    from regression_classes import compute_choquet_interaction_matrix
                    interaction_matrix = compute_choquet_interaction_matrix(np.insert(coef[0], 0, 0.0), nAttr, choquet_coalitions)
                    interaction_matrices_dict.setdefault("choquet", []).append(interaction_matrix.copy())
                except Exception as e:
                    print(f"Could not compute full choquet values: {e}")

            if method == "choquet_2add":
                try:
                    shapley_dict = model.compute_shapley_values()
                    sim_results[method]["shapley"] = shapley_dict["shapley"]
                    sim_results[method]["marginal"] = shapley_dict["marginal"]
                    shapley_2add.append(np.atleast_1d(shapley_dict["shapley"]))
                    marginal_2add.append(np.atleast_1d(shapley_dict["marginal"]))
                except Exception as e:
                    print(f"Could not compute values for choquet 2-add: {e}")
                
                from regression_classes import compute_choquet_interaction_matrix
                interaction_matrix = compute_choquet_interaction_matrix(
                    np.insert(coef[0], 0, 0.0), nAttr, choquet_2add_coalitions, 2
                )
                interaction_matrices_dict.setdefault("choquet_2add", []).append(interaction_matrix.copy())
                
                log_odds_test = model.decision_function(X_test)
                probs_test = model.predict_proba(X_test)
                log_odds_decision.append(log_odds_test)
                all_probs.append(probs_test)
                sim_results[method].update({
                    "log_odds_test": log_odds_test,
                    "predicted_probabilities_test": probs_test,
                })

            if method == "mlm":
                # For the MLM model
                from regression_classes import compute_mlm_interaction_matrix
                interaction_matrix = compute_mlm_interaction_matrix(np.insert(coef[0], 0, 0.0), nAttr, choquet_coalitions)
                interaction_matrices_dict.setdefault("mlm", []).append(interaction_matrix.copy())

                # For the full MLM model
                from regression_classes import compute_mlm_interaction_matrix, compute_banzhaf_power_indices
                
                # Get model coefficients (v values)
                v = np.insert(coef[0], 0, 0.0)  # Insert 0 for empty set
                
                # Compute interaction matrix using Banzhaf formula
                interaction_matrix = compute_mlm_interaction_matrix(v, nAttr, choquet_coalitions)
                interaction_matrices_dict["mlm"].append(interaction_matrix.copy())
                
                # Compute Banzhaf power indices
                power_indices = compute_banzhaf_power_indices(v, nAttr, choquet_coalitions)
                banzhaf_indices["mlm"].append(power_indices)
                
                # Store in simulation results
                sim_results[method]["banzhaf_indices"] = power_indices
                sim_results[method]["interaction_matrix"] = interaction_matrix


            if method == "mlm_2add":
                # For the MLM 2-add model
                from regression_classes import compute_mlm_interaction_matrix
                print(choquet_2add_coalitions)
                interaction_matrix = compute_mlm_interaction_matrix(np.insert(coef[0], 0, 0.0), nAttr, choquet_2add_coalitions,2)
                interaction_matrices_dict.setdefault("mlm_2add", []).append(interaction_matrix.copy())
                
                """
                # For the MLM 2-add model, similarly extract interaction terms.
                nAttr = X_train.shape[1]
                coef_ch = coef[0]
                interaction_coef = coef_ch[nAttr:]
                interaction_matrix = np.zeros((nAttr, nAttr))
                idx = 0
                for i in range(nAttr):
                    for j in range(i+1, nAttr):
                        interaction_matrix[i, j] = interaction_coef[idx]
                        interaction_matrix[j, i] = interaction_coef[idx]
                        idx += 1
                interaction_matrices_dict.setdefault("mlm_2add", []).append(interaction_matrix.copy())
                """
                # For the 2-additive MLM model
                from regression_classes import compute_mlm_interaction_matrix, compute_banzhaf_power_indices
                
                # Get model coefficients (v values)
                v = np.insert(coef[0], 0, 0.0)  # Insert 0 for empty set
                
                # Compute interaction matrix using Banzhaf formula for 2-additive model
                interaction_matrix = compute_mlm_interaction_matrix(v, nAttr, choquet_2add_coalitions, 2)
                interaction_matrices_dict["mlm_2add"].append(interaction_matrix.copy())
                
                # Compute Banzhaf power indices for 2-additive model
                power_indices = compute_banzhaf_power_indices(v, nAttr, choquet_2add_coalitions, 2)
                banzhaf_indices["mlm_2add"].append(power_indices)
                
                # Store in simulation results
                sim_results[method]["banzhaf_indices"] = power_indices
                sim_results[method]["interaction_matrix"] = interaction_matrix


        # Append the results from this simulation
        for key, result in sim_results.items():
            all_sim_results[key].append(result)

    # ---------------- Plotting ----------------
    from plotting_functions import (
        plot_shapley_full,
        plot_coef_full,
        plot_coef_2add,
        plot_shapley_2add,
        plot_marginal_2add,
        plot_interaction_matrix,
        plot_log_odds_hist,
        plot_log_odds_vs_prob,
        plot_shapley_vs_interaction,
        plot_test_accuracy,
        plot_decision_boundary,
        plot_overall_interaction
    )
    from os.path import join

    feature_names = get_feature_names(X)

    plot_shapley_full(feature_names, shapley_full, plot_folder)

    avg_coef_full = np.mean(np.vstack([sim["coef"] for sim in all_sim_results["choquet"] if "coef" in sim]), axis=0)
    plot_coef_full(X, avg_coef_full, plot_folder)

    avg_coef_2add = np.mean(np.vstack([sim["coef"] for sim in all_sim_results["choquet_2add"] if "coef" in sim]), axis=0)
    plot_coef_2add(feature_names, avg_coef_2add, plot_folder)

    plot_shapley_2add(feature_names, shapley_2add, plot_folder)
    plot_marginal_2add(feature_names, marginal_2add, plot_folder)
    # Plot the choquet_2add interaction matrices using X for correct axis labeling
    plot_interaction_matrix(X, feature_names, interaction_matrices_dict["choquet_2add"], plot_folder, method="choquet_2add")

    # Plot the full choquet interaction matrices (Banzhaf interaction) using X as well
    plot_interaction_matrix(X, feature_names, interaction_matrices_dict["choquet"], plot_folder, method="choquet")

    # Plot the MLM interaction matrices
    plot_interaction_matrix(X, feature_names, interaction_matrices_dict["mlm"], plot_folder, method="mlm")

    # Plot the MLM 2-add interaction matrices
    plot_interaction_matrix(X, feature_names, interaction_matrices_dict["mlm_2add"], plot_folder, method="mlm_2add")

    plot_log_odds_hist(log_odds_decision, log_odds_bins, plot_folder)
    plot_log_odds_vs_prob(log_odds_decision, all_probs, plot_folder)

    # In the plotting section:
    # New function to plot Banzhaf power indices
    def plot_banzhaf_indices(feature_names, banzhaf_indices, plot_folder, method):
        """
        Plot Banzhaf power indices for features.
        
        Parameters:
        -----------
        feature_names : list
            Names of the features
        banzhaf_indices : list of numpy.ndarray
            List of Banzhaf indices for each simulation
        plot_folder : str
            Directory to save the plot
        method : str
            Method name for title and filename
        """
        plt.figure(figsize=(10, 6))
        
        # Average over simulations
        avg_indices = np.mean(np.vstack(banzhaf_indices), axis=0)
        
        # Sort indices by magnitude for better visualization
        sorted_idx = np.argsort(avg_indices)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_indices = avg_indices[sorted_idx]
        
        plt.barh(range(len(sorted_features)), sorted_indices, align='center')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Banzhaf Power Index')
        plt.title(f'Banzhaf Power Indices - {method}')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(plot_folder, f'banzhaf_indices_{method}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # After the other plotting calls
    if "mlm" in methods and banzhaf_indices["mlm"]:
        plot_banzhaf_indices(feature_names, banzhaf_indices["mlm"], plot_folder, method="mlm")
        
    if "mlm_2add" in methods and banzhaf_indices["mlm_2add"]:
        plot_banzhaf_indices(feature_names, banzhaf_indices["mlm_2add"], plot_folder, method="mlm_2add")


    def plot_interaction_comparison(feature_names, choquet_interaction, mlm_interaction, plot_folder, method_suffix="2add"):
        """
        Compare interaction matrices between Choquet and MLM models.
        
        Parameters:
        -----------
        feature_names : list
            Names of the features
        choquet_interaction : list of numpy.ndarray
            List of interaction matrices from Choquet model
        mlm_interaction : list of numpy.ndarray
            List of interaction matrices from MLM model
        plot_folder : str
            Directory to save the plot
        method_suffix : str
            Suffix for method identification (e.g., "2add" for 2-additive models)
        """
        # Average over simulations
        avg_choquet = np.mean(np.array(choquet_interaction), axis=0)
        avg_mlm = np.mean(np.array(mlm_interaction), axis=0)
        
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot Choquet interaction
        im1 = axes[0].imshow(avg_choquet, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0].set_title(f'Choquet {method_suffix} Interaction')
        axes[0].set_xticks(np.arange(len(feature_names)))
        axes[0].set_yticks(np.arange(len(feature_names)))
        axes[0].set_xticklabels(feature_names, rotation=90)
        axes[0].set_yticklabels(feature_names)
        
        # Plot MLM interaction
        im2 = axes[1].imshow(avg_mlm, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1].set_title(f'MLM {method_suffix} Interaction')
        axes[1].set_xticks(np.arange(len(feature_names)))
        axes[1].set_yticks(np.arange(len(feature_names)))
        axes[1].set_xticklabels(feature_names, rotation=90)
        axes[1].set_yticklabels(feature_names)
        
        # Add colorbar
        fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f'interaction_comparison_{method_suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Add to simulation_loop.py plotting section:
    if "choquet" in methods and "mlm" in methods:
        if interaction_matrices_dict["choquet"] and interaction_matrices_dict["mlm"]:
            plot_interaction_comparison(feature_names, 
                                    interaction_matrices_dict["choquet"], 
                                    interaction_matrices_dict["mlm"], 
                                    plot_folder, 
                                    "full")

    if "choquet_2add" in methods and "mlm_2add" in methods:
        if interaction_matrices_dict["choquet_2add"] and interaction_matrices_dict["mlm_2add"]:
            plot_interaction_comparison(feature_names, 
                                    interaction_matrices_dict["choquet_2add"], 
                                    interaction_matrices_dict["mlm_2add"], 
                                    plot_folder, 
                                    "2add")

    # We'll implement three methods for the full Choquet model.
    nAttr = X_train.shape[1]
    if shapley_full and marginal_full and "choquet" in interaction_matrices_dict and len(interaction_matrices_dict["choquet"]) > 0:
        # Average over simulations for the full Choquet method:
        avg_shapley_full = np.mean(np.vstack(shapley_full), axis=0)
        avg_marginal_full = np.mean(np.vstack(marginal_full), axis=0)
        avg_interaction_matrix_full = np.mean(np.array(interaction_matrices_dict["choquet"]), axis=0)
        
        # Method 1: From the interaction matrix
        overall_method1 = 0.5 * np.sum(avg_interaction_matrix_full, axis=1)
        # Method 2: Difference between Shapley and marginal values
        overall_method2 = avg_shapley_full - avg_marginal_full
        # Method 3: Average of pairwise interaction indices computed with shapley_interaction_index
        from regression_classes import shapley_interaction_index
        nAttr = X_train.shape[1]
        overall_method3 = np.zeros(nAttr)
        # Compute the average game parameter vector (v) from the full Choquet model over simulations.
        # Here we assume that for each simulation in the full choquet branch, sim["coef"][0] contains v.
        avg_v_full = np.mean(np.vstack([sim["coef"][0] for sim in all_sim_results["choquet"] if "coef" in sim]), axis=0)

        # Use the coalitions from the full Choquet model (saved as choquet_coalitions)
        for j in range(nAttr):
            pairwise_vals = []
            for k in range(nAttr):
                if k != j:
                    # Compute the Shapley interaction index for the pair (j, k)
                    idx_val = shapley_interaction_index((j, k), choquet_coalitions, avg_v_full, m=nAttr)
                    pairwise_vals.append(idx_val)
            overall_method3[j] = np.mean(pairwise_vals)

        
        # Plot the overall interaction indices for all three methods:
        feature_names = get_feature_names(X)
        overall_dict = {
    "Matrix Method": overall_method1,
    "Shapley - Marginal": overall_method2,
    "Average Pairwise": overall_method3
}
        plot_overall_interaction(feature_names, overall_dict, "Overall Interaction Comparison", plot_folder)





    """
    # Plot the Shapley values vs. interaction effect for the full Choquet model
    from simulation_helper_functions import weighted_full_interaction_effect
    m = X_train.shape[1]
    indices = weighted_full_interaction_effect(avg_coef_full, all_coalitions, m)
    plot_shapley_vs_interaction(feature_names, shapley_full, indices, plot_folder, method="choquet")

    # Plot Shapelies vs. interaction effect for the 2-additive Choquet model
    from simulation_helper_functions import weighted_pairwise_interaction_effect
    indices = weighted_pairwise_interaction_effect(avg_coef_2add, interaction_matrix, m)
    plot_shapley_vs_interaction(feature_names, shapley_2add, indices, plot_folder, method="choquet_2add")

    
    # Plot ISR for both choquet and choquet_2add

    # Get the coalitions for choquet
    if scale_data:
        X_co = StandardScaler().fit_transform(X_train)
    else:
        X_co = X_train
    _, all_coalitions = choquet_matrix(X_co)
    choquet_coalitions = [sorted(coal) for coal in all_coalitions]

    # Get the coalitions for choquet_2add
    nAttr = X_train.shape[1]
    all_coalitions_2add = [()] + [(i,) for i in range(nAttr)] + [tuple(sorted(pair)) for pair in itertools.combinations(range(nAttr), 2)]
    choquet_2add_coalitions = [sorted(coal) for coal in all_coalitions_2add]

    # For full choquet shapley values:
    if shapley_full and len(shapley_full) > 0:
        avg_shapley_full = np.mean(np.vstack(shapley_full), axis=0)
    else:
        nAttr = X_train.shape[1]
        avg_shapley_full = np.zeros(nAttr)  # Default to zeros if no values

    # For choquet 2-add shapley values:
    if shapley_2add and len(shapley_2add) > 0:
        avg_shapley_2add = np.mean(np.vstack(shapley_2add), axis=0)
    else:
        nAttr = X_train.shape[1]
        avg_shapley_2add = np.zeros(nAttr)  # Default to zeros if no values

    # Compute ISR for both models
    from simulation_helper_functions import  interaction_shapley_ratio
    isr_full = interaction_shapley_ratio(avg_shapley_full, interaction_matrices_dict["choquet"], choquet_coalitions, m=nAttr)
    isr_2add = interaction_shapley_ratio(avg_shapley_2add, interaction_matrices_dict["choquet_2add"], choquet_2add_coalitions, m=nAttr)
    plot_shapley_vs_interaction(feature_names, isr_full, isr_2add, plot_folder, method="ISR")
    """
    

    model_names = sorted(all_sim_results.keys())
    plot_test_accuracy(model_names, all_sim_results, plot_folder)
    # print accuracies
    for model_name in model_names:
        accs = [sim["test_acc"] for sim in all_sim_results[model_name]]
        print(f"Model: {model_name}, Test Acc: {np.mean(accs):.2%} ± {np.std(accs):.2%}")

    if np.array(X).shape[1] == 2:
        plot_decision_boundary(X_train, y_train, lr_baseline, join(plot_folder, "boundary_lr.png"))
        model_ch2add_last = ChoquisticRegression(
            method="choquet_2add",
            logistic_params=choq_logistic_params,
            scale_data=scale_data,
            random_state=random_state,
        )
        model_ch2add_last.fit(X_train, y_train)
        plot_decision_boundary(X_train, y_train, model_ch2add_last, join(plot_folder, "boundary_choq.png"))

    final_results = {"simulations": all_sim_results}
    """
    with open(results_filename, "wb") as f:
        pickle.dump(final_results, f)
    print("Saved overall results to", results_filename)
    """
    return final_results

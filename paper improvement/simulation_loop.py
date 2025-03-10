import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle
import itertools
from sklearn.preprocessing import MinMaxScaler

# necessary for shapley values
os.environ["SCIPY_ARRAY_API"] = "1"

from regression_classes import (
    ChoquisticRegression,
    ChoquetTransformer, 
    compute_shapley_values,
    compute_banzhaf_power_indices, 
    compute_choquet_interaction_matrix,
    compute_banzhaf_interaction_matrix,
    compute_shapley_interaction_index,
    compute_banzhaf_interaction_index,
    compute_model_interactions
)
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

    ensure_folder(plot_folder)

    # Load and scale the entire dataset (always use MinMaxScaler)
    X, y = modGF.func_read_data(data_imp)
    if scale_data:
        X_co = MinMaxScaler().fit_transform(X)
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
    
    # Initialize the interaction matrices dictionary for all methods
    interaction_matrices_dict = {
        "choquet": [], 
        "choquet_2add": [], 
        "mlm": [], 
        "mlm_2add": []
    }

    # Initialize dictionaries to store power indices
    power_indices_dict = {
        "choquet": {"shapley": []}, 
        "choquet_2add": {"shapley": []},
        "mlm": {"banzhaf": []},
        "mlm_2add": {"banzhaf": []}
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
            scaler = MinMaxScaler()
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

        # Store transformed matrices for verification
        transformed_matrices = {}
        
        # Process each method
        for method in methods:
            print("Processing method:", method)
            k_add = 2 if method.endswith("_2add") else None
            choq_params = choq_logistic_params.copy()
            choq_params["random_state"] = sim_seed
            
            # For verification/debugging purposes
            if sim == 0 and len(X_train) > 0:
                transformer = ChoquetTransformer(
                    method=method, 
                    k_add=k_add,
                    scale_data=scale_data
                )
                X_sample = X_train[:min(3, len(X_train))]
                transformed = transformer.fit_transform(X_sample)
                transformed_matrices[method] = transformed
                
                print(f"  {method} transformation shape: {transformed.shape}")
                if transformed.shape[1] > 0:
                    print(f"  {method} first row sample: {transformed[0, :5]}")
                    non_zeros = np.count_nonzero(transformed[0])
                    print(f"  {method} non-zeros: {non_zeros}/{transformed.shape[1]} ({non_zeros/transformed.shape[1]:.2%})")
            
            # Create model with unified scaling approach
            model = ChoquisticRegression(
                method=method,
                k_add=k_add, 
                scale_data=scale_data,
                **choq_params
            )
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            coef = model.coef_
            n_iter = model.n_iter_
            
            print(f"  {method} Test Acc: {test_acc:.4f}")
            
            # Create a deep copy of the results to ensure no reference sharing
            import copy
            sim_results[method] = copy.deepcopy({
                "train_acc": train_acc, 
                "test_acc": test_acc,
                "coef": coef, 
                "n_iter": n_iter
            })
        
        # If we collected transformed matrices, verify they're different
        if sim == 0 and len(transformed_matrices) >= 2:
            methods_list = list(transformed_matrices.keys())
            for i in range(len(methods_list)):
                for j in range(i+1, len(methods_list)):
                    method1 = methods_list[i]
                    method2 = methods_list[j]
                    mat1 = transformed_matrices[method1]
                    mat2 = transformed_matrices[method2]
                    
                    if mat1.shape == mat2.shape:
                        # Check if matrices are identical
                        is_identical = np.allclose(mat1, mat2)
                        print(f"VERIFICATION: {method1} and {method2} transformations are {'IDENTICAL' if is_identical else 'DIFFERENT'}")
                        if is_identical:
                            print("WARNING: This could explain identical accuracy results!")

            # Get the appropriate coalitions based on method
            if method.endswith("_2add"):
                all_coalitions = coalitions_2add
                k_value = 2
            else:
                all_coalitions = coalitions_full
                k_value = None

            # Extract capacity values from the model (with 0 for empty set)
            v = np.insert(coef[0], 0, 0.0) 

            # Compute interpretability measures based on method type
            if method.startswith("choquet"):
                # Compute Shapley values and store them
                shapley_values = compute_shapley_values(v, nAttr, all_coalitions, k=k_value)
                power_indices_dict[method]["shapley"].append(shapley_values)
                sim_results[method]["shapley"] = shapley_values
                
                # Store Shapley values for plotting
                if method == "choquet":
                    shapley_full.append(shapley_values)
                    # For full Choquet, store direct coefficients as "marginal" values
                    marginal_full.append(coef[0][:nAttr])
                    sim_results[method]["marginal"] = coef[0][:nAttr]
                elif method == "choquet_2add":
                    # Compute Shapley values using model's method which returns a dictionary
                    shapley_result = model.compute_shapley_values()
                    
                    if isinstance(shapley_result, dict) and "shapley" in shapley_result:
                        # Store the actual Shapley values from the dictionary
                        shapley_values = shapley_result["shapley"]
                        marginal_values = shapley_result["marginal"]
                    else:
                        # Fall back to using the global function if needed
                        shapley_values = compute_shapley_values(v, nAttr, all_coalitions, k=k_value)
                        marginal_values = coef[0][:nAttr]

                    # Store values appropriately
                    shapley_2add.append(shapley_values)
                    marginal_2add.append(marginal_values)
                    sim_results[method]["shapley"] = shapley_values
                    sim_results[method]["marginal"] = marginal_values
                    
                    # For visualization purposes, we can also compute normalized/absolute values
                    # This helps in understanding feature importance regardless of direction
                    abs_shapley = np.abs(shapley_values)
                    sim_results[method]["abs_shapley"] = abs_shapley
                    
                # Compute interaction matrix
                interaction_matrix = compute_choquet_interaction_matrix(v, nAttr, all_coalitions, k=k_value)
                interaction_matrices_dict[method].append(interaction_matrix)
                sim_results[method]["interaction_matrix"] = interaction_matrix
                
            elif method.startswith("mlm"):
                # Compute Banzhaf power indices
                banzhaf_values = compute_banzhaf_power_indices(v, nAttr, all_coalitions, k=k_value)
                power_indices_dict[method]["banzhaf"].append(banzhaf_values)
                sim_results[method]["banzhaf"] = banzhaf_values
                
                # Compute Banzhaf interaction matrix
                interaction_matrix = compute_banzhaf_interaction_matrix(v, nAttr, all_coalitions, k=k_value)
                interaction_matrices_dict[method].append(interaction_matrix)
                sim_results[method]["interaction_matrix"] = interaction_matrix
                
            # For choquet_2add, also collect log odds and probabilities
            if method == "choquet_2add":
                log_odds_test = model.decision_function(X_test)
                probs_test = model.predict_proba(X_test)
                log_odds_decision.append(log_odds_test)
                all_probs.append(probs_test)
                sim_results[method].update({
                    "log_odds_test": log_odds_test,
                    "predicted_probabilities_test": probs_test,
                })

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
        plot_overall_interaction,
        plot_interaction_comparison
    )
    from os.path import join

    feature_names = get_feature_names(X)

    # Plot Shapley values for full Choquet model
    if shapley_full:
        plot_shapley_full(feature_names, shapley_full, plot_folder)

    # Plot coefficients for full Choquet model
    if "choquet" in methods and all_sim_results["choquet"]:
        coef_values = [sim["coef"] for sim in all_sim_results["choquet"] if "coef" in sim]
        if coef_values:
            print("Plotting coefficients for full Choquet model")
            avg_coef_full = np.mean(np.vstack(coef_values), axis=0)
            plot_coef_full(X, avg_coef_full, plot_folder)

    # Plot coefficients for 2-additive Choquet model
    if "choquet_2add" in methods and all_sim_results["choquet_2add"]:
        print("Plotting coefficients for 2-additive Choquet model")
        coef_values = [sim["coef"] for sim in all_sim_results["choquet_2add"] if "coef" in sim]
        if coef_values:
            avg_coef_2add = np.mean(np.vstack(coef_values), axis=0)
            plot_coef_2add(feature_names, avg_coef_2add, plot_folder)

    # Plot Shapley and marginal values for 2-additive model
    if shapley_2add:
        plot_shapley_2add(feature_names, shapley_2add, plot_folder)
    if marginal_2add:
        plot_marginal_2add(feature_names, marginal_2add, plot_folder)

    # Plot interaction matrices for all models
    for method in methods:
        if method in interaction_matrices_dict and interaction_matrices_dict[method]:
            plot_interaction_matrix(X, feature_names, interaction_matrices_dict[method], plot_folder, method=method)

    # Plot log-odds and probabilities
    if log_odds_decision:
        plot_log_odds_hist(log_odds_decision, log_odds_bins, plot_folder)
        plot_log_odds_vs_prob(log_odds_decision, all_probs, plot_folder)

    # Plot Banzhaf power indices 
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

    # Plot Banzhaf power indices for MLM models
    for method in ["mlm", "mlm_2add"]:
        if method in methods and method in power_indices_dict and "banzhaf" in power_indices_dict[method]:
            indices = power_indices_dict[method]["banzhaf"]
            if indices:
                plot_banzhaf_indices(feature_names, indices, plot_folder, method=method)

    # Compare interaction matrices between Choquet and MLM models
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

    # Compare interaction methods for the full Choquet model
    if shapley_full and marginal_full and "choquet" in interaction_matrices_dict and interaction_matrices_dict["choquet"]:
        # Average over simulations 
        avg_shapley_full = np.mean(np.vstack(shapley_full), axis=0)
        avg_marginal_full = np.mean(np.vstack(marginal_full), axis=0)
        avg_interaction_matrix_full = np.mean(np.array(interaction_matrices_dict["choquet"]), axis=0)
        
        # Method 1: From the interaction matrix - sum rows and divide by 2
        overall_method1 = 0.5 * np.sum(avg_interaction_matrix_full, axis=1)
        
        # Method 2: Difference between Shapley and marginal values
        overall_method2 = avg_shapley_full - avg_marginal_full
        
        # Method 3: Average of pairwise interaction indices computed with compute_shapley_interaction_index
        overall_method3 = np.zeros(nAttr)
        
        # Compute the average game parameter vector (v)
        avg_v_full = np.mean(np.vstack([sim["coef"][0] for sim in all_sim_results["choquet"] if "coef" in sim]), axis=0)
        avg_v_full = np.insert(avg_v_full, 0, 0.0)  # Insert 0 for empty set
        
        # Calculate average pairwise Shapley interaction indices
        for j in range(nAttr):
            pairwise_vals = []
            for k in range(nAttr):
                if k != j:
                    # Compute the Shapley interaction index for the pair (j, k)
                    idx_val = compute_shapley_interaction_index(avg_v_full, nAttr, coalitions_full, (j, k))
                    pairwise_vals.append(idx_val)
            overall_method3[j] = np.mean(pairwise_vals) if pairwise_vals else 0.0
        
        # Plot the overall interaction indices for all three methods
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
    
    # Print accuracy statistics
    for model_name in model_names:
        accs = [sim["test_acc"] for sim in all_sim_results[model_name]]
        print(f"Model: {model_name}, Test Acc: {np.mean(accs):.2%} ± {np.std(accs):.2%}")

    # Plot decision boundary for 2D data
    if X.shape[1] == 2:
        plot_decision_boundary(X_train, y_train, lr_baseline, join(plot_folder, "boundary_lr.png"))
        
        if "choquet_2add" in methods:
            # Create a model for visualization
            model_ch2add = ChoquisticRegression(
                method="choquet_2add",
                k_add=2,
                scale_data=scale_data,
                **choq_logistic_params
            )
            model_ch2add.fit(X_train, y_train)
            plot_decision_boundary(X_train, y_train, model_ch2add, join(plot_folder, "boundary_choq.png"))

    # Package final results
    final_results = {"simulations": all_sim_results}
    """
    if os.path.dirname(results_filename):
        ensure_folder(os.path.dirname(results_filename))
    
    with open(results_filename, "wb") as f:
        pickle.dump(final_results, f)
    print("Saved overall results to", results_filename)"
    """
    
    return final_results

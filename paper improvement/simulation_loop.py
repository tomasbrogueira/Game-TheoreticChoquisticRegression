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
from complexity_functions import (
    compute_model_complexities,
    plot_complexity_results,
    analyze_scaling_behavior,
    plot_scaling_behavior,
    measure_model_energy_and_flops
)
import mod_GenFuzzyRegression as modGF
from simulation_helper_functions import get_feature_names, ensure_folder, compare_transformations, extract_k_value
from robustness import test_model_robustness, compare_regularization_robustness

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

    X, y = modGF.func_read_data(data_imp)
    
    feature_names = get_feature_names(X)
    
    if scale_data:
        X_co = MinMaxScaler().fit_transform(X)
    else:
        X_co = X

    from regression_classes import choquet_matrix
    _, coalitions_full = choquet_matrix(X_co)

    k_add_coalitions = {}
    nAttr = X.shape[1]
    
    all_sim_results = {}
    for method in ["LR"] + methods:
        all_sim_results[method] = []
    
    interpretability_data = {
        method: {
            "power_indices": [],        # Shapley for choquet, Banzhaf for MLM
            "marginal_values": [],      # Direct singleton contributions
            "interaction_matrix": [],   # Pairwise interaction matrices
            "log_odds": [],             # For visualization
            "predicted_probs": []       # For visualization
        } for method in methods
    }

    # Prepare results for complexity analysis
    complexity_results = {
        sim: {
        } for sim in range(n_simulations)
    }
    
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
        sim_results["LR"] = {
            "train_acc": baseline_train_acc, 
            "test_acc": baseline_test_acc,
            "coef": lr_baseline.coef_, 
            "intercept": lr_baseline.intercept_[0] if hasattr(lr_baseline, 'intercept_') else None,
            "n_iter": lr_baseline.n_iter_
        }
        

        # calculate complexity metrics for LR
        complexity_results[sim].update(compute_model_complexities(
            [lr_baseline], X_train_base, y_train, X_test_base, labels=["LR"]
        ))

        # FLOPS and energy consumption
        flops, energy, runtime, is_estimated = measure_model_energy_and_flops(
            lr_baseline, X_test_base
        )
        complexity_results[sim]["LR"].update({
            "flops": flops,
            "energy_uj": energy,
            "pred_runtime": runtime,
            "is_estimated": is_estimated
        })

        # Process each method
        for method in methods:
            print("Processing method:", method)
            k_add = extract_k_value(method)
            choq_params = choq_logistic_params.copy()
            choq_params["random_state"] = sim_seed

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
            
            print(f"  {method} Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

            sim_results[method] = {
                "train_acc": train_acc, 
                "test_acc": test_acc,
                "coef": coef.copy(), 
                "intercept": model.intercept_[0] if hasattr(model, 'intercept_') else None,
                "n_iter": n_iter
            }

            # calculate complexity metrics for the model
            complexity_results[sim].update(
                compute_model_complexities([model], X_train, y_train, X_test, labels=[method])
            )

            # Add FLOPS and energy consumption for this method
            flops, energy, runtime, is_estimated = measure_model_energy_and_flops(
                model, X_test
            )
            # Store the energy and FLOPS results in the complexity dictionary
            complexity_results[sim][method].update({
                "flops": flops,
                "energy_uj": energy, 
                "pred_runtime": runtime,
                "is_estimated": is_estimated
            })
            
            # Get the appropriate coalitions based on method
            if k_add is not None:
                if k_add not in k_add_coalitions:
                    k_add_coalitions[k_add] = []
                    k_add_coalitions[k_add].extend([(i,) for i in range(nAttr)])
                    for size in range(2, k_add + 1):
                        k_add_coalitions[k_add].extend([tuple(sorted(combo)) for combo in itertools.combinations(range(nAttr), size)])
                
                all_coalitions = k_add_coalitions[k_add]
            else:
                all_coalitions = coalitions_full

            v = np.insert(coef[0], 0, 0.0)
            
            if method.startswith("choquet"):
                try:
                    shapley_result = model.compute_shapley_values()
                    
                    if isinstance(shapley_result, dict) and "shapley" in shapley_result:
                        shapley_values = shapley_result["shapley"]
                        marginal_values = shapley_result["marginal"]
                    else:
                        raise AttributeError("Invalid format from compute_shapley_values()")
                except (AttributeError, NotImplementedError):
                    shapley_values = compute_shapley_values(v, nAttr, all_coalitions, k=k_add)
                    marginal_values = np.zeros(nAttr)
                    for i in range(nAttr):
                        try:
                            singleton_idx = all_coalitions.index((i,))
                            # v[0] is empty set, v[idx+1] corresponds to coalition at all_coalitions[idx]
                            marginal_values[i] = v[singleton_idx+1]
                        except ValueError:
                            marginal_values[i] = coef[0][i]
                
                interpretability_data[method]["power_indices"].append(shapley_values)
                interpretability_data[method]["marginal_values"].append(marginal_values)
                
                interaction_matrix = compute_choquet_interaction_matrix(v, nAttr, all_coalitions, k=k_add)
                interpretability_data[method]["interaction_matrix"].append(interaction_matrix)
                
            elif method.startswith("mlm"):
                banzhaf_values = compute_banzhaf_power_indices(v, nAttr, all_coalitions, k=k_add)
                interpretability_data[method]["power_indices"].append(banzhaf_values)
                
                marginal_values = np.zeros(nAttr)
                for i in range(nAttr):
                    try:
                        singleton_idx = all_coalitions.index((i,))
                        # v[0] is empty set, v[idx+1] corresponds to coalition at all_coalitions[idx]
                        marginal_values[i] = v[singleton_idx+1]
                    except ValueError:
                        marginal_values[i] = coef[0][i]
                        
                interpretability_data[method]["marginal_values"].append(marginal_values)
                
                interaction_matrix = compute_banzhaf_interaction_matrix(v, nAttr, all_coalitions, k=k_add)
                interpretability_data[method]["interaction_matrix"].append(interaction_matrix)
            
            # Collect log odds and probabilities for visualization
            log_odds_test = model.decision_function(X_test)
            probs_test = model.predict_proba(X_test)
            interpretability_data[method]["log_odds"].append(log_odds_test)
            interpretability_data[method]["predicted_probs"].append(probs_test)

        # Append the results from this simulation
        for key, result in sim_results.items():
            all_sim_results[key].append(result)



    # ---------------- Analysis ----------------

    final_results = {
        "simulations": {},
        "interpretability": {},
        "robustness": {}
    }

    # --------------- Robustness Testing ----------------
    print("\nPerforming robustness analysis...")

    robustness_models_none = {}
    robustness_models_l2 = {}

    # 1. No regularization
    baseline_params_none = baseline_logistic_params.copy()
    baseline_params_none["random_state"] = random_state
    baseline_params_none["penalty"] = None
    lr_robustness_none = LogisticRegression(**baseline_params_none)
    lr_robustness_none.fit(X_train, y_train)
    robustness_models_none["LR"] = lr_robustness_none

    # 2. L2 regularization
    baseline_params_l2 = baseline_logistic_params.copy()
    baseline_params_l2["random_state"] = random_state
    baseline_params_l2["penalty"] = "l2"
    baseline_params_l2["C"] = 1.0  # Standard L2 strength
    lr_robustness_l2 = LogisticRegression(**baseline_params_l2)
    lr_robustness_l2.fit(X_train, y_train)
    robustness_models_l2["LR"] = lr_robustness_l2

    # Train one model of each type for robustness testing (both unregularized and L2)
    for method in methods:
        print(f"Training {method} models for robustness testing (None and L2)...")
        k_add = extract_k_value(method)
        
        # 1. No regularization
        choq_params_none = choq_logistic_params.copy()
        choq_params_none["random_state"] = random_state
        choq_params_none["penalty"] = None
        
        model_none = ChoquisticRegression(
            method=method,
            k_add=k_add,
            scale_data=scale_data,
            **choq_params_none
        )
        model_none.fit(X_train, y_train)
        robustness_models_none[method] = model_none
        
        # 2. L2 regularization
        choq_params_l2 = choq_logistic_params.copy()
        choq_params_l2["random_state"] = random_state
        choq_params_l2["penalty"] = "l2"
        choq_params_l2["C"] = 1.0  # Standard L2 strength
        
        model_l2 = ChoquisticRegression(
            method=method,
            k_add=k_add,
            scale_data=scale_data,
            **choq_params_l2
        )
        model_l2.fit(X_train, y_train)
        robustness_models_l2[method] = model_l2

    output_folder_none = os.path.join(plot_folder, "robustness", "None")
    output_folder_l2 = os.path.join(plot_folder, "robustness", "L2")

    print("\nTesting robustness of models WITHOUT regularization...")
    robustness_results_none = test_model_robustness(
        models=robustness_models_none,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        noise_levels=[0.05, 0.1, 0.2, 0.3, 0.5],
        n_permutations=10,
        feature_dropout_count=max(1, X.shape[1] // 3),
        n_bootstrap_samples=100,
        bootstrap_size=0.8,
        output_folder=output_folder_none,
        random_state=random_state
    )

    print("\nTesting robustness of models WITH L2 regularization...")
    robustness_results_l2 = test_model_robustness(
        models=robustness_models_l2,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        noise_levels=[0.05, 0.1, 0.2, 0.3, 0.5],
        n_permutations=10,
        feature_dropout_count=max(1, X.shape[1] // 3),
        n_bootstrap_samples=100,
        bootstrap_size=0.8,
        output_folder=output_folder_l2,
        random_state=random_state
    )

    robustness_results = {
        "None": robustness_results_none,
        "L2": robustness_results_l2
    }

    final_results["robustness"] = robustness_results

    comparison_folder = os.path.join(plot_folder, "robustness", "comparison")

    compare_regularization_robustness(
        robustness_results_none,
        robustness_results_l2,
        comparison_folder
    )

    print("Robustness comparison completed. See results in:", comparison_folder)


    # ---------------- Plotting ----------------
    from plotting_functions import (
        plot_shapley_full,
        plot_coef,
        plot_shapley_2add,
        plot_marginal_2add,
        plot_interaction_matrix,
        plot_log_odds_hist,
        plot_log_odds_vs_prob,
        plot_shapley_vs_interaction,
        plot_test_accuracy,
        plot_decision_boundary,
        plot_overall_interaction,
        plot_interaction_comparison,
    )
    from os.path import join
    from simulation_helper_functions import (
        overall_interaction_index,
        overall_interaction_index_abs,
        overall_interaction_from_shapley
    )

    # Plotting section - prepare data first then plot
    
    # 1. Extract aggregated interpretability data for plotting
    plot_data = {}
    
    for method in methods:
        plot_data[method] = {}
        
        # Average power indices (Shapley for choquet, Banzhaf for MLM)
        if interpretability_data[method]["power_indices"]:
            power_indices_array = np.vstack(interpretability_data[method]["power_indices"])
            plot_data[method]["power_indices_avg"] = np.mean(power_indices_array, axis=0)
            plot_data[method]["power_indices_std"] = np.std(power_indices_array, axis=0)
        
        # Average marginal values
        if interpretability_data[method]["marginal_values"]:
            marginal_array = np.vstack(interpretability_data[method]["marginal_values"])
            plot_data[method]["marginal_avg"] = np.mean(marginal_array, axis=0)
            plot_data[method]["marginal_std"] = np.std(marginal_array, axis=0)
        
        # Average interaction matrix
        if interpretability_data[method]["interaction_matrix"]:
            interaction_array = np.array(interpretability_data[method]["interaction_matrix"])
            plot_data[method]["interaction_matrix_avg"] = np.mean(interaction_array, axis=0)
        
        # Model coefficients and intercept
        if method in all_sim_results and all_sim_results[method]:
            coef_values = [sim["coef"] for sim in all_sim_results[method] if "coef" in sim]
            if coef_values:
                plot_data[method]["coef_avg"] = np.mean(np.vstack(coef_values), axis=0)
                plot_data[method]["coef_std"] = np.std(np.vstack(coef_values), axis=0)
                
                # Get intercepts if available
                intercepts = [sim["intercept"] for sim in all_sim_results[method] if "intercept" in sim and sim["intercept"] is not None]
                if intercepts:
                    plot_data[method]["intercept_avg"] = np.mean(intercepts)
                    plot_data[method]["intercept_std"] = np.std(intercepts)

        

    
    # 2. Now use the aggregated data for plotting
    
    # Plot Shapley values for Choquet methods
    for method in [m for m in methods if m.startswith("choquet")]:
        if method in plot_data and "power_indices_avg" in plot_data[method]:
            if method == "choquet":
                plot_shapley_full(
                    feature_names, 
                    interpretability_data[method]["power_indices"], 
                    plot_folder
                )
            elif method == "choquet_2add":
                plot_shapley_2add(
                    feature_names, 
                    interpretability_data[method]["power_indices"], 
                    plot_folder
                )
                
                # Also plot marginal values for 2-additive model
                if "marginal_values" in interpretability_data[method]:
                    plot_marginal_2add(
                        feature_names, 
                        interpretability_data[method]["marginal_values"], 
                        plot_folder
                    )
    
    # Plot Banzhaf indices for MLM methods
    for method in [m for m in methods if m.startswith("mlm")]:
        if "power_indices_avg" in plot_data.get(method, {}):
            from plotting_functions import plot_banzhaf_indices
            plot_banzhaf_indices(
                feature_names, 
                interpretability_data[method]["power_indices"], 
                plot_folder, 
                method
            )
    
    # Plot coefficients
    for method in methods:
        if "coef_avg" not in plot_data.get(method, {}):
            continue
            
        # Get the average intercept from plot_data
        intercept = plot_data[method].get("intercept_avg", None)
        
        # Extract model type and k-additivity from method name
        model_type = "choquet" if method.startswith("choquet") else "mlm"
        k_add = None
        if "_" in method:
            k_part = method.split("_")[1]
            if k_part.endswith("add"):
                k_add = int(k_part.replace("add", ""))
        
        # Use either X or feature_names depending on k_add
        X_or_names = feature_names if k_add else X
        
        # Call the unified plot function
        from plotting_functions import plot_coef
        plot_coef(
            X_or_names, 
            plot_data[method]["coef_avg"], 
            plot_folder, 
            model_type=model_type, 
            k_add=k_add, 
            intercept=intercept
        )
    
    # Plot interaction matrices for all methods
    for method in methods:
        if method in plot_data and "interaction_matrix_avg" in plot_data[method]:
                        plot_interaction_matrix(
                X, 
                feature_names, 
                interpretability_data[method]["interaction_matrix"], 
                plot_folder, 
                method=method
            )
    
    # Generate side-by-side comparison plots - one for full models, one for 2-add models
    # Compare full models: Choquet vs MLM
    if "choquet" in methods and "mlm" in methods:
        if (interpretability_data["choquet"]["interaction_matrix"] and 
            interpretability_data["mlm"]["interaction_matrix"]):
            print("  Generating side-by-side comparison for full models")
            plot_interaction_comparison(
                feature_names, 
                interpretability_data["choquet"]["interaction_matrix"], 
                interpretability_data["mlm"]["interaction_matrix"], 
                plot_folder, 
                "full"
            )
    
    # Compare 2-add models: Choquet vs MLM
    if "choquet_2add" in methods and "mlm_2add" in methods:
        if (interpretability_data["choquet_2add"]["interaction_matrix"] and 
            interpretability_data["mlm_2add"]["interaction_matrix"]):
            print("  Generating side-by-side comparison for 2-add models")
            plot_interaction_comparison(
                feature_names, 
                interpretability_data["choquet_2add"]["interaction_matrix"], 
                interpretability_data["mlm_2add"]["interaction_matrix"], 
                plot_folder, 
                "2add"
            )
    
    # Plot log-odds and probabilities using the choquet_2add method directly from interpretability_data
    if "choquet_2add" in methods and interpretability_data["choquet_2add"]["log_odds"]:
        plot_log_odds_hist(
            interpretability_data["choquet_2add"]["log_odds"], 
            log_odds_bins, 
            plot_folder
        )
        plot_log_odds_vs_prob(
            interpretability_data["choquet_2add"]["log_odds"], 
            interpretability_data["choquet_2add"]["predicted_probs"], 
            plot_folder
        )
    
    # Compare interaction matrices between Choquet and MLM models
    if "choquet" in methods and "mlm" in methods:
        if (interpretability_data["choquet"]["interaction_matrix"] and 
            interpretability_data["mlm"]["interaction_matrix"]):
            plot_interaction_comparison(
                feature_names, 
                interpretability_data["choquet"]["interaction_matrix"], 
                interpretability_data["mlm"]["interaction_matrix"], 
                plot_folder, 
                "full"
            )
    
    if "choquet_2add" in methods and "mlm_2add" in methods:
        if (interpretability_data["choquet_2add"]["interaction_matrix"] and 
            interpretability_data["mlm_2add"]["interaction_matrix"]):
            plot_interaction_comparison(
                feature_names, 
                interpretability_data["choquet_2add"]["interaction_matrix"], 
                interpretability_data["mlm_2add"]["interaction_matrix"], 
                plot_folder, 
                "2add"
            )
    
    # Compare overall interaction indices computed using different methods
    if "choquet" in methods:
        choquet_data = interpretability_data["choquet"]
        if (choquet_data["power_indices"] and 
            choquet_data["marginal_values"] and 
            choquet_data["interaction_matrix"]):
            
            # Average over simulations
            avg_shapley = np.mean(np.vstack(choquet_data["power_indices"]), axis=0)
            avg_marginal = np.mean(np.vstack(choquet_data["marginal_values"]), axis=0)
            avg_interaction = np.mean(np.array(choquet_data["interaction_matrix"]), axis=0)
            
            # Method 2 (reference): Difference between Shapley and marginal values (ground truth)
            overall_method2 = overall_interaction_from_shapley(avg_shapley, avg_marginal)
            
            # Method 1: From interaction matrix using standard approach
            overall_method1_standard = overall_interaction_index(avg_interaction)
            
            # Try the corrected function with scaling factor
            from simulation_helper_functions import overall_interaction_index_corrected
            overall_method1_corrected = overall_interaction_index_corrected(avg_interaction)
            
            # Use our direct fixing approach
            from simulation_helper_functions import direct_fix_interaction_matrix
            direct_fixed_matrix = direct_fix_interaction_matrix(avg_interaction, avg_shapley, avg_marginal)
            overall_method1_direct = overall_interaction_index(direct_fixed_matrix)
            
            # Method 3: From absolute interactions (unchanged)
            overall_method3 = overall_interaction_index_abs(avg_interaction)
            
            # ====== VERIFICATION CODE ======
            from simulation_helper_functions import debug_interaction_calculation
            debug_result = debug_interaction_calculation(
                avg_shapley, avg_marginal, avg_interaction, feature_names
            )
            
            # Compare different methods
            diff_standard = np.abs(overall_method1_standard - overall_method2)
            diff_corrected = np.abs(overall_method1_corrected - overall_method2)
            diff_direct = np.abs(overall_method1_direct - overall_method2)
            
            print("\nVerification of different calculation methods:")
            print(f"Standard method - Average absolute difference: {np.mean(diff_standard):.8f}")
            print(f"Corrected method - Average absolute difference: {np.mean(diff_corrected):.8f}") 
            print(f"Direct fixed method - Average absolute difference: {np.mean(diff_direct):.8f}")
            # ==============================
            
            # Plot the overall interaction indices for all methods
            overall_dict = {
                "Matrix Method (Standard)": overall_method1_standard,
                "Matrix Method (Corrected)": overall_method1_corrected, 
                "Matrix Method (Direct Fix)": overall_method1_direct,
                "Shapley - Marginal": overall_method2,
                "Absolute Matrix Method": overall_method3
            }
            
            plot_overall_interaction(
                feature_names, 
                overall_dict, 
                "Overall Interaction Comparison", 
                plot_folder
            )
            
            # Add verification output
            print("\nVerification of interaction calculation methods:")
            print("Shapley-Marginal values (reference):", overall_method2)
            print("Standard Matrix Method values:", overall_method1_standard)
            print("Corrected Matrix Method values:", overall_method1_corrected)
            print("Direct Fixed Method values:", overall_method1_direct)
            
            # Save the interaction matrices for further analysis
            debug_file = os.path.join(plot_folder, 'interaction_matrices_debug.pkl')
            with open(debug_file, 'wb') as f:
                pickle.dump({
                    'original': avg_interaction,
                    'direct_fixed': direct_fixed_matrix,
                    'feature_names': feature_names,
                    'shapley': avg_shapley,
                    'marginal': avg_marginal,
                    'overall_truth': overall_method2
                }, f)
            print(f"Saved interaction matrices debug data to {debug_file}")
            
            # Extra diagnostics about the interaction matrix properties
            print("\nInteraction Matrix Properties:")
            print(f"  Original: sum = {np.sum(avg_interaction):.4f}, abs sum = {np.sum(np.abs(avg_interaction)):.4f}")
            print(f"  Direct fixed: sum = {np.sum(direct_fixed_matrix):.4f}, abs sum = {np.sum(np.abs(direct_fixed_matrix)):.4f}")
            
            # Compare specific problematic features
            problem_idx = np.argmax(diff_standard)
            print(f"\nMost problematic feature: {feature_names[problem_idx]}")
            print(f"  Shapley-Marginal: {overall_method2[problem_idx]:.6f}")
            print(f"  Standard Matrix: {overall_method1_standard[problem_idx]:.6f}")
            print(f"  Corrected Matrix: {overall_method1_corrected[problem_idx]:.6f}")
            print(f"  Direct Fixed: {overall_method1_direct[problem_idx]:.6f}")

    # For final verification of scale consistency
    if "choquet" in methods and "choquet_2add" in methods:
        # Get all shapley values
        choquet_shapley = np.mean(np.vstack(interpretability_data["choquet"]["power_indices"]), axis=0)  
        choquet_2add_shapley = np.mean(np.vstack(interpretability_data["choquet_2add"]["power_indices"]), axis=0)
        
        print("\nScaling verification:")
        print(f"Choquet full shapley range: {np.min(choquet_shapley):.4f} to {np.max(choquet_shapley):.4f}")
        print(f"Choquet 2add shapley range: {np.min(choquet_2add_shapley):.4f} to {np.max(choquet_2add_shapley):.4f}")
        
        # Let's calculate reasonable scale factors for debugging purposes
        shapley_ratio = np.mean(np.abs(choquet_shapley)) / np.mean(np.abs(choquet_2add_shapley))
        print(f"Average magnitude ratio (full/2add): {shapley_ratio:.4f}")

    # Model accuracy comparison
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


    aggregated_complexity = {}

    model_names = []
    for sim in complexity_results:
        for model in complexity_results[sim].keys():
            if model not in model_names:
                model_names.append(model)

    # Aggregate metrics for each model
    for model_name in model_names:
        aggregated_complexity[model_name] = {}
        
        train_times = []
        train_stds = []
        pred_times = []
        pred_stds = []
        memory_usages = []
        
        for sim in complexity_results:
            if model_name in complexity_results[sim]:
                model_data = complexity_results[sim][model_name]
                if "train_time" in model_data: train_times.append(model_data["train_time"])
                if "train_std" in model_data: train_stds.append(model_data["train_std"])
                if "pred_time" in model_data: pred_times.append(model_data["pred_time"])
                if "pred_std" in model_data: pred_stds.append(model_data["pred_std"])
                if "memory_mb" in model_data: memory_usages.append(model_data["memory_mb"])
        
        if train_times: aggregated_complexity[model_name]["train_time"] = np.mean(train_times)
        else: aggregated_complexity[model_name]["train_time"] = 0
        
        if train_stds: aggregated_complexity[model_name]["train_std"] = np.mean(train_stds)
        else: aggregated_complexity[model_name]["train_std"] = 0
        
        if pred_times: aggregated_complexity[model_name]["pred_time"] = np.mean(pred_times)
        else: aggregated_complexity[model_name]["pred_time"] = 0
        
        if pred_stds: aggregated_complexity[model_name]["pred_std"] = np.mean(pred_stds)
        else: aggregated_complexity[model_name]["pred_std"] = 0
        
        if memory_usages: aggregated_complexity[model_name]["memory_mb"] = np.mean(memory_usages)
        else: aggregated_complexity[model_name]["memory_mb"] = 0

    # Plot aggregated complexity across all simulations
    fig = plot_complexity_results(aggregated_complexity, title="Average Model Complexity Across Simulations")
    fig.savefig(os.path.join(plot_folder, "complexity_comparison_avg.png"))

    # Plot individual simulation complexity
    idx = np.random.randint(0, n_simulations)
    if idx in complexity_results and complexity_results[idx]:
        fig_sim = plot_complexity_results(complexity_results[idx], title=f"Model Complexity (Simulation {idx+1})")
        fig_sim.savefig(os.path.join(plot_folder, f"complexity_comparison_sim{idx+1}.png"))
    else:
        print(f"Could not find complexity data for simulation {idx+1}.")



    # Package final results
    final_results = {
        "simulations": all_sim_results,
        "interpretability": interpretability_data
    }

    # Save results to pickle file
    results_filename_only = os.path.basename(results_filename)
    if not results_filename_only.endswith('.pkl'):
        results_filename_only = "results.pkl"

    # Always save in the dataset-specific plot folder
    results_path = os.path.join(plot_folder, results_filename_only)

    # Make sure directory exists
    ensure_folder(os.path.dirname(results_path))

    print(f"\nSaving simulation results to {results_path}")
    with open(results_path, 'wb') as f:
        pickle.dump(final_results, f)
    print("Results saved successfully.")

    return final_results
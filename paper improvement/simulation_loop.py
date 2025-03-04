import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle
from os.path import join

# necessary for shapley values
os.environ["SCIPY_ARRAY_API"] = "1"

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

from regression_classes import ChoquisticRegression
import mod_GenFuzzyRegression as modGF

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
    if baseline_max_iter is not None:
        baseline_logistic_params["max_iter"] = baseline_max_iter
        choq_logistic_params["max_iter"] = baseline_max_iter
    if penalty_lr is not None:
        baseline_logistic_params["penalty"] = penalty_lr
        choq_logistic_params["penalty"] = penalty_lr

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # 1. Load and normalize the data
    X, y = modGF.func_read_data(data_imp)

    # Containers for results and plots
    all_sim_results = []
    all_shapley_full = []    # for full choquet shapley values (each should be 1D array of length n_features)
    all_shapley_2add = []    # for choquet 2-add shapley values (1D arrays)
    all_marginal_2add = []   # for choquet 2-add marginal contributions (direct main effects)
    all_interaction_matrices = []  # for choquet 2-add interaction matrices
    all_log_odds = []        # concatenated log-odds for test set
    all_probs = []           # concatenated predicted probabilities for test set

    for sim in range(n_simulations):
        sim_results = {}
        print(f"\nSimulation {sim+1}/{n_simulations}")
        sim_seed = random_state + sim

        # 2. Split data into train/test sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=sim_seed
        )

        # Scale data if required
        if scale_data:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # 3. Baseline Logistic Regression on raw features
        baseline_params = baseline_logistic_params.copy()
        baseline_params["random_state"] = sim_seed
        lr_baseline = LogisticRegression(**baseline_params)
        lr_baseline.fit(X_train, y_train)
        baseline_train_acc = lr_baseline.score(X_train, y_train)
        baseline_test_acc = lr_baseline.score(X_test, y_test)
        print("Baseline LR Train Acc: {:.2%}, Test Acc: {:.2%}".format(baseline_train_acc, baseline_test_acc))
        sim_results["LR"] = {"train_acc": baseline_train_acc, "test_acc": baseline_test_acc, "coef": lr_baseline.coef_}

        # 4. Run Choquistic Models using various methods
        for method in methods:
            print("Processing method:", method)
            model = ChoquisticRegression(
                method=method,
                logistic_params=choq_logistic_params,
                scale_data=scale_data,
                random_state=sim_seed,
            )
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            coef = model.classifier_.coef_ if hasattr(model, "classifier_") else model.coef_
            sim_results[method] = {"train_acc": train_acc, "test_acc": test_acc, "coef": coef}
            
            if method == "choquet":
                try:
                    shapley_vals = model.compute_shapley_values()
                    sim_results[method]["shapley"] = shapley_vals
                    # Ensure the returned array is 1D
                    all_shapley_full.append(np.atleast_1d(shapley_vals))
                    print(f"Full Choquet Shapley values: {shapley_vals}")
                except Exception as e:
                    print(f"Could not compute Shapley values for full choquet: {e}")
            elif method == "choquet_2add":
                try:
                    shapley_dict = model.compute_shapley_values()
                    sim_results[method]["shapley"] = shapley_dict["shapley"]
                    sim_results[method]["marginal"] = shapley_dict["marginal"]
                    all_shapley_2add.append(np.atleast_1d(shapley_dict["shapley"]))
                    all_marginal_2add.append(np.atleast_1d(shapley_dict["marginal"]))
                    print(f"Choquet 2-add Shapley values: {shapley_dict['shapley']}")
                    print(f"Choquet 2-add Marginal contributions: {shapley_dict['marginal']}")
                except Exception as e:
                    print(f"Could not compute values for choquet 2-add: {e}")
            print(f"Method: {method:12s} | Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")

            if method == "choquet_2add":
                if hasattr(model, "classifier_"):
                    coef_ch = model.classifier_.coef_[0]
                else:
                    coef_ch = model.coef_[0]
                nAttr = X_train.shape[1]
                interaction_coef = coef_ch[nAttr:]
                interaction_matrix = np.zeros((nAttr, nAttr))
                idx = 0
                for i in range(nAttr):
                    for j in range(i + 1, nAttr):
                        interaction_matrix[i, j] = interaction_coef[idx]
                        interaction_matrix[j, i] = interaction_coef[idx]
                        idx += 1
                all_interaction_matrices.append(interaction_matrix)
                # Collect log-odds and probabilities from the 2-add model
                log_odds_test = model.decision_function(X_test)
                probs_test = model.predict_proba(X_test)
                all_log_odds.append(log_odds_test)
                all_probs.append(probs_test)
                sim_results["choquet_2add"].update({
                    "log_odds_test": log_odds_test,
                    "predicted_probabilities_test": probs_test,
                })
        all_sim_results.append(sim_results)

    # ---------------- Plotting ----------------

    # (A) Average Shapley Values (Full Choquet)
    if all_shapley_full:
        all_shapley_full_arr = np.vstack(all_shapley_full)
        mean_shapley_full = np.mean(all_shapley_full_arr, axis=0)
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            nAttr = X.shape[1]
            feature_names = [f"F{i}" for i in range(nAttr)]
        ordered_indices = np.argsort(mean_shapley_full)[::-1]
        ordered_names = np.array(feature_names)[ordered_indices]
        ordered_values = mean_shapley_full[ordered_indices]
        plt.figure(figsize=(10, 8))
        plt.barh(ordered_names, ordered_values, color="steelblue", edgecolor="black")
        plt.xlabel("Avg. Shapley Value", fontsize=16)
        plt.title("Average Shapley Values (Full Choquet)", fontsize=18)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        full_plot_path = join(plot_folder, "avg_shapley_values_full.png")
        plt.savefig(full_plot_path)
        plt.close()
        print("Saved Full Choquet Shapley values plot to:", full_plot_path)
    else:
        print("No Full Choquet Shapley values computed; skipping plot.")

    # (B) Average Shapley Values (Choquet 2-add)
    if all_shapley_2add:
        all_shapley_2add_arr = np.vstack(all_shapley_2add)
        mean_shapley_2add = np.mean(all_shapley_2add_arr, axis=0)
        std_shapley_2add = np.std(all_shapley_2add_arr, axis=0)
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            nAttr = X.shape[1]
            feature_names = [f"F{i}" for i in range(nAttr)]
        ordered_indices = np.argsort(mean_shapley_2add)[::-1]
        ordered_names = np.array(feature_names)[ordered_indices]
        ordered_values = mean_shapley_2add[ordered_indices]
        ordered_std = std_shapley_2add[ordered_indices]
        plt.figure(figsize=(10, 8))
        plt.barh(ordered_names, ordered_values, xerr=ordered_std, color="seagreen", edgecolor="black")
        plt.xlabel("Avg. Shapley Value", fontsize=16)
        plt.title("Average Shapley Values (Choquet 2-add)", fontsize=18)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        add_shapley_plot_path = join(plot_folder, "avg_shapley_values_2add.png")
        plt.savefig(add_shapley_plot_path)
        plt.close()
        print("Saved Choquet 2-add Shapley values plot to:", add_shapley_plot_path)
    else:
        print("No Choquet 2-add Shapley values computed; skipping plot.")

    # (C) Marginal Contributions (Choquet 2-add, Direct Main Effects)
    if all_marginal_2add:
        all_marginal_2add_arr = np.vstack(all_marginal_2add)
        mean_marginal_2add = np.mean(all_marginal_2add_arr, axis=0)
        std_marginal_2add = np.std(all_marginal_2add_arr, axis=0)
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            nAttr = X.shape[1]
            feature_names = [f"F{i}" for i in range(nAttr)]
        ordered_indices = np.argsort(mean_marginal_2add)[::-1]
        ordered_names = np.array(feature_names)[ordered_indices]
        ordered_values = mean_marginal_2add[ordered_indices]
        ordered_std = std_marginal_2add[ordered_indices]
        plt.figure(figsize=(10, 8))
        plt.barh(ordered_names, ordered_values, xerr=ordered_std, color="darkorange", edgecolor="black")
        plt.xlabel("Avg. Marginal Contribution", fontsize=16)
        plt.title("Marginal Contributions (Choquet 2-add, Direct Main Effects)", fontsize=18)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        marginal_plot_path = join(plot_folder, "marginal_contributions_2add.png")
        plt.savefig(marginal_plot_path)
        plt.close()
        print("Saved marginal contributions plot to:", marginal_plot_path)
    else:
        print("No marginal contributions computed; skipping marginal contributions plot.")

    # (D) Average Interaction Matrix (Choquet 2-additive)
    if all_interaction_matrices:
        mean_interaction_matrix = np.mean(np.array(all_interaction_matrices), axis=0)
        plt.figure(figsize=(8, 6))
        plt.imshow(mean_interaction_matrix, cmap="viridis", interpolation="nearest")
        plt.colorbar(orientation="vertical", label="Interaction Value")
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            nAttr = X.shape[1]
            feature_names = [f"F{i}" for i in range(nAttr)]
        plt.xticks(range(nAttr), feature_names, rotation=90, fontsize=12)
        plt.yticks(range(nAttr), feature_names, fontsize=12)
        plt.title("Avg. Interaction Matrix (Choquet 2-add)", fontsize=16)
        plt.tight_layout()
        interaction_plot_path = join(plot_folder, "avg_interaction_matrix.png")
        plt.savefig(interaction_plot_path)
        plt.close()
        print("Saved interaction effects plot to:", interaction_plot_path)

    # (E) Log-Odds Distribution Histogram (Choquet 2-additive)
    if all_log_odds:
        all_log_odds_concat = np.concatenate(all_log_odds)
        plt.figure(figsize=(10, 6))
        plt.hist(
            all_log_odds_concat,
            bins=log_odds_bins,
            color="mediumseagreen",
            edgecolor="black",
        )
        plt.xlabel("Log-Odds", fontsize=16)
        plt.ylabel("Frequency", fontsize=16)
        plt.title("Log-Odds Distribution (Choquet 2-add)", fontsize=18)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        log_odds_plot_path = join(plot_folder, "log_odds_distribution.png")
        plt.savefig(log_odds_plot_path)
        plt.close()
        print("Saved log-odds histogram to:", log_odds_plot_path)

    # (F) Log-Odds vs. Predicted Probability Scatter
    if all_log_odds and all_probs:
        all_log_odds_concat = np.concatenate(all_log_odds)
        all_probs_concat = np.concatenate(all_probs, axis=0)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            all_log_odds_concat,
            all_probs_concat[:, 1],
            alpha=0.7,
            color="darkorange",
            edgecolor="k",
        )
        plt.xlabel("Log-Odds", fontsize=16)
        plt.ylabel("Predicted Probability (Positive)", fontsize=16)
        plt.title("Log-Odds vs. Predicted Probability", fontsize=18)
        plt.grid(axis="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        log_odds_prob_plot_path = join(plot_folder, "log_odds_vs_probability.png")
        plt.savefig(log_odds_prob_plot_path)
        plt.close()
        print("Saved log-odds vs. probability plot to:", log_odds_prob_plot_path)

    # (G) Shapley vs. Banzhaf Comparison (Choquet)
    shapleys = []
    banzhaf_indices = []
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        m = X.shape[1]
        feature_names = [f"F{i}" for i in range(m)]
    m = X.shape[1]
    if all_sim_results:
        from regression_classes import (
            choquet_matrix,
            compute_banzhaf_indices,
            compute_banzhaf_interaction_matrix,
        )

        _, all_coalitions = choquet_matrix(X)
        for sim in all_sim_results:
            if "choquet" in sim and "shapley" in sim["choquet"]:
                v = sim["choquet"]["coef"][0]
                banzhaf = compute_banzhaf_indices(v, m, all_coalitions)
                banzhaf_indices.append(banzhaf)
                shapleys.append(sim["choquet"]["shapley"])
        if shapleys:
            mean_shapley = np.mean(shapleys, axis=0)
            mean_banzhaf = np.mean(banzhaf_indices, axis=0)
            plt.figure(figsize=(12, 6))
            indices = np.arange(m)
            width = 0.4
            plt.bar(
                indices - width / 2,
                mean_shapley,
                width,
                color="dodgerblue",
                label="Shapley",
            )
            plt.bar(
                indices + width / 2,
                mean_banzhaf,
                width,
                color="crimson",
                label="Banzhaf",
            )
            plt.xticks(indices, feature_names, rotation=45, fontsize=12)
            plt.xlabel("Features", fontsize=14)
            plt.ylabel("Value", fontsize=14)
            plt.title("Shapley vs. Banzhaf Comparison", fontsize=16)
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            comparison_plot_path = join(plot_folder, "shapley_vs_banzhaf.png")
            plt.savefig(comparison_plot_path)
            plt.close()
            print("Saved Shapley vs. Banzhaf plot to:", comparison_plot_path)
        else:
            print(
                "No Shapley values computed for choquet method; skipping comparison plot."
            )

    # (H) Average Banzhaf Interaction Matrix (Choquet)
    banzhaf_interaction_matrices = []
    for sim in all_sim_results:
        if "choquet" in sim:
            v = sim["choquet"]["coef"][0]
            from regression_classes import choquet_matrix

            if scale_data:
                from sklearn.preprocessing import StandardScaler

                X_co = StandardScaler().fit_transform(X)
            else:
                X_co = X
            _, all_coalitions = choquet_matrix(X_co)
            bi_matrix = compute_banzhaf_interaction_matrix(v, m, all_coalitions)
            banzhaf_interaction_matrices.append(bi_matrix)
    if banzhaf_interaction_matrices:
        mean_banzhaf_interaction = np.mean(
            np.array(banzhaf_interaction_matrices), axis=0
        )
        plt.figure(figsize=(8, 6))
        plt.imshow(mean_banzhaf_interaction, cmap="viridis", interpolation="nearest")
        plt.colorbar(orientation="vertical", label="Banzhaf Value")
        plt.xticks(range(m), feature_names, rotation=90, fontsize=12)
        plt.yticks(range(m), feature_names, fontsize=12)
        plt.title("Avg. Banzhaf Interaction Matrix", fontsize=16)
        plt.tight_layout()
        banzhaf_interaction_plot_path = join(plot_folder, "avg_banzhaf_interaction.png")
        plt.savefig(banzhaf_interaction_plot_path)
        plt.close()
        print("Saved Banzhaf interaction plot to:", banzhaf_interaction_plot_path)
    else:
        print("No Banzhaf interaction data; skipping Banzhaf plot.")

    # (I) Average Test Accuracy by Model
    model_names = set()
    for sim in all_sim_results:
        for key in sim.keys():
            if key not in ["choquet_2add_extra"]:
                model_names.add(key)
    model_names = sorted(list(model_names))
    acc_data = {model: [] for model in model_names}
    for sim in all_sim_results:
        for model in model_names:
            if model in sim:
                acc_data[model].append(sim[model]["test_acc"])
            else:
                acc_data[model].append(None)
    avg_acc = {
        model: np.mean([acc for acc in acc_data[model] if acc is not None])
        for model in model_names
    }

    # Use a consistent color cycle from 'tab10'
    colors = plt.get_cmap("tab10").colors
    plt.figure(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.8
    for i, model in enumerate(model_names):
        plt.bar(
            x[i],
            avg_acc[model],
            width=width,
            color=colors[i % len(colors)],
            edgecolor="black",
            label=model,
        )
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.title("Average Test Accuracy by Model", fontsize=16)
    plt.xticks(x, model_names, fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    acc_plot_path = join(plot_folder, "avg_test_accuracy.png")
    plt.savefig(acc_plot_path)
    plt.close()
    print("Saved test accuracy plot to:", acc_plot_path)

    # (J) Optional: Decision Boundary Plot for 2D data
    def plot_decision_boundary(X, y, model, filename):
        from matplotlib.colors import ListedColormap

        X = np.array(X)
        y = np.array(y)
        cmap_light = ListedColormap(["#FFCCCC", "#CCFFCC"])
        cmap_bold = ListedColormap(["#FF0000", "#00AA00"])
        if X.shape[1] != 2:
            print("Decision boundary plot only works for 2D data.")
            return
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
        )
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Decision Boundary")
        plt.savefig(filename)
        plt.close()
        print("Saved decision boundary plot to:", filename)

    if np.array(X).shape[1] == 2:
        plot_decision_boundary(
            X_train, y_train, lr_baseline, join(plot_folder, "decision_boundary_lr.png")
        )
        model_ch2add_last = ChoquisticRegression(
            method="choquet_2add",
            logistic_params=choq_logistic_params,
            scale_data=scale_data,
            random_state=random_state,
        )
        model_ch2add_last.fit(X_train, y_train)
        plot_decision_boundary(
            X_train,
            y_train,
            model_ch2add_last,
            join(plot_folder, "decision_boundary_choquistic.png"),
        )

    # (K) Main Effects from Regression Coefficients (Choquet 2-add)
    choq2add_coefs = []
    for sim in all_sim_results:
        if "choquet_2add" in sim:
            coef = sim["choquet_2add"]["coef"][0]  # [main effects | interactions]
            choq2add_coefs.append(coef)
    choq2add_coefs = np.array(choq2add_coefs)
    nAttr = X.shape[1]
    main_effects = choq2add_coefs[:, :nAttr]
    mean_main = np.mean(main_effects, axis=0)
    std_main = np.std(main_effects, axis=0)
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"F{i}" for i in range(nAttr)]
    ordered_idx = np.argsort(mean_main)[::-1]
    ordered_names = np.array(feature_names)[ordered_idx]
    ordered_mean = mean_main[ordered_idx]
    ordered_std = std_main[ordered_idx]

    plt.figure(figsize=(10, 8))
    plt.barh(
        ordered_names,
        ordered_mean,
        xerr=ordered_std,
        color="dodgerblue",
        edgecolor="black",
    )
    plt.xlabel("Coefficient Value", fontsize=14)
    plt.title("Main Effects: Regression Coefficients (Choquet 2-add)", fontsize=16)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    reg_coef_plot_path = join(plot_folder, "main_effects_reg_coef.png")
    plt.savefig(reg_coef_plot_path)
    plt.close()
    print("Saved regression coefficients plot to:", reg_coef_plot_path)

    # Save overall results
    final_results = {"simulations": all_sim_results}
    with open(results_filename, "wb") as f:
        pickle.dump(final_results, f)
    print("Saved overall results to", results_filename)

    return final_results

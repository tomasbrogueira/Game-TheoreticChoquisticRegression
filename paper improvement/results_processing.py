def results_processing(
    results_filename="results.pkl",
    data_imp="dados_covid_sbpo_atual",
    plot_folder="plots", 
    log_odds_bins=30,
    scale_data=True
):
    """
    Loads the pre-computed results from results_filename and generates
    all plots and analyses as in the simulation function.

    Parameters:
      - results_filename: filename of the pre-computed results (default: "results.pkl")
      - data_imp: dataset identifier used to re-read the data (for feature names, etc.)
      - plot_folder: directory to save plots (customizable)
      - log_odds_bins: number of bins for the log-odds histogram
      - scale_data: whether data was scaled during simulation (used when re-reading data)

    Returns:
      - final_results: the loaded results dictionary.
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from os.path import join
    import mod_GenFuzzyRegression as modGF

    if not os.path.exists(results_filename):
        print(f"Results file {results_filename} not found. Please run the simulation first.")
        return None

    with open(results_filename, "rb") as f:
        final_results = pickle.load(f)
    print(f"Loaded results from {results_filename}")

    # Re-read data for labels and feature names
    X, y = modGF.func_read_data(data_imp)
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        nAttr = X.shape[1]
        feature_names = [f"F{i}" for i in range(nAttr)]

    # Ensure the plot folder exists
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Reconstruct aggregated arrays from simulation results
    all_sim_results = final_results["simulations"]
    all_shapley_full = []    # For full Choquet Shapley values
    all_shapley_2add = []    # For Choquet 2-add Shapley values
    all_marginal_2add = []   # For Choquet 2-add marginal contributions
    all_interaction_matrices = []  # For Choquet 2-add interaction matrices
    all_log_odds = []        # For log-odds from test sets
    all_probs = []           # For predicted probabilities

    for sim in all_sim_results:
        if "choquet" in sim and "shapley" in sim["choquet"]:
            all_shapley_full.append(np.atleast_1d(sim["choquet"]["shapley"]))
        if "choquet_2add" in sim:
            if "shapley" in sim["choquet_2add"]:
                all_shapley_2add.append(np.atleast_1d(sim["choquet_2add"]["shapley"]))
            if "marginal" in sim["choquet_2add"]:
                all_marginal_2add.append(np.atleast_1d(sim["choquet_2add"]["marginal"]))
            if "log_odds_test" in sim["choquet_2add"]:
                all_log_odds.append(sim["choquet_2add"]["log_odds_test"])
            if "predicted_probabilities_test" in sim["choquet_2add"]:
                all_probs.append(sim["choquet_2add"]["predicted_probabilities_test"])
            if "interaction_matrix" in sim["choquet_2add"]:
                all_interaction_matrices.append(sim["choquet_2add"]["interaction_matrix"])

    # ---------------- Plotting Section ----------------

    # (A) Average Shapley Values (Full Choquet)
    if all_shapley_full:
        all_shapley_full_arr = np.vstack(all_shapley_full)
        mean_shapley_full = np.mean(all_shapley_full_arr, axis=0)
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
        shapley2add_plot_path = join(plot_folder, "avg_shapley_values_2add.png")
        plt.savefig(shapley2add_plot_path)
        plt.close()
        print("Saved Choquet 2-add Shapley values plot to:", shapley2add_plot_path)
    else:
        print("No Choquet 2-add Shapley values computed; skipping plot.")

    # (C) Marginal Contributions (Choquet 2-add, Direct Main Effects)
    if all_marginal_2add:
        all_marginal_2add_arr = np.vstack(all_marginal_2add)
        mean_marginal_2add = np.mean(all_marginal_2add_arr, axis=0)
        std_marginal_2add = np.std(all_marginal_2add_arr, axis=0)
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
        plt.xticks(range(len(feature_names)), feature_names, rotation=90, fontsize=12)
        plt.yticks(range(len(feature_names)), feature_names, fontsize=12)
        plt.title("Avg. Interaction Matrix (Choquet 2-add)", fontsize=16)
        plt.tight_layout()
        interaction_plot_path = join(plot_folder, "avg_interaction_matrix.png")
        plt.savefig(interaction_plot_path)
        plt.close()
        print("Saved interaction effects plot to:", interaction_plot_path)
    else:
        print("No interaction matrix data; skipping interaction matrix plot.")

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
    else:
        print("No log-odds data available; skipping log-odds histogram.")

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
    else:
        print("Insufficient data for log-odds vs. probability plot; skipping.")

    print("Results processing and plotting complete.")
    return final_results

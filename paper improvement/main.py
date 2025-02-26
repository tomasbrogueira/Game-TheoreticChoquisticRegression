import os
import matplotlib.pyplot as plt
from experiment_runner_new import run_experiment

# Datasets
datasets = ['dados_covid_sbpo_atual','banknotes','transfusion','mammographic','raisin','rice','diabetes','skin']
#datasets = ['dados_covid_sbpo_atual']

for dataset in datasets:
    # Create a dataset-specific folder inside the "plots" folder.
    dataset_folder = os.path.join("plots", dataset)
    os.makedirs(dataset_folder, exist_ok=True)
    
    # Run the experiment, saving all plots and results into the dataset folder.
    results = run_experiment(
        data_imp=dataset,
        test_size=0.2,
        random_state=0,
        n_simulations=1,  # You can set a higher number of simulation runs.
        solver_lr='newton-cg',
        baseline_max_iter=1000,
        penalty_lr=None,
        methods=["choquet_2add", "choquet", "mlm_2add", "mlm"],
        scale_data=False,
        plot_folder=dataset_folder,  # Pass the dataset-specific folder.
        results_filename=os.path.join(dataset_folder, "results.pkl"),
        log_odds_bins=30
    )
    
    # Collect baseline Logistic Regression test accuracies from each simulation.
    simulation_ids = []
    lr_test_accuracies = []
    for i, sim in enumerate(results['simulations']):
        simulation_ids.append(f"Sim {i+1}")
        lr_test_accuracies.append(sim['LR']['test_acc'])
    
    # Plot the test accuracies.
    plt.figure(figsize=(8, 6))
    plt.bar(simulation_ids, lr_test_accuracies, color='skyblue', edgecolor='black')
    plt.xlabel("Simulation", fontsize=14)
    plt.ylabel("Baseline LR Test Accuracy", fontsize=14)
    plt.title(f"Baseline LR Test Accuracy for {dataset}", fontsize=16)
    plt.ylim(0, 1)  # Accuracy is between 0 and 1.
    plt.tight_layout()
    
    # Save the accuracy plot to the dataset-specific folder.
    acc_plot_path = os.path.join(dataset_folder, "baseline_lr_test_accuracy.png")
    plt.savefig(acc_plot_path)
    plt.close()
    
    print(f"Finished experiment for {dataset}. Baseline LR test accuracy plot saved at: {acc_plot_path}")

import os
from experiment_runner_new import run_experiment

# Datasets
datasets = ['dados_covid_sbpo_atual','banknotes','transfusion','mammographic','raisin','rice','diabetes','skin']
#datasets = ['transfusion']
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
        n_simulations=50,  # You can set a higher number of simulation runs.
        solver_lr='newton-cg',
        baseline_max_iter=10000,
        penalty_lr=None,
        methods=["choquet_2add", "choquet", "mlm_2add", "mlm"],
        scale_data=False,
        plot_folder=dataset_folder,  # Pass the dataset-specific folder.
        results_filename=os.path.join(dataset_folder, "results.pkl"),
        log_odds_bins=30
    )
    
    print(f"Finished experiment for {dataset}.")

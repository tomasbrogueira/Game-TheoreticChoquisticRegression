from experiment_runner_new import run_experiment

#choose the dataset to be used
#datasets = ['dados_covid_sbpo_atual','banknotes','transfusion','mammographic','raisin','rice','diabetes','skin']
datasets = ['dados_covid_sbpo_atual']

# Adjust hyperparameters, including number of simulations, as desired.
for dataset in datasets:
    results = run_experiment(
        data_imp=dataset,
        test_size=0.2,
        random_state=0,
        n_simulations=1,  # set the number of simulation runs here
        solver_lr='newton-cg',
        baseline_max_iter=10000,
        baseline_logistic_params={'penalty': None},
        choq_logistic_params={'penalty': None},
        methods=["choquet_2add", "choquet", "mlm_2add", "mlm"],
        scale_data=True,
        plot_folder="plots",
        results_filename="results.pkl",
        log_odds_bins=30
    )

    # Further analysis can be performed using the 'results' dictionary.
    print(dataset + " finished. Baseline LR test accuracy for simulation 1: {:.2%}".format(results['simulations'][0]['LR']['test_acc']))

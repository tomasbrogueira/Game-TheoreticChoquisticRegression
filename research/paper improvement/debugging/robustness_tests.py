from regression_classes import ChoquisticRegression_Composition, nParam_kAdd
from robustness import test_model_robustness, compare_regularization_robustness
from mod_GenFuzzyRegression import func_read_data

if __name__ == "__main__":
    datasets = ['dados_covid_sbpo_atual', 'banknotes', 'transfusion', 'mammographic', 'raisin', 'rice', 'diabetes', 'skin']
    
    robustness_models = {}

    
    for dataset in datasets:  
        try:
            # Read data
            X, y = func_read_data(dataset)
            # Data parameters
            nSamp, nAttr = X.shape

            for k in range(1, nAttr + 1):
                # Choquet model with inheritance
                model = ChoquisticRegression_Composition(
                    method="choquet",
                    representation="game",
                    k_add=k,
                    scale_data=True,
                    logistic_params={"max_iter": 1000, "random_state": 0, "solver": "newton-cg"}
                )
                model.fit(X, y)
                robustness_models[dataset] = {
                    'model': model,
                    'k_add': k,
                    'nParam': nParam_kAdd(k, nAttr),
                    'dataset': dataset
                }
            
        except Exception as e:
            print(f"Error analyzing dataset {dataset}: {str(e)}")
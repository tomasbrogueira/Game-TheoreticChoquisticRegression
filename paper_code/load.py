import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load data
accuracy_linear_train, accuracy_linear_test, accuracy_choquet_kadd_train, accuracy_choquet_kadd_test, \
accuracy_choquet_train, accuracy_choquet_test, accuracy_mlm_kadd_train, accuracy_mlm_kadd_test, \
accuracy_mlm_train, accuracy_mlm_test, data_imp, param_linear_train, param_choquet_train, \
param_choquet_kadd_train, param_mlm_train, param_mlm_kadd_train, solver_lr = np.load('results_logistic_all_test.npy', allow_pickle=True)

# Compute mean and std values
metrics = {
    "Linear": (accuracy_linear_train, accuracy_linear_test),
    "Choquet": (accuracy_choquet_train, accuracy_choquet_test),
    "Choquet Kadd": (accuracy_choquet_kadd_train, accuracy_choquet_kadd_test),
    "MLM": (accuracy_mlm_train, accuracy_mlm_test),
    "MLM Kadd": (accuracy_mlm_kadd_train, accuracy_mlm_kadd_test)
}

results = {}
for key, (train, test) in metrics.items():
    results[key] = {
    "Train Mean": np.mean(train, axis=2),
    "Train Std": np.std(train, axis=2),
    "Test Mean": np.mean(test, axis=2),
    "Test Std": np.std(test, axis=2)
}


# Ensure the "plots/tables" folder exists
os.makedirs("plots/tables", exist_ok=True)

def save_table_as_image(df, title, filename):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title, fontsize=14)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved table: {filename}")

# Generate tables for each model
for model, data in results.items():
    n_datasets = data["Train Mean"].shape[0]
    for i in range(n_datasets):
         df = pd.DataFrame({
             "Train Mean": [f"{val*100:.2f}%" for val in data["Train Mean"][i, :]],
             "Train Std": [f"{val*100:.2f}%" for val in data["Train Std"][i, :]],
             "Test Mean": [f"{val*100:.2f}%" for val in data["Test Mean"][i, :]],
             "Test Std": [f"{val*100:.2f}%" for val in data["Test Std"][i, :]]
         }, index=[f"Solver {j+1}" for j in range(data["Train Mean"].shape[1])])
         
         # Create a folder for the current dataset inside "plots/tables"
         dataset_folder = os.path.join("plots", "tables", data_imp[i])
         os.makedirs(dataset_folder, exist_ok=True)
         
         # Save the table image in that folder
         save_table_as_image(df, 
             f"{model} Model Accuracy - {data_imp[i]}", 
             os.path.join(dataset_folder, f"{model}_accuracy.png")
         )


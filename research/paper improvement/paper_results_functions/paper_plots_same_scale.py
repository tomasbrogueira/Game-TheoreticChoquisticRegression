import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np

def parse_summary_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    table_start_index = -1
    for i, line in enumerate(lines):
        if "FULL RESULTS TABLE:" in line:
            table_start_index = i + 1
            break
    
    if table_start_index == -1:
        return None

    # The table data is from table_start_index to the end of the file
    table_lines = lines[table_start_index:]
    
    # The first line is the header, the rest is data
    header = re.split(r'\s+', table_lines[0].strip())
    
    data = []
    for line in table_lines[1:]:
        if line.strip():
            # The first element is the index from the file, which we can ignore
            data.append(re.split(r'\s+', line.strip())[1:])

    df = pd.DataFrame(data, columns=header)
    
    # Convert columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def plot_k_additivity_robustness(df, title, output_path, y_lims):
    """
    Plots the model accuracy against k for different noise levels, with error bars for standard deviation.
    """
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, 3))  # 3 noise levels

    noise_levels = [0.1, 0.2, 0.3]
    for i, noise_level in enumerate(noise_levels):
        noise_col = f'noise_{noise_level}'
        std_col = f'noise_{noise_level}_std'
        
        if std_col in df.columns and not df[std_col].isnull().all():
            plt.errorbar(df['k_value'], df[noise_col], yerr=df[std_col], fmt='o-',
                         color=colors[i], linewidth=2, label=f'Noise level: {noise_level}', capsize=5)
        else:
            plt.plot(df['k_value'], df[noise_col], 'o-',
                     color=colors[i], linewidth=2, label=f'Noise level: {noise_level}')

    plt.plot(df['k_value'], df['baseline_accuracy'], 'k--',
             linewidth=2, label='Baseline (no noise)')

    plt.title(title)
    plt.xlabel('k-additivity')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(y_lims)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

def plot_bootstrap_stability(df, title, output_path, y_lims):
    """
    Plots the bootstrap stability with uncertainty.
    """
    plt.figure(figsize=(12, 8))
    plt.errorbar(df['k_value'], df['bootstrap_mean'], 
                 yerr=df['bootstrap_std'], fmt='o-', capsize=5, linewidth=2)
    plt.title(title)
    plt.xlabel('k-additivity')
    plt.ylabel('Bootstrap Accuracy (mean ± std)')
    plt.grid(True, alpha=0.3)
    plt.ylim(y_lims)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

def main(dataset_name='dados_covid_sbpo_atual'):
    base_dir = 'c:/Users/Tomas/OneDrive - Universidade de Lisboa/3ºano_LEFT/PIC-I/research/paper improvement/k_add_analysis'
    output_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(output_dir, exist_ok=True)

    results_paths = {}
    representations = ['mobius', 'shapley']
    regularizations = ['none', 'l1', 'l2']

    for rep in representations:
        for reg in regularizations:
            key = f'{rep}_{reg}'
            path = os.path.join(base_dir, f'k_additivity_analysis_{rep}_{reg}/{dataset_name}/summary.txt')
            if os.path.exists(path):
                results_paths[key] = path

    dfs = {}
    for key, path in results_paths.items():
        dfs[key] = parse_summary_file(path)

    # Determine shared y-axis limits for bootstrap plots
    min_acc = 1.0
    max_acc = 0.0
    for key, df in dfs.items():
        if df is not None:
            cols_to_check = []
            if 'bootstrap_mean' in df.columns:
                cols_to_check.append('bootstrap_mean')
            
            for col in cols_to_check:
                if col in df and not df[col].isnull().all():
                    # Consider std for min/max calculation
                    min_val = (df[col] - df['bootstrap_std']).min()
                    max_val = (df[col] + df['bootstrap_std']).max()
                    min_acc = min(min_acc, min_val)
                    max_acc = max(max_acc, max_val)
    
    y_margin = (max_acc - min_acc) * 0.05  # 5% margin
    y_lims = (min_acc - y_margin, max_acc + y_margin) if min_acc < 1.0 else (0.0, 1.0)


    # Generate plots
    for key, df in dfs.items():
        if df is not None:
            rep, reg = key.split('_')
            plot_dataset_name = "COVID" if dataset_name == 'dados_covid_sbpo_atual' else dataset_name.replace("_", " ").title()
            
            # Plot bootstrap stability
            if 'bootstrap_mean' in df.columns and 'bootstrap_std' in df.columns:
                title_stability = f'Bootstrap Stability for {plot_dataset_name}\n({rep.title()}, Regularization: {reg.upper()})'
                output_path_stability = os.path.join(output_dir, f'bootstrap_stability_{dataset_name}_{key}.png')
                plot_bootstrap_stability(df, title_stability, output_path_stability, y_lims)

if __name__ == '__main__':
    datasets = ['dados_covid_sbpo_atual', 'pure_pairwise_interaction', 'rice']
    for dataset in datasets:
        main(dataset)

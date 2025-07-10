import pandas as pd
import os
import re
import glob
from io import StringIO

def parse_summary_txt(file_path):
    """
    Parses a summary.txt file and extracts key metrics and the full results table.
    """
    results = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        dataset_match = re.search(r"Dataset: (.*)", content)
        if dataset_match:
            results['dataset'] = dataset_match.group(1).strip()
        else:
            results['dataset'] = os.path.basename(os.path.dirname(file_path))

        best_k_acc_match = re.search(r"- Best k for accuracy: (\d+) \(accuracy: ([\d.]+)\)", content)
        if best_k_acc_match:
            results['best_k_accuracy'] = int(best_k_acc_match.group(1))
            results['accuracy'] = float(best_k_acc_match.group(2))
            # Placeholder for std, to be found in the full table
            results['accuracy_std'] = 0.0

        best_k_robust_match = re.search(r"- Best k for noise robustness: (\d+) \(accuracy at noise 0.3: ([\d.]+), std: ([\d.]+)\)", content)
        if best_k_robust_match:
            results['best_k_robustness'] = int(best_k_robust_match.group(1))
            results['robustness_accuracy'] = float(best_k_robust_match.group(2))
            results['robustness_std'] = float(best_k_robust_match.group(3))

        best_k_stability_match = re.search(r"- Best k for stability: (\d+) \(std dev: ([\d.]+)\)", content)
        if best_k_stability_match:
            results['best_k_stability'] = int(best_k_stability_match.group(1))
            results['bootstrap_stability_std'] = float(best_k_stability_match.group(2))
        else:
            # Add a default value if not found, to prevent KeyError
            results['best_k_stability'] = 'N/A'
            results['bootstrap_stability_std'] = 'N/A'
        
        full_results_match = re.search(r"FULL RESULTS TABLE:\n(.*)", content, re.DOTALL)
        if full_results_match:
            table_str = full_results_match.group(1).strip()
            # Use StringIO to treat the string as a file for pandas
            df_full = pd.read_csv(StringIO(table_str), sep=r'\s+', engine='python')
            results['full_results_df'] = df_full

            # Find std for best_k_accuracy from the full table
            if 'best_k_accuracy' in results and 'bootstrap_std' in df_full.columns and results.get('accuracy_std') == 0.0:
                best_k = results['best_k_accuracy']
                # The column in the CSV is 'k_value'
                best_k_row = df_full[df_full['k_value'] == best_k]
                if not best_k_row.empty:
                    # Get the bootstrap_std for the corresponding k
                    results['accuracy_std'] = best_k_row.iloc[0]['bootstrap_std']


    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while parsing {file_path}: {e}")
    
    return results

def generate_latex_table_from_df(df, caption=None, label=None):
    """
    Takes a pandas DataFrame and returns a LaTeX table string.
    """
    try:
        # Clean up dataset names for LaTeX
        if 'dataset' in df.columns:
            df['dataset'] = df['dataset'].str.replace('_', r'\_', regex=False)
        
        # Clean up column headers
        df.columns = [col.replace('_', ' ').title() for col in df.columns]
        
        # Rename specific columns for clarity
        df = df.rename(columns={
            'Dataset': 'Dataset',
            'Best K Accuracy': 'Best K (Acc)',
            'Accuracy (\\pm Std)': r'Accuracy ($\pm$ Std)',
            'Best K Robustness': 'Best K (Robust)',
            'Robustness Accuracy (\\pm Std)': r'Robustness Accuracy ($\pm$ Std)',
            'Best K Stability': 'Best K (Stab)',
            'Bootstrap Stability Std': 'Bootstrap Stability (Std)'
        })

        # Convert DataFrame to LaTeX tabular format
        tabular_string = df.to_latex(
            index=False, 
            float_format="%.4f", 
            escape=False
        ).strip()
        
        # Construct the full LaTeX table environment
        latex_string = rf'''\begin{{table}}[htbp]
    \begin{{adjustbox}}{{width=\textwidth,center}}
{tabular_string}
    \end{{adjustbox}}
    \caption{{{caption}}}
    \label{{{label}}}
\end{{table}}'''
        return latex_string

    except Exception as e:
        print(f"An error occurred while generating table from DataFrame: {e}")
        # Return an empty string or some indicator of failure
        return ""

if __name__ == '__main__':
    # Correct the base path to point to the 'paper improvement' directory
    base_path = os.path.join(os.path.dirname(__file__), '..')
    output_filename = os.path.join(os.path.dirname(__file__), "appendix.tex")

    # Define the subdirectories for each analysis type
    analysis_folders = {
        'shapley_none': 'k_additivity_analysis_Shapley_none',
        'shapley_l1': 'k_additivity_analysis_Shapley_l1',
        'shapley_l2': 'k_additivity_analysis_Shapley_l2'
    }

    with open(output_filename, 'w') as f:
        f.write("\\onecolumn\n")
        f.write("\\chapter{Supplementary Results}\n")
        f.write(r"% Reminder: Ensure your LaTeX document has '\usepackage{booktabs}', '\usepackage{adjustbox}', and '\usepackage{amsmath}'." + "\n\n")
        f.write("\\section{Overall Performance Summary}\n\n")
        f.write("This section summarizes the performance of the Choquistic regression model under different regularization settings (None, L1, L2). For each dataset, we report the k-value that achieves the highest accuracy, the k-value that provides the best stability under bootstrapping, and the k-value that shows the most robustness to noise in the input data. Accuracy values are presented as mean $\\pm$ standard deviation.\n\n")


        all_robustness_dfs = {}

        # Process summary.txt files for each analysis type
        for key, folder_name in sorted(analysis_folders.items()):
            analysis_dir = os.path.join(base_path, folder_name)
            
            # Find all summary.txt files in the dataset subdirectories
            summary_files = glob.glob(os.path.join(analysis_dir, '*', 'summary.txt'))
            
            all_results = []
            dataset_robustness_data = []

            for txt_file in sorted(summary_files): # Sort for consistent order
                results = parse_summary_txt(txt_file)
                if 'dataset' in results: # Only add if parsing was successful
                    all_results.append(results)
                    if 'full_results_df' in results:
                        df_full = results['full_results_df'].copy()
                        df_full['dataset'] = results['dataset']
                        dataset_robustness_data.append(df_full)

            
            if all_results:
                df = pd.DataFrame(all_results)
                
                # Combine accuracy and its std into a single column
                if 'accuracy' in df.columns and 'accuracy_std' in df.columns:
                    df[r'Accuracy ($\pm$ Std)'] = df.apply(
                        lambda row: f"{row['accuracy']:.4f} \\pm {row['accuracy_std']:.4f}" if pd.notna(row['accuracy']) and pd.notna(row['accuracy_std']) else '-',
                        axis=1
                    )
                    df = df.drop(columns=['accuracy', 'accuracy_std'])

                # Combine robustness accuracy and std into a single column
                if 'robustness_accuracy' in df.columns and 'robustness_std' in df.columns:
                    df[r'Robustness Accuracy ($\pm$ Std)'] = df.apply(
                        lambda row: f"{row['robustness_accuracy']:.4f} \\pm {row['robustness_std']:.4f}" if pd.notna(row['robustness_accuracy']) and pd.notna(row['robustness_std']) else '-',
                        axis=1
                    )
                    df = df.drop(columns=['robustness_accuracy', 'robustness_std'])

                # Define column order for the final table
                column_order = [
                    'dataset', 
                    'best_k_accuracy', r'Accuracy ($\pm$ Std)', 
                    'best_k_robustness', r'Robustness Accuracy ($\pm$ Std)',
                    'best_k_stability', 'bootstrap_stability_std'
                ]
                # Reorder and drop any other columns
                df_summary = df[[col for col in column_order if col in df.columns]]

                # Generate and write the summary LaTeX table
                caption = f"Overall Performance Summary for {key.replace('_', ' ').title()}"
                label = f"tab:summary_{key}"
                latex_summary_table = generate_latex_table_from_df(df_summary, caption, label)
                if latex_summary_table:
                    f.write(latex_summary_table)
                    f.write("\n\n\\clearpage\n\n") # Add a page break after the table

            # Store the concatenated full results for the robustness section
            if dataset_robustness_data:
                all_robustness_dfs[key] = pd.concat(dataset_robustness_data, ignore_index=True)

        # --- Section for Detailed Robustness Tables ---
        f.write("\\section{Detailed Noise Robustness Analysis}\n\n")
        f.write("This section provides a detailed breakdown of model accuracy under different levels of noise for each dataset and k-value. Each table shows the accuracy for k-values 1-15 (rows) across different noise levels (columns), with format: accuracy $\\pm$ standard deviation.\n\n")

        target_datasets = ['dados_covid_sbpo_atual', 'pure_pairwise_interaction', 'skin']
        
        for key, df_robust in sorted(all_robustness_dfs.items()):
            f.write(f"\\subsection{{Robustness Analysis for {key.replace('_', ' ').title()}}}\\n\\n")

            for dataset_name in target_datasets:
                # Use contains for flexible matching
                df_dataset = df_robust[df_robust['dataset'].str.contains(dataset_name, na=False)].copy()

                if df_dataset.empty:
                    continue

                # Create a full k-range dataframe
                k_range = pd.DataFrame({'k_value': range(1, 16)})
                df_dataset_merged = pd.merge(k_range, df_dataset, on='k_value', how='left')

                # Create a new DataFrame for the robustness table with k as rows and noise levels as columns
                df_robust_final = pd.DataFrame()
                df_robust_final['k'] = df_dataset_merged['k_value']

                # Add baseline column first
                if 'baseline_accuracy' in df_dataset_merged.columns and 'bootstrap_std' in df_dataset_merged.columns:
                    df_robust_final['Baseline'] = df_dataset_merged.apply(
                        lambda row: f"{row['baseline_accuracy']:.4f} $\\pm$ {row['bootstrap_std']:.4f}" if pd.notna(row['baseline_accuracy']) and pd.notna(row['bootstrap_std']) else '-',
                        axis=1
                    )
                else:
                    df_robust_final['Baseline'] = '-'
                
                # Dynamically find noise levels from the columns
                noise_columns = [col for col in df_dataset_merged.columns if isinstance(col, str) and col.startswith('noise_') and not col.endswith('_std')]
                noise_levels = []
                for col in noise_columns:
                    numeric_part = col.replace('noise_', '')
                    try:
                        noise_levels.append(float(numeric_part))
                    except ValueError:
                        continue
                noise_levels = sorted(list(set(noise_levels)))

                # Add columns for each noise level
                for noise_level in noise_levels:
                    acc_col = f'noise_{noise_level}'
                    std_col = f'noise_{noise_level}_std'
                    col_name = f'Noise {noise_level}'
                    
                    if acc_col in df_dataset_merged.columns and std_col in df_dataset_merged.columns:
                        df_robust_final[col_name] = df_dataset_merged.apply(
                            lambda row: f"{row[acc_col]:.4f} $\\pm$ {row[std_col]:.4f}" if pd.notna(row[acc_col]) and pd.notna(row[std_col]) else '-',
                            axis=1
                        )
                    else:
                        df_robust_final[col_name] = '-'

                # Generate and write the detailed robustness LaTeX table
                clean_dataset_name = dataset_name.replace('_', ' ').title()
                caption_robust = f"Noise Robustness Analysis: {clean_dataset_name} with {key.replace('_', ' ').title()} Regularization"
                label_robust = f"tab:robustness_{key}_{dataset_name.replace('_', '')}"
                latex_robust_table = generate_latex_table_from_df(df_robust_final, caption_robust, label_robust)
                if latex_robust_table:
                    f.write(latex_robust_table)
                    f.write("\n\n")
            
            f.write("\\clearpage\n\n") # Add a page break after each analysis type's section

    print(f"LaTeX appendix successfully generated at {output_filename}")

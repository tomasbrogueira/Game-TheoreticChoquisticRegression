import pandas as pd
import os
import re
import glob

def parse_summary_txt(file_path):
    """
    Parses a summary.txt file and extracts key metrics.
    """
    results = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        dataset_match = re.search(r"Dataset: (.*)", content)
        if dataset_match:
            results['dataset'] = dataset_match.group(1).strip()

        best_k_acc_match = re.search(r"- Best k for accuracy: (\d+) \(accuracy: ([\d.]+)\)", content)
        if best_k_acc_match:
            results['best_k_accuracy'] = int(best_k_acc_match.group(1))
            results['accuracy'] = float(best_k_acc_match.group(2))

        best_k_robust_match = re.search(r"- Best k for noise robustness: (\d+) \(accuracy at noise 0.3: ([\d.]+), std: ([\d.]+)\)", content)
        if best_k_robust_match:
            results['best_k_robustness'] = int(best_k_robust_match.group(1))
            results['robustness_accuracy'] = float(best_k_robust_match.group(2))
            results['robustness_std'] = float(best_k_robust_match.group(3))
        
        if 'dataset' not in results:
             # Fallback to get dataset name from path if not in file
            results['dataset'] = os.path.basename(os.path.dirname(file_path))

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
    return None

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
        f.write(r"% Reminder: Ensure your LaTeX document has '\usepackage{booktabs}' and '\usepackage{adjustbox}'." + "\n\n")
        f.write("\\section{Overall Performance Summary}\n\n")

        # Process summary.txt files for each analysis type
        for key, folder_name in sorted(analysis_folders.items()):
            analysis_dir = os.path.join(base_path, folder_name)
            
            # Find all summary.txt files in the dataset subdirectories
            summary_files = glob.glob(os.path.join(analysis_dir, '*', 'summary.txt'))
            
            all_results = []
            for txt_file in sorted(summary_files): # Sort for consistent order
                results = parse_summary_txt(txt_file)
                if 'dataset' in results: # Only add if parsing was successful
                    all_results.append(results)
            
            if all_results:
                df = pd.DataFrame(all_results)
                
                # Combine robustness accuracy and std into a single column
                if 'robustness_accuracy' in df.columns and 'robustness_std' in df.columns:
                    df['Robustness Accuracy ($\pm$ Std)'] = df.apply(
                        lambda row: f"{row['robustness_accuracy']:.4f} $\pm$ {row['robustness_std']:.4f}",
                        axis=1
                    )
                    df = df.drop(columns=['robustness_accuracy', 'robustness_std'])

                # Define column order for the final table
                column_order = [
                    'dataset', 
                    'best_k_accuracy', 'accuracy', 
                    'best_k_robustness', 'Robustness Accuracy ($\pm$ Std)'
                ]
                
                # Filter and reorder dataframe to match desired columns
                df = df[[col for col in column_order if col in df.columns]]

                # Create title, caption, and label for the LaTeX table
                analysis_type = key.replace('_', ' ')
                subsection_title = analysis_type.title()
                intro = f"\\subsection{{Summary for {subsection_title}}}\n"
                caption = f"Summary of k-additivity analysis for {analysis_type}, generated from summary.txt files."
                label = f"tab:summary_{key}"
                
                f.write(intro)
                table_string = generate_latex_table_from_df(df, caption=caption, label=label)
                if table_string:
                    f.write(table_string + "\n\n")
            else:
                print(f"Warning: No summary.txt files found or parsed for {key}")

    print(f"Successfully generated minimal LaTeX appendix at: {output_filename}")

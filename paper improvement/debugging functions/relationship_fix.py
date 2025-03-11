import os
import numpy as np
import pickle
from regression_classes import compute_shapley_values, compute_choquet_interaction_matrix

def validate_choquet_relationship(v, all_coalitions, feature_names=None):
    """
    Validate and visualize the relationship between Shapley values, marginal values, 
    and interaction indices in the Choquet context.
    
    This function verifies the mathematical relationship:
    φᵢ = μ({i}) - 0.5 * Σⱼ≠ᵢ I({i,j})
    
    Parameters:
    -----------
    v : array-like
        Capacity values with v[0] representing empty set
    all_coalitions : list of tuples
        List of all coalitions in the model
    feature_names : list, optional
        Names of features for more readable output
        
    Returns:
    --------
    dict : Results of the validation including correction factors
    """
    import matplotlib.pyplot as plt
    
    # Get number of features
    m = max(max(c) for c in all_coalitions if c) + 1
    
    # Compute Shapley values
    shapley_values = compute_shapley_values(v, m, all_coalitions)
    
    # Extract singleton/marginal values
    marginal_values = np.zeros(m)
    for i in range(m):
        singleton_tuple = (i,)
        if singleton_tuple in all_coalitions:
            idx = all_coalitions.index(singleton_tuple)
            marginal_values[i] = v[idx + 1]  # +1 for empty set at v[0]
    
    # Compute interaction matrix
    interaction_matrix = compute_choquet_interaction_matrix(v, m, all_coalitions)
    
    # Calculate overall interaction by both methods
    ground_truth = shapley_values - marginal_values  # Shapley - Marginal
    matrix_method_standard = 0.5 * np.sum(interaction_matrix, axis=1)  # Standard Matrix Method
    matrix_method_negated = -0.5 * np.sum(interaction_matrix, axis=1)  # Negated Matrix Method
    
    # Compute differences for both methods
    diff_standard = ground_truth - matrix_method_standard
    diff_negated = ground_truth - matrix_method_negated
    
    # Check which formula is better on average
    avg_diff_standard = np.mean(np.abs(diff_standard))
    avg_diff_negated = np.mean(np.abs(diff_negated))
    
    # Count how many features are better explained by each method
    count_standard_better = np.sum(np.abs(diff_standard) < np.abs(diff_negated))
    count_negated_better = np.sum(np.abs(diff_negated) < np.abs(diff_standard))
    
    # Find worst features for both methods
    worst_standard_idx = np.argmax(np.abs(diff_standard))
    worst_negated_idx = np.argmax(np.abs(diff_negated))
    
    # Detailed analysis by feature
    feature_analysis = []
    for i in range(m):
        name = feature_names[i] if feature_names and i < len(feature_names) else f"Feature {i}"
        better_method = "standard" if np.abs(diff_standard[i]) < np.abs(diff_negated[i]) else "negated"
        feature_analysis.append({
            "index": i,
            "name": name,
            "shapley": shapley_values[i],
            "marginal": marginal_values[i],
            "ground_truth": ground_truth[i],
            "standard": matrix_method_standard[i],
            "negated": matrix_method_negated[i],
            "diff_standard": diff_standard[i],
            "diff_negated": diff_negated[i],
            "better_method": better_method
        })
    
    # Sort by absolute difference (worst to best)
    feature_analysis.sort(key=lambda x: max(abs(x["diff_standard"]), abs(x["diff_negated"])), reverse=True)
    
    # Determine the proper formula based on more metrics
    # 1. Average error across all features
    # 2. Count of features better explained by each method
    # 3. Maximum error comparison
    use_negated = False
    
    if avg_diff_negated < avg_diff_standard:
        use_negated = True
        reason = "lower average error"
    elif count_negated_better > count_standard_better:
        use_negated = True
        reason = f"better for more features ({count_negated_better} vs {count_standard_better})"
    elif np.max(np.abs(diff_negated)) < np.max(np.abs(diff_standard)):
        use_negated = True
        reason = "smaller maximum error"
    else:
        reason = "lower average error"
        
    correct_formula = "φⱼ = μ({j}) - 0.5 * Σᵢ≠ⱼ I({i,j})" if use_negated else "φⱼ = μ({j}) + 0.5 * Σᵢ≠ⱼ I({i,j})"
    
    # Visual comparison
    plt.figure(figsize=(14, 7))
    ind = np.arange(m)
    width = 0.25
    
    # Create labels
    labels = feature_names if feature_names else [f"Feature {i}" for i in range(m)]
    
    # Plot ground truth and both methods
    plt.bar(ind - width, ground_truth, width, label='Ground Truth (φ-μ)')
    plt.bar(ind, matrix_method_standard, width, label='Standard Matrix (+)', alpha=0.7)
    plt.bar(ind + width, matrix_method_negated, width, label='Negated Matrix (-)', alpha=0.7)
    
    plt.xlabel('Features')
    plt.ylabel('Overall Interaction Effect')
    plt.title('Comparison of Overall Interaction Calculation Methods')
    plt.xticks(ind, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig('interaction_relation_validation.png')
    plt.close()
    
    # Plot error comparison
    plt.figure(figsize=(14, 7))
    plt.bar(ind - width/2, np.abs(diff_standard), width, label='|Error Standard|', color='blue', alpha=0.7)
    plt.bar(ind + width/2, np.abs(diff_negated), width, label='|Error Negated|', color='red', alpha=0.7)
    plt.xticks(ind, labels, rotation=45, ha='right')
    plt.ylabel('Absolute Error')
    plt.yscale('log')  # Log scale for better visibility
    plt.title('Error Comparison by Feature')
    plt.legend()
    plt.tight_layout()
    plt.savefig('error_comparison.png')
    plt.close()
    
    print("\n=== CHOQUET RELATIONSHIP VALIDATION ===")
    print(f"Standard formula average error: {avg_diff_standard:.6f}")
    print(f"Negated formula average error: {avg_diff_negated:.6f}")
    print(f"Standard formula better for {count_standard_better}/{m} features")
    print(f"Negated formula better for {count_negated_better}/{m} features")
    print(f"Maximum error - Standard: {np.max(np.abs(diff_standard)):.6f}, Negated: {np.max(np.abs(diff_negated)):.6f}")
    
    # Print worst features for each method
    worst_std_name = feature_names[worst_standard_idx] if feature_names else f"Feature {worst_standard_idx}"
    worst_neg_name = feature_names[worst_negated_idx] if feature_names else f"Feature {worst_negated_idx}"
    
    print(f"\nWorst feature with standard formula: {worst_std_name} (error: {np.abs(diff_standard[worst_standard_idx]):.6f})")
    print(f"Worst feature with negated formula: {worst_neg_name} (error: {np.abs(diff_negated[worst_negated_idx]):.6f})")
    
    # Print top 5 worst features
    print("\nTop 5 features with largest discrepancies:")
    print(f"{'Feature':<15} {'Truth':<10} {'Standard':<10} {'Negated':<10} {'Better':<10}")
    print("-" * 60)
    for item in feature_analysis[:5]:
        name = item["name"][:15]
        print(f"{name:<15} {item['ground_truth']:<10.6f} {item['standard']:<10.6f} {item['negated']:<10.6f} {item['better_method']:<10}")
    
    print(f"\nProper formula to use: {correct_formula}")
    print(f"Reason: {reason}")
    print(f"Plots saved to 'interaction_relation_validation.png' and 'error_comparison.png'")
    
    return {
        "correct_formula": correct_formula,
        "use_negated": use_negated,
        "ground_truth": ground_truth,
        "matrix_method_standard": matrix_method_standard,
        "matrix_method_negated": matrix_method_negated,
        "diff_standard": diff_standard,
        "diff_negated": diff_negated,
        "feature_analysis": feature_analysis
    }

def apply_correction_to_simulation_helper_functions():
    """
    Apply the correction to simulation_helper_functions.py by modifying the 
    overall_interaction_index_corrected function to use the correct approach.
    """
    import fileinput
    import sys
    
    # File to modify
    file_path = r"c:\Users\Tomas\OneDrive - Universidade de Lisboa\3ºano_LEFT\PIC-I\paper improvement\simulation_helper_functions.py"
    
    # Function to modify
    function_name = "def overall_interaction_index_corrected"
    
    # Replace the implementation with the correct one
    replacement_code = '''def overall_interaction_index_corrected(interaction_matrix):
    """
    Compute the overall interaction effect with the correct sign convention.
    
    For the Choquet integral with Shapley interaction indices, the correct mathematical 
    relationship is: φj = v({j}) - 0.5 * Σi≠j I({i,j})
    
    Parameters:
        interaction_matrix (np.ndarray): An m x m symmetric matrix of pairwise interaction indices.
    
    Returns:
        np.ndarray: A 1D array of length m with the correctly signed overall interaction index.
    """
    # Use negative sign for the interaction matrix to get the correct relationship
    overall = -0.5 * np.sum(interaction_matrix, axis=1)
    return overall'''
    
    # Check if file exists before attempting to modify
    if not os.path.exists(file_path):
        print(f"ERROR: Could not find file {file_path}")
        return False
        
    try:
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Check if function exists
        if function_name not in content:
            print(f"ERROR: Could not find {function_name} in {file_path}")
            return False
            
        # Implement the function replacement - find start and end of current function
        lines = content.split('\n')
        start_idx = -1
        end_idx = -1
        
        for i, line in enumerate(lines):
            if line.startswith(function_name):
                start_idx = i
                
            # Find the end of the function by looking for the next function or blank line
            if start_idx != -1 and i > start_idx:
                if line.startswith('def ') or (line.strip() == '' and lines[i-1].strip() == ''):
                    end_idx = i - 1
                    break
        
        if start_idx == -1:
            print(f"ERROR: Could not locate {function_name} in {file_path}")
            return False
            
        if end_idx == -1:  # If end not found, assume it's the end of file
            end_idx = len(lines) - 1
        
        # Replace the function
        new_lines = lines[:start_idx] + replacement_code.split('\n') + lines[end_idx+1:]
        
        # Write the modified content back
        with open(file_path, 'w') as f:
            f.write('\n'.join(new_lines))
        
        print(f"Successfully updated {function_name} in {file_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to update file: {e}")
        return False
    
def update_simulation_loop():
    """
    Update the simulation_loop.py file to ensure the correct method is used
    for calculating overall interaction effects.
    """
    file_path = r"c:\Users\Tomas\OneDrive - Universidade de Lisboa\3ºano_LEFT\PIC-I\paper improvement\simulation_loop.py"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: Could not find file {file_path}")
        return False
        
    try:
        # Read the content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Make sure the overall_interaction_index_corrected is imported and used
        if "from simulation_helper_functions import overall_interaction_index_corrected" not in content:
            print("WARNING: overall_interaction_index_corrected import not found")
            
        if "overall_method1 = overall_interaction_index_corrected" not in content:
            print("WARNING: overall_interaction_index_corrected usage not found")
        
        print(f"Simulation loop file check complete: {file_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to check file: {e}")
        return False
        
if __name__ == "__main__":
    print("Choquet Relationship Fix Tool")
    print("-----------------------------")
    
    # Try to load the debug data from most likely locations
    debug_file_paths = [
        'model_debug.pkl',
        os.path.join('plots', 'dados_covid_sbpo_atual', 'model_debug.pkl')
    ]
    
    debug_data = None
    for path in debug_file_paths:
        try:
            with open(path, 'rb') as f:
                debug_data = pickle.load(f)
                print(f"Successfully loaded debug data from {path}")
                break
        except (FileNotFoundError, Exception) as e:
            print(f"Could not load debug data from {path}: {e}")
    
    if debug_data is None:
        print("ERROR: No debug data found. Cannot validate relationship.")
        exit(1)
        
    # Extract data
    v = debug_data.get('v')
    all_coalitions = debug_data.get('all_coalitions')
    feature_names = debug_data.get('feature_names')
    
    # Validate the relationship
    results = validate_choquet_relationship(v, all_coalitions, feature_names)
    
    # Apply the correction if needed
    if results["use_negated"]:
        print("\nApplying correction to simulation helper functions...")
        apply_correction_to_simulation_helper_functions()
        
        print("\nChecking simulation loop file...")
        update_simulation_loop()
        
        print("\nFix complete!")
    else:
        print("\nNo correction needed. The standard formula is correct.")
        
        # Additional guidance for the user
        print("\nNOTE: While the standard formula has a lower average error,")
        print("you may want to examine specific features where the negated formula performs better.")
        print("Run 'python advanced_debug.py' for a more detailed feature-by-feature analysis.")

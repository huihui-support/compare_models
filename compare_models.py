from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from tqdm import tqdm

# Set environment variables and thread configuration
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)

def calculate_weight_diff(weight1, weight2):
    """Calculate the mean absolute difference between two weights."""
    return torch.abs(weight1 - weight2).mean().item()

def get_weight_components(module):
    """Extract weight components' names and weights from a module, removing .weight suffix."""
    components = {}
    for name, param in module.named_parameters():
        if 'weight' in name.lower():  # Only include weight parameters
            # Remove .weight or .Weight suffix (case-insensitive)
            clean_name = name
            if clean_name.endswith('.weight'):
                clean_name = clean_name[:-7]  # Remove .weight
            elif clean_name.endswith('.Weight'):
                clean_name = clean_name[:-7]  # Remove .Weight
            components[clean_name] = param
    return components

def visualize_layer_diffs(layer_diffs, global_diffs, model_name1, model_name2):
    """Visualize layer and global weight differences with heatmaps, adding border padding."""
    num_layers = len(layer_diffs)
    num_components = len(layer_diffs[0]) + len(global_diffs)  # Total components including global weights

    # Set figure size with narrow bars (width coefficient 2) and fixed height
    fig, axs = plt.subplots(1, num_components, figsize=(3 * num_components, 8))
    fig.suptitle(f"{model_name1} <> {model_name2}", fontsize=16)

    # Plot layer weight differences
    for i, component in enumerate(layer_diffs[0].keys()):
        component_diffs = [[layer_diff[component]] for layer_diff in layer_diffs]
        sns.heatmap(component_diffs, annot=True, fmt=".6f", cmap="YlGnBu", ax=axs[i], cbar_kws={"shrink": 0.8})
        axs[i].set_title(component)
        axs[i].set_xlabel("Layer")
        axs[i].set_ylabel("Difference")
        axs[i].set_xticks([])
        axs[i].set_yticks(range(num_layers))
        axs[i].set_yticklabels(range(num_layers))
        axs[i].invert_yaxis()

    # Plot global weight differences
    start_idx = len(layer_diffs[0].keys())
    for j, (global_component, diff_value) in enumerate(global_diffs.items()):
        sns.heatmap([[diff_value]], annot=True, fmt=".6f", cmap="YlGnBu", ax=axs[start_idx + j], cbar_kws={"shrink": 0.8})
        axs[start_idx + j].set_title(global_component)
        axs[start_idx + j].set_xlabel("Global")
        axs[start_idx + j].set_ylabel("Difference")
        axs[start_idx + j].set_xticks([])
        axs[start_idx + j].set_yticks([])
    
    # Add padding to keep distance from borders
    plt.tight_layout(pad=3.0)
    
    # Generate output filename based on model names
    output_file = f"{model_name1.replace('/', '-')}_vs_{model_name2.replace('/', '-')}_diffs.png"
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.2)  # Ensure saved image retains padding
    plt.show()

def cmp_model(model_name1, model_name2, plot=False):
    """Compare weights of two models, including layers and top-level parameters."""
    # Load models
    model1 = AutoModelForCausalLM.from_pretrained(
        model_name1, device_map='cpu', torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model2 = AutoModelForCausalLM.from_pretrained(
        model_name2, device_map='cpu', torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    
    results = {'layers': [], 'top_level': {}}
    
    # Compare top-level parameters (excluding layers)
    print("Comparing top-level parameters...")
    top_level_diff = {}
    model1_components = get_weight_components(model1)
    model2_components = get_weight_components(model2)
    common_components = set(model1_components.keys()) & set(model2_components.keys())
    
    # Exclude layer-related parameters
    for comp_name in common_components:
        if 'layers' not in comp_name:
            top_level_diff[comp_name] = calculate_weight_diff(
                model1_components[comp_name],
                model2_components[comp_name]
            )
    results['top_level'] = top_level_diff
    
    # Compare layers
    total_layers = len(model1.model.layers)
    for layer_idx, (layer1, layer2) in enumerate(tqdm(zip(model1.model.layers, model2.model.layers), 
                                                    total=total_layers, 
                                                    desc="Comparing layers")):
        layer_diff = {}
        
        # Extract weight components for the layer
        components1 = get_weight_components(layer1)
        components2 = get_weight_components(layer2)
        
        # Calculate differences for common components
        common_components = set(components1.keys()) & set(components2.keys())
        for comp_name in common_components:
            layer_diff[comp_name] = calculate_weight_diff(
                components1[comp_name],
                components2[comp_name]
            )
        
        results['layers'].append(layer_diff)
    
    # Visualize if enabled
    if plot:
        visualize_layer_diffs(results['layers'], results['top_level'], model_name1, model_name2)
    
    return results

def main():
    """Main function to parse arguments and run model comparison."""
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Compare weights of two transformer models.")
    parser.add_argument('--model1', type=str, required=True, 
                        help="Name or path of the first model")
    parser.add_argument('--model2', type=str, required=True, 
                        help="Name or path of the second model")
    parser.add_argument('--plot', action='store_true', 
                        help="Enable plotting of weight differences")
    
    args = parser.parse_args()
    
    # Run model comparison
    results = cmp_model(args.model1, args.model2, args.plot)
    
    # Print results
    print("\nTop-level parameters differences:")
    for comp, value in results['top_level'].items():
        print(f"  {comp}: {value:.6f}")
    
    for i, diff in enumerate(results['layers']):
        print(f"\nLayer {i} differences:")
        for comp, value in diff.items():
            print(f"  {comp}: {value:.6f}")

if __name__ == "__main__":
    main()
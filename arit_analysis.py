import argparse
import copy
import os
import random
import torch
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import confusion_matrix

# Import necessary modules from our ARIT codebase
from arit_citations import MultiHeadCitationPredictor
from arit_config import ARITConfig
from arit_model import ARITModel
from arit_environment import ARITEnvironment, ARITState
from arit_evaluation import CitationMetrics
from start_training import setup_model_and_optimizer

def parse_args():
    parser = argparse.ArgumentParser(description='ARIT Model Analysis')
    parser.add_argument('--model-type', type=str, choices=['high', 'low', 'both'], required=True,
                      help='Type of model to analyze: high citation, low citation, or both')
    parser.add_argument('--folder', type=str, required=True,
                      help='Path to the model results folder')
    parser.add_argument('--data-dir', type=str, default='./arit_data',
                      help='Directory containing the ARIT data')
    parser.add_argument('--threshold', type=int, default=20,
                      help='Citation threshold separating low and high models')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for analysis results (defaults to analysis_results in the model folder)')
    parser.add_argument('--analyses', type=str, nargs='+', 
                      default=['all'],
                      help='List of analyses to run: prediction, field, strategy, temporal, network, importance, error, comparison, impact, robustness, or all')
    
    return parser.parse_args()

def setup_output_directory(base_folder, model_type):
    """Create an output directory for analysis results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_type == 'both':
        output_dir = os.path.join(base_folder, f'analysis_results_{timestamp}_combined')
    else:
        output_dir = os.path.join(base_folder, f'analysis_results_{timestamp}_{model_type}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each analysis type
    analysis_types = [
        'prediction', 'field', 'strategy', 'temporal', 
        'network', 'importance', 'error', 'comparison', 
        'impact', 'robustness'
    ]
    
    for analysis_type in analysis_types:
        os.makedirs(os.path.join(output_dir, analysis_type), exist_ok=True)
    
    return output_dir

def fix_collaboration_info(data, model_type):
    """Add or fix collaboration information in the model's states"""
    print(f"Fixing collaboration information for {model_type} model...")
    
    # Extract data
    states = data['states']
    citation_network = data['citation_network']
    state_id_to_arxiv_id = data['state_id_to_arxiv_id']
    
    # Check if citation network is available
    if citation_network is None:
        print("Citation network not available for institution data. Using citation count as a proxy.")
    
    modified_count = 0
    paper_found_count = 0
    simulated_count = 0
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Process each state
    for state_id, state in states.items():
        # Get the paper_id (arxiv_id) for this state
        paper_id = state_id_to_arxiv_id.get(state_id)
        paper_data = None
        
        if paper_id and citation_network and paper_id in citation_network:
            paper_data = citation_network.get(paper_id, {})
            paper_found_count += 1
        
        # If we have real author data from the citation network
        if paper_data and 'authors' in paper_data and paper_data['authors']:
            # Calculate a real collaboration score
            num_authors = len(paper_data['authors'])
            
            # Use citation count as a proxy for institutional diversity
            # Papers with more citations tend to have more institutional diversity
            citation_factor = min(state.citation_count / 50, 1.0) 
            
            # Combine factors - more authors usually means more institutions
            collab_score = 0.2 + (0.4 * min(num_authors / 5, 1.0)) + (0.4 * citation_factor)
            
            # Ensure the score is in [0,1] range
            state.collaboration_info = min(max(collab_score, 0.1), 1.0)
            modified_count += 1
        else:
            # Simulate a reasonable collaboration info value based on citation count
            # Papers with higher citation counts tend to have more collaborators
            citation_factor = min(state.citation_count / 50, 1.0)
            
            # Papers with higher reference diversity also tend to have more collaborators
            diversity_factor = min(state.reference_diversity * 2, 1.0)
            
            # Add some random variation to prevent all values being the same
            random_component = np.random.uniform(0.05, 0.15)
            
            # Combine factors into a meaningful collaboration score
            collab_score = 0.1 + (0.4 * citation_factor) + (0.4 * diversity_factor) + (0.1 * random_component)
            
            # Ensure the score is in [0,1] range
            state.collaboration_info = min(max(collab_score, 0.1), 1.0)
            simulated_count += 1
    
    print(f"Fixed collaboration information for {modified_count + simulated_count} states")
    print(f"Used real paper data for {paper_found_count} papers")
    print(f"Used simulated collaboration metrics for {simulated_count} papers")
    
    return modified_count + simulated_count > 0

def load_model_and_data(args, model_type):
    """Load model and data based on model type and folder path"""
    print(f"Loading {model_type} citation model from {args.folder}...")
    
    # Determine model path based on model type
    if model_type == 'high':
        model_path = os.path.join(args.folder, 'trained_high_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.folder, 'rl_training', 'high_model', 'trained_high_model.pt')
    else:  # low model
        model_path = os.path.join(args.folder, 'trained_low_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.folder, 'rl_training', 'low_model', 'trained_low_model.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model at {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    config = ARITConfig.from_dict(checkpoint['config']) if isinstance(checkpoint['config'], dict) else checkpoint['config']
    
    # --- Load data FIRST ---
    data_dir = args.data_dir
    processed_dir = os.path.join(data_dir, 'processed')
    
    # Load metadata
    with open(os.path.join(processed_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Load train and validation states
    with open(os.path.join(processed_dir, 'train_states.pkl'), 'rb') as f:
        train_states = pickle.load(f)
    with open(os.path.join(processed_dir, 'val_states.pkl'), 'rb') as f:
        val_states = pickle.load(f)
    
    # Convert to dictionary of ARITState objects using unpacking
    def filter_state_data(state_data):
        valid_keys = {
            "content_embedding", "field_centroid", "reference_diversity", "citation_count",
            "field_impact_factor", "collaboration_info", "time_index", "state_id",
            "future_citations", "emerging_topics", "field_saturation", "strategy_memory",
            "network_data", "primary_category", "field_target"  # Exclude paper_id
        }
        return {k: v for k, v in state_data.items() if k in valid_keys}

    # Convert to dictionary of ARITState objects and build mapping
    states = {}
    state_id_to_arxiv_id = {}
    sample_count = 0
    max_samples = 20  # Limit to 20 samples
    for state_data in train_states + val_states:
        state_id = state_data['state_id']
        states[state_id] = ARITState(**filter_state_data(state_data))
        if 'paper_id' in state_data:
            state_id_to_arxiv_id[state_id] = state_data['paper_id']
            if sample_count < max_samples:
                print(f"DEBUG: state_id={state_id}, mapped paper_id={state_data['paper_id']}")
                sample_count += 1
        else:
            if sample_count < max_samples:
                print(f"WARNING: No paper_id found for state_id={state_id}")
                sample_count += 1

    # Identify train and validation state IDs
    train_state_ids = [s['state_id'] for s in train_states]
    val_state_ids = [s['state_id'] for s in val_states]
    
    # Create citation targets dictionary
    citation_targets = {}
    for state_data in train_states + val_states:
        state_id = state_data['state_id']
        citation_targets[state_id] = state_data['future_citations']
    
    # Create field targets dictionary
    field_targets = {}
    for state_data in train_states + val_states:
        state_id = state_data['state_id']
        field_targets[state_id] = state_data['field_target']
    
    # Load citation network if available
    citation_network = None
    if metadata.get('has_citation_network', False):
        try:
            with open(os.path.join(processed_dir, 'citation_network.pkl'), 'rb') as f:
                citation_network = pickle.load(f)
            print("Loaded citation network data")
            print(f"DEBUG: Citation network keys sample (first 20): {list(citation_network.keys())[:20]}")
        except Exception as e:
            print(f"Citation network data not available: {str(e)}")
    
    # Organize all data into a dictionary
    data = {
        'metadata': metadata,
        'states': states,
        'train_states': train_state_ids,
        'val_states': val_state_ids,
        'citation_targets': citation_targets,
        'field_targets': field_targets,
        'citation_network': citation_network,
        'field_to_id': metadata.get('field_to_id', {}),
        'id_to_field': {v: k for k, v in metadata.get('field_to_id', {}).items()},
        'state_id_to_arxiv_id': state_id_to_arxiv_id
    }
    print(f"DEBUG: Created state_id_to_arxiv_id mapping with {len(state_id_to_arxiv_id)} entries")
    
    # --- End Data Loading ---
    
    # Compute state dimension from a sample state
    first_state = next(iter(data['states'].values()))
    state_dim = len(first_state.to_numpy())
    
    # Create the model with the same configuration as during training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = setup_model_and_optimizer(config, state_dim, device)
    
    # Load model weights and set to evaluation mode
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, data, config

def filter_by_citation_count(states, citation_targets, field_targets, threshold, model_type):
    """Filter states based on citation count threshold for the appropriate model"""
    filtered_states = {}
    filtered_citation_targets = {}
    filtered_field_targets = {}
    
    for state_id, state in states.items():
        # For low model, keep only papers with current and future citations <= threshold
        if model_type == 'low' and (state.citation_count <= threshold and max(citation_targets[state_id]) <= threshold):
            filtered_states[state_id] = state
            filtered_citation_targets[state_id] = citation_targets[state_id]
            filtered_field_targets[state_id] = field_targets[state_id]
        
        # For high model, keep only papers with current or future citations > threshold
        elif model_type == 'high' and (state.citation_count > threshold or max(citation_targets[state_id]) > threshold):
            filtered_states[state_id] = state
            filtered_citation_targets[state_id] = citation_targets[state_id]
            filtered_field_targets[state_id] = field_targets[state_id]
    
    return filtered_states, filtered_citation_targets, filtered_field_targets

def run_model_on_dataset(model, states, device='cpu', batch_size=64):
    """Run model on a dataset and return predictions"""
    model.eval()
    all_predictions = {}
    all_actions = {}
    all_values = {}
    all_field_preds = {}
    
    # Convert states to list for batch processing
    state_ids = list(states.keys())
    
    with torch.no_grad():
        for i in range(0, len(state_ids), batch_size):
            batch_ids = state_ids[i:i+batch_size]
            batch_states = [states[sid].to_numpy() for sid in batch_ids]
            
            # Convert to tensor (improve performance by converting the list into a NumPy array first)
            state_tensor = torch.tensor(np.array(batch_states), dtype=torch.float32, device=device).unsqueeze(1)
            
            # Forward pass: policy outputs, value, citation parameters, field logits
            policy_outputs, values, citation_params, field_logits = model(state_tensor)
            
            # Get horizon predictions for citation (unchanged)
            horizon_predictions = []
            for h in range(model.horizons):
                nb_dist, mean = model.get_citation_distribution(citation_params, h)
                horizon_predictions.append(mean)
            batch_predictions = torch.stack(horizon_predictions, dim=1)
            if hasattr(model, 'use_log_transform') and model.use_log_transform:
                batch_predictions = torch.expm1(batch_predictions)  # convert back to original space
            
            # Instead of using a non-existent get_action_components method,
            # sample the actions for the entire batch.
            actions = model.sample_action(policy_outputs)
            
            for j, sid in enumerate(batch_ids):
                all_predictions[sid] = batch_predictions[j].cpu().numpy()
                # Extract the j-th action's components from the ARITAction object.
                all_actions[sid] = {
                    "field_positioning": actions.field_positioning[j],
                    "novelty_level": actions.novelty_level[j],
                    "collaboration_strategy": actions.collaboration_strategy[j],
                    "citation_choices": actions.citation_choices[j],
                    "combined_focus": actions.combined_focus[j],
                    "timing": actions.timing[j]
                }
                all_values[sid] = values[j].cpu().item()
                all_field_preds[sid] = torch.argmax(field_logits[j], dim=0).cpu().item()
    
    return {
        'predictions': all_predictions,
        'actions': all_actions,
        'values': all_values,
        'field_preds': all_field_preds
    }


def analyze_citation_prediction(model, data, model_results, output_dir, model_type, threshold):
    """Analyze citation prediction accuracy with detailed breakdowns by citation range and field"""
    print(f"Running citation prediction analysis for {model_type} model...")
    prediction_dir = os.path.join(output_dir, 'prediction')
    
    # Extract data
    states = data['states']
    citation_targets = data['citation_targets']
    val_state_ids = data['val_states']
    predictions = model_results['predictions']
    field_targets = data['field_targets']
    id_to_field = data['id_to_field']
    
    # Prepare validation data
    val_predictions = []
    val_targets = []
    citation_counts = []
    paper_fields = []
    horizons = []
    
    for state_id in val_state_ids:
        if state_id not in predictions or state_id not in citation_targets:
            continue
        
        val_predictions.append(predictions[state_id])
        val_targets.append(citation_targets[state_id])
        citation_counts.append(states[state_id].citation_count)
        
        # Get field information if available
        if state_id in field_targets:
            field_id = field_targets[state_id]
            field_name = id_to_field.get(field_id, f"Field {field_id}")
            paper_fields.append(field_name)
        else:
            paper_fields.append("Unknown")
            
        # Track horizons for temporal analysis
        horizons.append(list(range(len(citation_targets[state_id]))))
    
    # Convert to numpy arrays for easier analysis
    val_predictions = np.array(val_predictions)
    val_targets = np.array(val_targets)
    citation_counts = np.array(citation_counts)
    
    # Convert to tensors for metrics computation
    val_predictions_tensor = torch.tensor(val_predictions)
    val_targets_tensor = torch.tensor(val_targets)
    
    # Compute overall metrics
    metrics = CitationMetrics().compute(val_predictions_tensor, val_targets_tensor)
    
    # Save metrics to file
    with open(os.path.join(prediction_dir, 'citation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 1. Create error histogram
    plt.figure(figsize=(10, 6))
    errors = np.abs(val_predictions - val_targets).mean(axis=1)
    sns.histplot(errors, bins=30, kde=True)
    plt.title(f"{model_type.capitalize()} Model - Prediction Error Distribution")
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Count")
    plt.savefig(os.path.join(prediction_dir, 'error_distribution.png'))
    plt.close()
    
    # 2. Analyze error by citation range
    citation_bins = [0, 5, 10, 20, 50, 100, 500]
    bin_labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '101-500']
    
    # Get bin indices for each paper
    bin_indices = np.digitize(citation_counts, citation_bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)  # Ensure valid indices
    
    # Calculate error statistics by citation bin
    bin_errors = {}
    for i, label in enumerate(bin_labels):
        mask = (bin_indices == i)
        if not any(mask):
            continue
            
        bin_predictions = val_predictions[mask]
        bin_targets = val_targets[mask]
        bin_mae = np.abs(bin_predictions - bin_targets).mean()
        bin_rmse = np.sqrt(((bin_predictions - bin_targets) ** 2).mean())
        
        bin_errors[label] = {
            'count': np.sum(mask),
            'mae': float(bin_mae),
            'rmse': float(bin_rmse)
        }
    
    # Plot error by citation range
    plt.figure(figsize=(12, 6))
    labels = list(bin_errors.keys())
    mae_values = [bin_errors[label]['mae'] for label in labels]
    counts = [bin_errors[label]['count'] for label in labels]
    
    bars = plt.bar(labels, mae_values, color='skyblue')
    
    # Add count annotations
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'n={counts[i]}', ha='center', va='bottom', fontsize=9)
    
    plt.title(f"{model_type.capitalize()} Model - MAE by Citation Range")
    plt.xlabel("Paper Citation Count")
    plt.ylabel("Mean Absolute Error")
    plt.savefig(os.path.join(prediction_dir, 'error_by_citation_range.png'))
    plt.close()
    
    # 3. Analyze error by field
    field_errors = {}
    for field in set(paper_fields):
        field_mask = np.array([f == field for f in paper_fields])
        if not any(field_mask):
            continue
            
        field_predictions = val_predictions[field_mask]
        field_targets = val_targets[field_mask]
        field_mae = np.abs(field_predictions - field_targets).mean()
        field_rmse = np.sqrt(((field_predictions - field_targets) ** 2).mean())
        
        field_errors[field] = {
            'count': np.sum(field_mask),
            'mae': float(field_mae),
            'rmse': float(field_rmse)
        }
    
    # Plot error by field
    plt.figure(figsize=(14, 7))
    fields = list(field_errors.keys())
    field_mae = [field_errors[field]['mae'] for field in fields]
    field_counts = [field_errors[field]['count'] for field in fields]
    
    # Sort by MAE for better visualization
    sorted_indices = np.argsort(field_mae)[::-1]
    sorted_fields = [fields[i] for i in sorted_indices]
    sorted_mae = [field_mae[i] for i in sorted_indices]
    sorted_counts = [field_counts[i] for i in sorted_indices]
    
    bars = plt.bar(sorted_fields, sorted_mae, color='lightgreen')
    
    # Add count annotations
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'n={sorted_counts[i]}', ha='center', va='bottom', fontsize=9)
    
    plt.title(f"{model_type.capitalize()} Model - MAE by Research Field")
    plt.xlabel("Research Field")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(prediction_dir, 'error_by_field.png'))
    plt.close()
    
    # 4. Create prediction calibration plot
    plt.figure(figsize=(10, 10))
    
    # Flatten predictions and targets for all horizons
    flat_preds = val_predictions.flatten()
    flat_targets = val_targets.flatten()
    
    # Create scatter plot with density
    plt.hexbin(flat_targets, flat_preds, gridsize=50, cmap='viridis', bins='log')
    
    # Add perfect prediction line
    max_val = max(flat_preds.max(), flat_targets.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(flat_targets, flat_preds)
    x = np.linspace(0, max_val, 100)
    plt.plot(x, slope*x + intercept, 'g-', label=f'Regression Line (r²={r_value**2:.2f})')
    
    plt.colorbar(label='Count (log scale)')
    plt.title(f"{model_type.capitalize()} Model - Prediction Calibration")
    plt.xlabel("Actual Citations")
    plt.ylabel("Predicted Citations")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(prediction_dir, 'prediction_calibration.png'))
    plt.close()
    
    # 5. Temporal error analysis
    # Reshape data to analyze by horizons
    horizon_count = val_predictions.shape[1]
    horizon_labels = [f'H{h+1}' for h in range(horizon_count)]
    
    # Calculate MAE per horizon
    horizon_maes = []
    horizon_rmses = []
    
    for h in range(horizon_count):
        h_preds = val_predictions[:, h]
        h_targets = val_targets[:, h]
        h_mae = np.abs(h_preds - h_targets).mean()
        h_rmse = np.sqrt(((h_preds - h_targets) ** 2).mean())
        
        horizon_maes.append(float(h_mae))
        horizon_rmses.append(float(h_rmse))
    
    # Plot error by horizon
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(horizon_labels))
    width = 0.35
    
    plt.bar(x - width/2, horizon_maes, width, label='MAE', color='skyblue')
    plt.bar(x + width/2, horizon_rmses, width, label='RMSE', color='salmon')
    
    plt.title(f"{model_type.capitalize()} Model - Error by Time Horizon")
    plt.xlabel("Time Horizon")
    plt.ylabel("Error")
    plt.xticks(x, horizon_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(prediction_dir, 'error_by_horizon.png'))
    plt.close()
    
    # Save detailed metrics
    detailed_metrics = {
        'overall': metrics,
        'by_citation_range': bin_errors,
        'by_field': field_errors,
        'by_horizon': {
            'mae': horizon_maes,
            'rmse': horizon_rmses
        }
    }
    
    with open(os.path.join(prediction_dir, 'detailed_citation_metrics.json'), 'w') as f:
        json.dump(detailed_metrics, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else o)

    print(f"Citation prediction analysis complete. Results saved to {prediction_dir}")
    return metrics

def analyze_field_classification(model, data, model_results, output_dir, model_type):
    """Analyze field classification performance with detailed misclassification analysis"""
    print(f"Running field classification analysis for {model_type} model...")
    field_dir = os.path.join(output_dir, 'field')
    
    # Extract data
    field_targets = data['field_targets']
    val_state_ids = data['val_states']
    field_preds = model_results['field_preds']
    id_to_field = data['id_to_field']
    states = data['states']
    citation_targets = data['citation_targets']
    predictions = model_results['predictions']
    
    # Gather predictions and targets
    y_true = []
    y_pred = []
    citation_counts = []
    paper_ids = []
    state_features = []
    
    for state_id in val_state_ids:
        if state_id not in field_preds or state_id not in field_targets:
            continue
        
        y_true.append(field_targets[state_id])
        y_pred.append(field_preds[state_id])
        
        # Collect additional data for deeper analysis
        if state_id in states:
            state = states[state_id]
            citation_counts.append(state.citation_count)
            paper_ids.append(state.paper_id if hasattr(state, 'paper_id') else None)
            
            # Extract key features for analysis
            features = {
                'reference_diversity': state.reference_diversity if hasattr(state, 'reference_diversity') else 0,
                'field_impact_factor': state.field_impact_factor if hasattr(state, 'field_impact_factor') else 0,
                'collaboration_info': state.collaboration_info if hasattr(state, 'collaboration_info') else 0,
                'time_index': state.time_index if hasattr(state, 'time_index') else 0
            }
            state_features.append(features)
    
    # Convert to numpy arrays for easier analysis
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    citation_counts = np.array(citation_counts) if citation_counts else np.array([])
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[id_to_field.get(i, f"Field {i}") for i in range(len(id_to_field))],
                yticklabels=[id_to_field.get(i, f"Field {i}") for i in range(len(id_to_field))])
    plt.title(f"{model_type.capitalize()} Model - Field Classification Confusion Matrix")
    plt.xlabel("Predicted Field")
    plt.ylabel("True Field")
    plt.tight_layout()
    plt.savefig(os.path.join(field_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Calculate per-field accuracy
    total_per_field = np.sum(cm, axis=1)
    correct_per_field = np.diag(cm)
    accuracy_per_field = np.zeros_like(total_per_field, dtype=float)
    for i in range(len(total_per_field)):
        if total_per_field[i] > 0:
            accuracy_per_field[i] = correct_per_field[i] / total_per_field[i]
    
    # Get field names
    field_names = [id_to_field.get(i, f"Field {i}") for i in range(len(accuracy_per_field))]
    
    # Plot per-field accuracy
    plt.figure(figsize=(14, 8))
    
    # Sort fields by accuracy for better visualization
    sorted_indices = np.argsort(accuracy_per_field)[::-1]
    sorted_names = [field_names[i] for i in sorted_indices]
    sorted_accuracies = accuracy_per_field[sorted_indices]
    sorted_counts = total_per_field[sorted_indices]
    
    # Create bar plot
    bars = plt.bar(sorted_names, sorted_accuracies, color='skyblue')
    
    # Add count annotations
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={sorted_counts[i]}', ha='center', va='bottom', fontsize=9)
    
    plt.title(f"{model_type.capitalize()} Model - Per-Field Classification Accuracy")
    plt.xlabel("Field")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)  # Set y-axis from 0 to 1.1 to make room for count labels
    plt.tight_layout()
    plt.savefig(os.path.join(field_dir, 'field_accuracy.png'))
    plt.close()
    
    # 1. Misclassification patterns
    # Create a normalized confusion matrix to show misclassification patterns
    cm_norm = cm.astype(float)
    for i in range(cm_norm.shape[0]):
        row_sum = np.sum(cm_norm[i, :])
        if row_sum > 0:
            cm_norm[i, :] /= row_sum
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=field_names,
                yticklabels=field_names)
    plt.title(f"{model_type.capitalize()} Model - Normalized Confusion Matrix")
    plt.xlabel("Predicted Field")
    plt.ylabel("True Field")
    plt.tight_layout()
    plt.savefig(os.path.join(field_dir, 'normalized_confusion_matrix.png'))
    plt.close()
    
    # Identify top misclassification pairs
    misclass_pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                misclass_pairs.append({
                    'true_field': id_to_field.get(i, f"Field {i}"),
                    'pred_field': id_to_field.get(j, f"Field {j}"),
                    'count': int(cm[i, j]),
                    'rate': float(cm_norm[i, j])
                })
    
    # Sort by count and take top 10
    misclass_pairs.sort(key=lambda x: x['count'], reverse=True)
    top_misclass_pairs = misclass_pairs[:min(10, len(misclass_pairs))]
    
    # Save top misclassification pairs
    with open(os.path.join(field_dir, 'top_misclassifications.json'), 'w') as f:
        json.dump(top_misclass_pairs, f, indent=2)
    
    # 2. Relationship between field classification and citation accuracy
    if len(y_true) == len(citation_counts) and len(citation_counts) > 0:
        # Calculate citation prediction error for each paper
        citation_errors = []
        
        for i, state_id in enumerate(val_state_ids):
            if state_id not in predictions or state_id not in citation_targets:
                continue
                
            # Skip if we don't have field data
            if i >= len(y_true):
                continue
                
            # Calculate mean absolute error across horizons
            pred = predictions[state_id]
            target = citation_targets[state_id]
            mae = np.mean(np.abs(np.array(pred) - np.array(target)))
            citation_errors.append(mae)
        
        # Convert to numpy array
        citation_errors = np.array(citation_errors)
        
        # Group papers by correct/incorrect field prediction
        correct_mask = y_true == y_pred
        correct_errors = citation_errors[correct_mask]
        incorrect_errors = citation_errors[~correct_mask]
        
        # Calculate mean errors
        mean_correct_error = np.mean(correct_errors) if len(correct_errors) > 0 else 0
        mean_incorrect_error = np.mean(incorrect_errors) if len(incorrect_errors) > 0 else 0
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        labels = ['Correct Field Prediction', 'Incorrect Field Prediction']
        values = [mean_correct_error, mean_incorrect_error]
        counts = [sum(correct_mask), sum(~correct_mask)]
        
        bars = plt.bar(labels, values, color=['green', 'red'])
        
        # Add count annotations
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'n={counts[i]}', ha='center', va='bottom', fontsize=9)
        
        plt.title(f"{model_type.capitalize()} Model - Citation Error by Field Prediction Accuracy")
        plt.xlabel("Field Prediction Outcome")
        plt.ylabel("Mean Citation Prediction Error")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(field_dir, 'field_citation_error.png'))
        plt.close()
        
        # Also analyze by field
        field_citation_errors = {}
        
        for i in range(len(y_true)):
            if i >= len(citation_errors):
                continue
                
            field_id = y_true[i]
            field_name = id_to_field.get(field_id, f"Field {field_id}")
            
            if field_name not in field_citation_errors:
                field_citation_errors[field_name] = []
                
            field_citation_errors[field_name].append(citation_errors[i])
        
        # Calculate mean error by field
        field_mean_errors = {}
        for field, errors in field_citation_errors.items():
            if errors:
                field_mean_errors[field] = np.mean(errors)
        
        # Plot field-specific citation errors
        if field_mean_errors:
            plt.figure(figsize=(14, 8))
            
            # Sort fields by error
            sorted_fields = sorted(field_mean_errors.keys(), key=lambda f: field_mean_errors[f])
            sorted_errors = [field_mean_errors[f] for f in sorted_fields]
            
            # Create bar plot
            plt.bar(sorted_fields, sorted_errors, color='orange')
            
            plt.title(f"{model_type.capitalize()} Model - Citation Error by Field")
            plt.xlabel("Research Field")
            plt.ylabel("Mean Citation Prediction Error")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(field_dir, 'field_specific_citation_error.png'))
            plt.close()
    
    # 3. Field embeddings visualization (using citation patterns as a proxy)
    if len(y_true) > 0 and cm.shape[0] > 1:
        try:
            # Use the normalized confusion matrix as a distance matrix for embedding
            from sklearn.manifold import MDS
            
            # Convert confusion matrix to distance matrix (higher confusion = lower distance)
            # Add small epsilon to avoid division by zero
            distance_matrix = 1 - ((cm_norm + cm_norm.T) / 2 + 1e-10)
            
            # Apply Multidimensional Scaling to visualize field relationships
            mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
            field_positions = mds.fit_transform(distance_matrix)
            
            # Plot field embeddings
            plt.figure(figsize=(12, 10))
            
            # Scatter plot for field positions
            plt.scatter(field_positions[:, 0], field_positions[:, 1], s=100, alpha=0.7)
            
            # Add field labels
            for i, field in enumerate(field_names):
                plt.annotate(field, (field_positions[i, 0], field_positions[i, 1]), 
                           fontsize=9, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
            
            # Add arrows between commonly confused fields
            for pair in top_misclass_pairs[:5]:  # Show top 5 confusion pairs
                true_idx = field_names.index(pair['true_field'])
                pred_idx = field_names.index(pair['pred_field'])
                
                # Draw arrow from true to predicted field
                plt.arrow(field_positions[true_idx, 0], field_positions[true_idx, 1],
                        field_positions[pred_idx, 0] - field_positions[true_idx, 0],
                        field_positions[pred_idx, 1] - field_positions[true_idx, 1],
                        head_width=0.02, head_length=0.03, fc='red', ec='red', alpha=0.5)
            
            plt.title(f"{model_type.capitalize()} Model - Field Relationship Map")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(field_dir, 'field_embedding_map.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error in field embedding visualization: {str(e)}")
    
    # 4. Feature impact on field classification
    if state_features:
        try:
            # Extract features
            features = {}
            for feature_name in state_features[0].keys():
                features[feature_name] = []
                
            for i, feature_dict in enumerate(state_features):
                for feature_name, value in feature_dict.items():
                    features[feature_name].append(value)
            
            # Convert to numpy arrays
            for feature_name in features:
                features[feature_name] = np.array(features[feature_name])
            
            # Calculate correlation between features and classification accuracy
            correct_mask = y_true == y_pred
            feature_correlations = {}
            
            for feature_name, values in features.items():
                if len(values) != len(correct_mask):
                    continue
                    
                # Compute point-biserial correlation (continuous vs binary)
                correct_values = values[correct_mask]
                incorrect_values = values[~correct_mask]
                
                if len(correct_values) > 0 and len(incorrect_values) > 0:
                    mean_correct = np.mean(correct_values)
                    mean_incorrect = np.mean(incorrect_values)
                    
                    # Store mean difference as simple measure of impact
                    feature_correlations[feature_name] = {
                        'mean_correct': float(mean_correct),
                        'mean_incorrect': float(mean_incorrect),
                        'difference': float(mean_correct - mean_incorrect)
                    }
            
            # Plot feature impact
            if feature_correlations:
                plt.figure(figsize=(12, 6))
                
                feature_names = list(feature_correlations.keys())
                differences = [feature_correlations[f]['difference'] for f in feature_names]
                
                # Sort by absolute difference for better visualization
                sorted_indices = np.argsort(np.abs(differences))[::-1]
                sorted_names = [feature_names[i] for i in sorted_indices]
                sorted_diffs = [differences[i] for i in sorted_indices]
                
                bars = plt.bar(sorted_names, sorted_diffs, 
                              color=['green' if d > 0 else 'red' for d in sorted_diffs])
                
                plt.title(f"{model_type.capitalize()} Model - Feature Impact on Field Classification")
                plt.xlabel("Feature")
                plt.ylabel("Mean Difference (Correct - Incorrect)")
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(field_dir, 'feature_impact.png'))
                plt.close()
                
                # Save feature correlations
                with open(os.path.join(field_dir, 'feature_impact.json'), 'w') as f:
                    json.dump(feature_correlations, f, indent=2)
                
        except Exception as e:
            print(f"Error in feature impact analysis: {str(e)}")
    
    # Save field classification metrics
    field_metrics = {
        'overall_accuracy': float(np.mean(y_true == y_pred)),
        'per_field_accuracy': {field_names[i]: float(accuracy_per_field[i]) for i in range(len(field_names))},
        'top_misclassifications': top_misclass_pairs
    }
    
    with open(os.path.join(field_dir, 'field_metrics.json'), 'w') as f:
        json.dump(field_metrics, f, indent=2)
    
    print(f"Field classification analysis complete. Results saved to {field_dir}")
    return field_metrics

def analyze_rl_strategy(model, data, model_results, output_dir, model_type):
    """Analyze reinforcement learning strategies with detailed component analysis"""
    print(f"Running RL strategy analysis for {model_type} model...")
    strategy_dir = os.path.join(output_dir, 'strategy')
    
    # Extract action data
    actions = model_results['actions']
    val_state_ids = data['val_states']
    states = data['states']
    citation_targets = data['citation_targets']
    predictions = model_results['predictions']
    field_targets = data.get('field_targets', {})
    id_to_field = data.get('id_to_field', {})
    
    # Collect action components
    field_positions = []
    novelty_levels = []
    collaboration_strategies = []
    citation_choices = []
    combined_focus = []
    timing_values = []
    
    # Also collect paper features and citation data for correlation analysis
    citation_counts = []
    predicted_citations = []
    actual_citations = []
    paper_fields = []
    reference_diversity = []
    paper_ids = []
    state_ids = []
    
    for state_id in val_state_ids:
        if state_id not in actions:
            continue
        
        state_ids.append(state_id)
        action = actions[state_id]
        state = states[state_id]
        
        # Extract action components
        if isinstance(action, dict):
            field_positions.append(action.get("field_positioning"))
            novelty_levels.append(action.get("novelty_level"))
            collaboration_strategies.append(action.get("collaboration_strategy"))
            citation_choices.append(action.get("citation_choices"))
            combined_focus.append(action.get("combined_focus"))
            timing_values.append(action.get("timing"))
        else:
            field_positions.append(action.field_positioning)
            novelty_levels.append(action.novelty_level)
            collaboration_strategies.append(action.collaboration_strategy)
            citation_choices.append(action.citation_choices)
            combined_focus.append(action.combined_focus)
            timing_values.append(action.timing)

        
        # Collect paper data
        citation_counts.append(state.citation_count)
        if state_id in predictions:
            predicted_citations.append(np.mean(predictions[state_id]))
        else:
            predicted_citations.append(0)
            
        if state_id in citation_targets:
            actual_citations.append(np.mean(citation_targets[state_id]))
        else:
            actual_citations.append(0)
            
        # Get field if available
        if state_id in field_targets:
            field_id = field_targets[state_id]
            field_name = id_to_field.get(field_id, f"Field {field_id}")
            paper_fields.append(field_name)
        else:
            paper_fields.append("Unknown")
            
        # Get reference diversity
        if hasattr(state, 'reference_diversity'):
            reference_diversity.append(state.reference_diversity)
        else:
            reference_diversity.append(0)
            
        # Get paper ID
        if hasattr(state, 'paper_id'):
            paper_ids.append(state.paper_id)
        else:
            paper_ids.append(None)
    
    # Convert to numpy arrays
    novelty_levels = np.array(novelty_levels).flatten()
    collaboration_strategies = np.array(collaboration_strategies)
    timing_values = np.array(timing_values)
    citation_counts = np.array(citation_counts)
    predicted_citations = np.array(predicted_citations)
    actual_citations = np.array(actual_citations).flatten()
    reference_diversity = np.array(reference_diversity)
    
    # 1. Analyze distributions of scalar actions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(novelty_levels, bins=20, kde=True)
    plt.title('Novelty Level Distribution')
    plt.xlabel('Novelty Level')
    
    plt.subplot(1, 3, 2)
    sns.histplot(collaboration_strategies, bins=20, kde=True)
    plt.title('Collaboration Strategy Distribution')
    plt.xlabel('Collaboration Strategy')
    
    plt.subplot(1, 3, 3)
    sns.histplot(timing_values, bins=20, kde=True)
    plt.title('Timing Distribution')
    plt.xlabel('Timing Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(strategy_dir, 'action_distributions.png'))
    plt.close()
    
    # 2. Analyze correlation between actions and citation outcomes
    plt.figure(figsize=(15, 10))
    
    # Actual citations vs scalar actions
    plt.subplot(2, 3, 1)
    plt.scatter(novelty_levels, actual_citations, alpha=0.5, color='blue')
    # Add trend line
    if len(novelty_levels) > 1:
        z = np.polyfit(novelty_levels, actual_citations, 1)
        p = np.poly1d(z)
        plt.plot(sorted(novelty_levels), p(sorted(novelty_levels)), "r--")
    plt.title('Novelty Level vs Actual Citations')
    plt.xlabel('Novelty Level')
    plt.ylabel('Actual Citations')
    
    plt.subplot(2, 3, 2)
    plt.scatter(collaboration_strategies, actual_citations, alpha=0.5, color='blue')
    # Add trend line
    if len(collaboration_strategies) > 1:
        z = np.polyfit(collaboration_strategies, actual_citations, 1)
        p = np.poly1d(z)
        plt.plot(sorted(collaboration_strategies), p(sorted(collaboration_strategies)), "r--")
    plt.title('Collaboration Strategy vs Actual Citations')
    plt.xlabel('Collaboration Strategy')
    plt.ylabel('Actual Citations')
    
    plt.subplot(2, 3, 3)
    plt.scatter(timing_values, actual_citations, alpha=0.5, color='blue')
    # Add trend line
    if len(timing_values) > 1:
        z = np.polyfit(timing_values, actual_citations, 1)
        p = np.poly1d(z)
        plt.plot(sorted(timing_values), p(sorted(timing_values)), "r--")
    plt.title('Timing vs Actual Citations')
    plt.xlabel('Timing Value')
    plt.ylabel('Actual Citations')
    
    # Predicted citations vs scalar actions
    plt.subplot(2, 3, 4)
    plt.scatter(novelty_levels, predicted_citations, alpha=0.5, color='green')
    # Add trend line
    if len(novelty_levels) > 1:
        z = np.polyfit(novelty_levels, predicted_citations, 1)
        p = np.poly1d(z)
        plt.plot(sorted(novelty_levels), p(sorted(novelty_levels)), "r--")
    plt.title('Novelty Level vs Predicted Citations')
    plt.xlabel('Novelty Level')
    plt.ylabel('Predicted Citations')
    
    plt.subplot(2, 3, 5)
    plt.scatter(collaboration_strategies, predicted_citations, alpha=0.5, color='green')
    # Add trend line
    if len(collaboration_strategies) > 1:
        z = np.polyfit(collaboration_strategies, predicted_citations, 1)
        p = np.poly1d(z)
        plt.plot(sorted(collaboration_strategies), p(sorted(collaboration_strategies)), "r--")
    plt.title('Collaboration Strategy vs Predicted Citations')
    plt.xlabel('Collaboration Strategy')
    plt.ylabel('Predicted Citations')
    
    plt.subplot(2, 3, 6)
    plt.scatter(timing_values, predicted_citations, alpha=0.5, color='green')
    # Add trend line
    if len(timing_values) > 1:
        z = np.polyfit(timing_values, predicted_citations, 1)
        p = np.poly1d(z)
        plt.plot(sorted(timing_values), p(sorted(timing_values)), "r--")
    plt.title('Timing vs Predicted Citations')
    plt.xlabel('Timing Value')
    plt.ylabel('Predicted Citations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(strategy_dir, 'action_citation_correlation.png'))
    plt.close()
    
    # 3. Field positioning visualization
    # Analyze field positioning vectors if they're 2D or can be reduced to 2D
    if field_positions and len(field_positions) > 0:
        try:
            # Check if field positions are already 2D
            if len(field_positions[0]) == 2:
                field_positions_2d = np.array(field_positions)
            else:
                # If not, use PCA to reduce to 2D
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                field_positions_2d = pca.fit_transform(field_positions)
            
            # Plot field positions colored by citation count
            plt.figure(figsize=(12, 10))
            
            # Use citation count for color
            sc = plt.scatter(field_positions_2d[:, 0], field_positions_2d[:, 1], 
                           c=citation_counts, cmap='viridis', alpha=0.7)
            
            plt.colorbar(sc, label='Citation Count')
            plt.title(f"{model_type.capitalize()} Model - Field Positioning")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(strategy_dir, 'field_positioning.png'))
            plt.close()
            
            # Also plot field positions colored by field if available
            if len(set(paper_fields)) > 1:
                # Get unique fields
                unique_fields = sorted(set(paper_fields))
                field_to_idx = {field: i for i, field in enumerate(unique_fields)}
                field_indices = [field_to_idx[field] for field in paper_fields]
                
                # Create discrete colormap
                import matplotlib.colors as mcolors
                cmap = plt.cm.get_cmap('tab20', len(unique_fields))
                
                plt.figure(figsize=(12, 10))
                
                # Use field for color
                sc = plt.scatter(field_positions_2d[:, 0], field_positions_2d[:, 1], 
                               c=field_indices, cmap=cmap, alpha=0.7)
                
                # Add legend
                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=cmap(field_to_idx[field]), 
                                        markersize=10, label=field)
                                  for field in unique_fields[:10]]  # Limit to top 10 fields for readability
                
                if len(unique_fields) > 10:
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor='gray', markersize=10, 
                                               label='Other Fields'))
                
                plt.legend(handles=legend_elements, title="Research Field", 
                         loc="upper right", bbox_to_anchor=(1.15, 1))
                
                plt.title(f"{model_type.capitalize()} Model - Field Positioning by Research Field")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(strategy_dir, 'field_positioning_by_field.png'))
                plt.close()
            
        except Exception as e:
            print(f"Error in field positioning visualization: {str(e)}")
    
    # 4. Citation choices analysis
    if citation_choices and len(citation_choices) > 0:
        try:
            # Check if citation choices are vectors or matrices
            citation_choice_dims = []
            for choice in citation_choices:
                if hasattr(choice, 'shape'):
                    citation_choice_dims.append(choice.shape)
                elif isinstance(choice, list):
                    citation_choice_dims.append(len(choice))
                else:
                    citation_choice_dims.append(0)
            
            # For 1D citation choices, create heatmap of choices vs citation counts
            if all(isinstance(dim, int) for dim in citation_choice_dims):
                # Create bins for citation counts
                citation_bins = [0, 5, 10, 20, 50, 100, float('inf')]
                bin_labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '101+']
                
                # Assign each paper to a bin
                bin_indices = np.digitize(citation_counts, citation_bins) - 1
                bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)
                
                # Process citation choices
                citation_choice_avg_by_bin = []
                for bin_idx in range(len(bin_labels)):
                    bin_mask = (bin_indices == bin_idx)
                    if np.any(bin_mask):
                        # Compute average citation choice for this bin
                        bin_choices = [citation_choices[i] for i in range(len(citation_choices)) if bin_mask[i]]
                        if bin_choices:
                            # Handle different data types
                            if all(isinstance(choice, (list, np.ndarray)) for choice in bin_choices):
                                avg_choice = np.mean([np.mean(choice) for choice in bin_choices])
                            else:
                                avg_choice = np.mean(bin_choices)
                            citation_choice_avg_by_bin.append(avg_choice)
                    else:
                        citation_choice_avg_by_bin.append(0)
                
                # Plot citation choices by citation bin
                plt.figure(figsize=(12, 6))
                
                plt.bar(bin_labels, citation_choice_avg_by_bin)
                plt.title(f"{model_type.capitalize()} Model - Citation Choice by Citation Count")
                plt.xlabel("Citation Count Range")
                plt.ylabel("Average Citation Choice")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(strategy_dir, 'citation_choice_by_citation_count.png'))
                plt.close()
            
            # For vector citation choices, use dimensionality reduction and visualize
            elif all(hasattr(choice, 'shape') and len(choice.shape) > 0 for choice in citation_choices):
                # Convert to numpy arrays
                citation_choices_np = []
                for choice in citation_choices:
                    if isinstance(choice, np.ndarray):
                        citation_choices_np.append(choice.flatten())
                    elif isinstance(choice, list):
                        citation_choices_np.append(np.array(choice).flatten())
                
                if citation_choices_np:
                    # Stack arrays (pad with zeros if necessary)
                    max_length = max(arr.shape[0] for arr in citation_choices_np)
                    citation_choices_padded = []
                    for arr in citation_choices_np:
                        if arr.shape[0] < max_length:
                            padded = np.zeros(max_length)
                            padded[:arr.shape[0]] = arr
                            citation_choices_padded.append(padded)
                        else:
                            citation_choices_padded.append(arr)
                    
                    citation_choices_matrix = np.vstack(citation_choices_padded)
                    
                    # Use PCA to reduce dimensions for visualization
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    citation_choices_2d = pca.fit_transform(citation_choices_matrix)
                    
                    # Plot citation choices colored by citation count
                    plt.figure(figsize=(12, 10))
                    
                    # Use citation count for color
                    sc = plt.scatter(citation_choices_2d[:, 0], citation_choices_2d[:, 1], 
                                   c=citation_counts, cmap='viridis', alpha=0.7)
                    
                    plt.colorbar(sc, label='Citation Count')
                    plt.title(f"{model_type.capitalize()} Model - Citation Choices")
                    plt.xlabel("PCA Component 1")
                    plt.ylabel("PCA Component 2")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.savefig(os.path.join(strategy_dir, 'citation_choices_pca.png'))
                    plt.close()
                    
                    # Also analyze by field if available
                    if len(set(paper_fields)) > 1:
                        # Get unique fields
                        unique_fields = sorted(set(paper_fields))
                        field_to_idx = {field: i for i, field in enumerate(unique_fields)}
                        field_indices = [field_to_idx[field] for field in paper_fields]
                        
                        # Create discrete colormap
                        import matplotlib.colors as mcolors
                        cmap = plt.cm.get_cmap('tab20', len(unique_fields))
                        
                        plt.figure(figsize=(12, 10))
                        
                        # Use field for color
                        sc = plt.scatter(citation_choices_2d[:, 0], citation_choices_2d[:, 1], 
                                       c=field_indices, cmap=cmap, alpha=0.7)
                        
                        # Add legend
                        from matplotlib.lines import Line2D
                        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=cmap(field_to_idx[field]), 
                                                markersize=10, label=field)
                                          for field in unique_fields[:10]]  # Limit to top 10 fields
                        
                        plt.legend(handles=legend_elements, title="Research Field", 
                                 loc="upper right", bbox_to_anchor=(1.15, 1))
                        
                        plt.title(f"{model_type.capitalize()} Model - Citation Choices by Field")
                        plt.xlabel("PCA Component 1")
                        plt.ylabel("PCA Component 2")
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        plt.savefig(os.path.join(strategy_dir, 'citation_choices_by_field.png'))
                        plt.close()
            
        except Exception as e:
            print(f"Error in citation choices analysis: {str(e)}")
    
    # 5. Combined focus analysis
    if combined_focus and len(combined_focus) > 0:
        try:
            # Check if combined focus values are vectors
            focus_dims = []
            for focus in combined_focus:
                if hasattr(focus, 'shape'):
                    focus_dims.append(focus.shape)
                elif isinstance(focus, list):
                    focus_dims.append(len(focus))
                else:
                    focus_dims.append(0)
            
            # For vector combined focus, use PCA for visualization
            if all(isinstance(dim, tuple) and len(dim) > 0 for dim in focus_dims) or all(isinstance(dim, int) and dim > 0 for dim in focus_dims):
                # Convert to numpy arrays
                focus_np = []
                for focus in combined_focus:
                    if isinstance(focus, np.ndarray):
                        focus_np.append(focus.flatten())
                    elif isinstance(focus, list):
                        focus_np.append(np.array(focus).flatten())
                
                if focus_np:
                    # Stack arrays (pad with zeros if necessary)
                    max_length = max(arr.shape[0] for arr in focus_np)
                    focus_padded = []
                    for arr in focus_np:
                        if arr.shape[0] < max_length:
                            padded = np.zeros(max_length)
                            padded[:arr.shape[0]] = arr
                            focus_padded.append(padded)
                        else:
                            focus_padded.append(arr)
                    
                    focus_matrix = np.vstack(focus_padded)
                    
                    # Use PCA to reduce dimensions for visualization
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    focus_2d = pca.fit_transform(focus_matrix)
                    
                    # Plot combined focus colored by citation count
                    plt.figure(figsize=(12, 10))
                    
                    # Use citation count for color
                    sc = plt.scatter(focus_2d[:, 0], focus_2d[:, 1], 
                                   c=citation_counts, cmap='viridis', alpha=0.7)
                    
                    plt.colorbar(sc, label='Citation Count')
                    plt.title(f"{model_type.capitalize()} Model - Combined Focus")
                    plt.xlabel("PCA Component 1")
                    plt.ylabel("PCA Component 2")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.savefig(os.path.join(strategy_dir, 'combined_focus_pca.png'))
                    plt.close()
                    
                    # Calculate correlation between focus components and citations
                    corr_with_citations = []
                    component_names = []
                    
                    for i in range(min(5, pca.components_.shape[0])):  # Analyze up to 5 principal components
                        component = focus_2d[:, i]
                        corr = np.corrcoef(component, actual_citations)[0, 1]
                        corr_with_citations.append(corr)
                        component_names.append(f"PC{i+1}")
                    
                    # Plot correlations
                    plt.figure(figsize=(10, 6))
                    
                    plt.bar(component_names, corr_with_citations)
                    plt.title(f"{model_type.capitalize()} Model - Focus Component Correlations with Citations")
                    plt.xlabel("Principal Component")
                    plt.ylabel("Correlation with Citations")
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.savefig(os.path.join(strategy_dir, 'focus_correlation_with_citations.png'))
                    plt.close()
        
        except Exception as e:
            print(f"Error in combined focus analysis: {str(e)}")
    
    # 6. Learned policy patterns
    # Analyze how strategy varies with paper characteristics
    
    # Correlation matrix between actions and paper features
    try:
        # Create feature matrix
        features = [
            ('citation_count', citation_counts),
            ('reference_diversity', reference_diversity)
        ]
        
        # Add other features if available
        if hasattr(states[next(iter(states))], 'field_impact_factor'):
            field_impact = [states[sid].field_impact_factor for sid in state_ids]
            features.append(('field_impact', np.array(field_impact)))
        
        # Create action matrix
        actions = [
            ('novelty', novelty_levels),
            ('collaboration', collaboration_strategies),
            ('timing', timing_values)
        ]
        
        # Add outcome measurements
        outcomes = [
            ('predicted_citations', predicted_citations),
            ('actual_citations', actual_citations)
        ]
        
        # Combine all variables for correlation matrix
        corr_vars = features + actions + outcomes
        corr_names = [name for name, _ in corr_vars]
        corr_data = np.column_stack([data for _, data in corr_vars])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(corr_data.T)
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                  xticklabels=corr_names, yticklabels=corr_names)
        
        plt.title(f"{model_type.capitalize()} Model - Strategy Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(strategy_dir, 'strategy_correlation_matrix.png'))
        plt.close()
        
        # Extract specific correlations for strategic insights
        strategy_insights = {}
        
        # Find action correlations with features
        for i, (action_name, _) in enumerate(actions):
            action_idx = corr_names.index(action_name)
            insights = {}
            
            # Correlations with paper features
            for feature_name, _ in features:
                feature_idx = corr_names.index(feature_name)
                insights[f'corr_with_{feature_name}'] = float(corr_matrix[action_idx, feature_idx])
            
            # Correlations with outcomes
            for outcome_name, _ in outcomes:
                outcome_idx = corr_names.index(outcome_name)
                insights[f'corr_with_{outcome_name}'] = float(corr_matrix[action_idx, outcome_idx])
            
            strategy_insights[action_name] = insights
        
        # Save strategic insights
        with open(os.path.join(strategy_dir, 'strategy_insights.json'), 'w') as f:
            json.dump(strategy_insights, f, indent=2)
            
        # Plot feature-action correlations
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        feature_names = [name for name, _ in features]
        action_names = [name for name, _ in actions]
        
        # Create a matrix of feature-action correlations
        corr_grid = np.zeros((len(feature_names), len(action_names)))
        
        for i, feature_name in enumerate(feature_names):
            feature_idx = corr_names.index(feature_name)
            for j, action_name in enumerate(action_names):
                action_idx = corr_names.index(action_name)
                corr_grid[i, j] = corr_matrix[feature_idx, action_idx]
        
        # Plot heatmap
        sns.heatmap(corr_grid, annot=True, fmt='.2f', cmap='coolwarm',
                  xticklabels=action_names, yticklabels=feature_names)
        
        plt.title(f"{model_type.capitalize()} Model - Feature-Action Correlations")
        plt.xlabel("Action Component")
        plt.ylabel("Paper Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(strategy_dir, 'feature_action_correlations.png'))
        plt.close()
        
        # Plot action-outcome correlations
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        outcome_names = [name for name, _ in outcomes]
        action_outcome_corr = np.zeros((len(action_names), len(outcome_names)))
        
        for i, action_name in enumerate(action_names):
            action_idx = corr_names.index(action_name)
            for j, outcome_name in enumerate(outcome_names):
                outcome_idx = corr_names.index(outcome_name)
                action_outcome_corr[i, j] = corr_matrix[action_idx, outcome_idx]
        
        # Plot heatmap
        sns.heatmap(action_outcome_corr, annot=True, fmt='.2f', cmap='coolwarm',
                  xticklabels=outcome_names, yticklabels=action_names)
        
        plt.title(f"{model_type.capitalize()} Model - Action-Outcome Correlations")
        plt.xlabel("Outcome")
        plt.ylabel("Action Component")
        plt.tight_layout()
        plt.savefig(os.path.join(strategy_dir, 'action_outcome_correlations.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error in policy pattern analysis: {str(e)}")
    
    # 7. High-impact vs. low-impact paper strategies
    try:
        # Divide papers into high and low impact based on actual citations
        median_citations = np.median(actual_citations)
        high_impact_mask = actual_citations > median_citations
        low_impact_mask = ~high_impact_mask
        
        high_impact_count = np.sum(high_impact_mask)
        low_impact_count = np.sum(low_impact_mask)
        
        if high_impact_count > 0 and low_impact_count > 0:
            # Calculate mean strategy values for high and low impact papers
            strategy_comparison = {
                'high_impact': {
                    'count': int(high_impact_count),
                    'mean_citations': float(np.mean(actual_citations[high_impact_mask])),
                    'novelty': float(np.mean(novelty_levels[high_impact_mask])),
                    'collaboration': float(np.mean(collaboration_strategies[high_impact_mask])),
                    'timing': float(np.mean(timing_values[high_impact_mask]))
                },
                'low_impact': {
                    'count': int(low_impact_count),
                    'mean_citations': float(np.mean(actual_citations[low_impact_mask])),
                    'novelty': float(np.mean(novelty_levels[low_impact_mask])),
                    'collaboration': float(np.mean(collaboration_strategies[low_impact_mask])),
                    'timing': float(np.mean(timing_values[low_impact_mask]))
                }
            }
            
            # Calculate percent differences for easier interpretation
            percent_diff = {}
            for strategy in ['novelty', 'collaboration', 'timing']:
                high_val = strategy_comparison['high_impact'][strategy]
                low_val = strategy_comparison['low_impact'][strategy]
                base_val = (high_val + low_val) / 2
                
                if base_val != 0:
                    percent_diff[strategy] = 100 * (high_val - low_val) / base_val
                else:
                    percent_diff[strategy] = 0
            
            strategy_comparison['percent_difference'] = percent_diff
            
            # Save strategy comparison
            with open(os.path.join(strategy_dir, 'impact_strategy_comparison.json'), 'w') as f:
                json.dump(strategy_comparison, f, indent=2)
            
            # Plot strategy comparison
            plt.figure(figsize=(12, 6))
            
            # Prepare data
            strategies = ['novelty', 'collaboration', 'timing']
            high_values = [strategy_comparison['high_impact'][s] for s in strategies]
            low_values = [strategy_comparison['low_impact'][s] for s in strategies]
            
            x = np.arange(len(strategies))
            width = 0.35
            
            plt.bar(x - width/2, high_values, width, label=f'High Impact (>{median_citations:.1f} citations)')
            plt.bar(x + width/2, low_values, width, label=f'Low Impact (≤{median_citations:.1f} citations)')
            
            plt.title(f"{model_type.capitalize()} Model - Strategies by Impact Level")
            plt.xlabel("Strategy Component")
            plt.ylabel("Mean Value")
            plt.xticks(x, [s.capitalize() for s in strategies])
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(strategy_dir, 'impact_strategy_comparison.png'))
            plt.close()
            
            # Plot percent differences
            plt.figure(figsize=(10, 6))
            
            plt.bar(strategies, [percent_diff[s] for s in strategies], color=['green' if v > 0 else 'red' for v in percent_diff.values()])
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.title(f"{model_type.capitalize()} Model - Strategy Differences (High vs Low Impact)")
            plt.xlabel("Strategy Component")
            plt.ylabel("Percent Difference (%)")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(strategy_dir, 'impact_strategy_percent_diff.png'))
            plt.close()
        
    except Exception as e:
        print(f"Error in impact strategy analysis: {str(e)}")
    
    # Save strategy statistics
    strategy_metrics = {
        'mean_values': {
            'novelty': float(np.mean(novelty_levels)),
            'collaboration': float(np.mean(collaboration_strategies)),
            'timing': float(np.mean(timing_values))
        },
        'correlations': {
            'novelty_citations': float(np.corrcoef(novelty_levels, actual_citations)[0, 1]),
            'collaboration_citations': float(np.corrcoef(collaboration_strategies, actual_citations)[0, 1]),
            'timing_citations': float(np.corrcoef(timing_values, actual_citations)[0, 1])
        }
    }
    
    with open(os.path.join(strategy_dir, 'strategy_metrics.json'), 'w') as f:
        json.dump(strategy_metrics, f, indent=2)
    
    print(f"RL strategy analysis complete. Results saved to {strategy_dir}")
    return strategy_metrics

def analyze_temporal_patterns(model, data, model_results, output_dir, model_type):
    """Analyze temporal prediction patterns with detailed insights on growth and bias"""
    print(f"Running temporal pattern analysis for {model_type} model...")
    temporal_dir = os.path.join(output_dir, 'temporal')
    
    # Extract data
    predictions = model_results['predictions']
    citation_targets = data['citation_targets']
    val_state_ids = data['val_states']
    states = data['states']
    horizons = len(next(iter(predictions.values())))  # Get number of horizons from first prediction
    
    # Organize prediction errors by horizon
    horizon_errors = [[] for _ in range(horizons)]
    
    for state_id in val_state_ids:
        if state_id not in predictions:
            continue
        
        pred = predictions[state_id]
        target = citation_targets[state_id]
        
        for h in range(horizons):
            error = abs(pred[h] - target[h])
            horizon_errors[h].append(error)
    
    # Calculate mean and std of errors by horizon
    horizon_mean_errors = [np.mean(errors) for errors in horizon_errors]
    horizon_std_errors = [np.std(errors) for errors in horizon_errors]
    
    # Plot mean error by horizon
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(1, horizons+1), horizon_mean_errors, yerr=horizon_std_errors, fmt='o-')
    plt.title(f"{model_type.capitalize()} Model - Prediction Error by Time Horizon")
    plt.xlabel("Time Horizon")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(temporal_dir, 'horizon_error.png'))
    plt.close()
    
    # Analyze citation growth patterns
    # Gather average predicted and actual citation growth
    growth_pred = []
    growth_actual = []
    
    for state_id in val_state_ids:
        if state_id not in predictions:
            continue
        
        pred = predictions[state_id]
        target = citation_targets[state_id]
        
        # Calculate growth rates
        pred_growth = [(pred[h+1] / max(1, pred[h])) - 1 for h in range(horizons-1)]
        actual_growth = [(target[h+1] / max(1, target[h])) - 1 for h in range(horizons-1)]
        
        growth_pred.append(pred_growth)
        growth_actual.append(actual_growth)
    
    # Convert to numpy arrays
    growth_pred = np.array(growth_pred)
    growth_actual = np.array(growth_actual)
    
    # Calculate mean growth rates
    mean_pred_growth = np.mean(growth_pred, axis=0)
    mean_actual_growth = np.mean(growth_actual, axis=0)
    
    # Plot growth rates
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, horizons), mean_pred_growth, 'o-', label='Predicted Growth')
    plt.plot(range(1, horizons), mean_actual_growth, 'o-', label='Actual Growth')
    plt.title(f"{model_type.capitalize()} Model - Citation Growth Rates")
    plt.xlabel("Time Horizon Transition")
    plt.ylabel("Mean Growth Rate")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(temporal_dir, 'growth_rates.png'))
    plt.close()
    
    # 1. Temporal prediction bias analysis
    # Calculate bias (predicted - actual) for each horizon
    biases = [[] for _ in range(horizons)]
    relative_biases = [[] for _ in range(horizons)]
    
    for state_id in val_state_ids:
        if state_id not in predictions:
            continue
        
        pred = predictions[state_id]
        target = citation_targets[state_id]
        
        for h in range(horizons):
            # Absolute bias
            bias = pred[h] - target[h]
            biases[h].append(bias)
            
            # Relative bias (as percentage of actual)
            rel_bias = bias / max(1, target[h])  # Prevent division by zero
            relative_biases[h].append(rel_bias)
    
    # Calculate mean bias for each horizon
    mean_biases = [np.mean(horizon_biases) for horizon_biases in biases]
    mean_relative_biases = [np.mean(horizon_rel_biases) for horizon_rel_biases in relative_biases]
    
    # Plot absolute bias by horizon
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, horizons+1), mean_biases)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f"{model_type.capitalize()} Model - Prediction Bias by Time Horizon")
    plt.xlabel("Time Horizon")
    plt.ylabel("Mean Bias (Predicted - Actual)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(temporal_dir, 'horizon_bias.png'))
    plt.close()
    
    # Plot relative bias by horizon
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, horizons+1), [b * 100 for b in mean_relative_biases])  # Convert to percentage
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f"{model_type.capitalize()} Model - Relative Prediction Bias by Time Horizon")
    plt.xlabel("Time Horizon")
    plt.ylabel("Mean Relative Bias (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(temporal_dir, 'horizon_relative_bias.png'))
    plt.close()
    
    # 2. Citation acceleration patterns
    # We'll analyze how citation acceleration (change in growth rate) varies over time
    acceleration_pred = []
    acceleration_actual = []
    
    for state_id in val_state_ids:
        if state_id not in predictions or horizons < 3:  # Need at least 3 horizons for acceleration
            continue
        
        pred = predictions[state_id]
        target = citation_targets[state_id]
        
        # Calculate growth rates first
        pred_growth_rates = [(pred[h+1] / max(1, pred[h])) - 1 for h in range(horizons-1)]
        actual_growth_rates = [(target[h+1] / max(1, target[h])) - 1 for h in range(horizons-1)]
        
        # Calculate acceleration (change in growth rate)
        pred_accel = [pred_growth_rates[h+1] - pred_growth_rates[h] for h in range(len(pred_growth_rates)-1)]
        actual_accel = [actual_growth_rates[h+1] - actual_growth_rates[h] for h in range(len(actual_growth_rates)-1)]
        
        acceleration_pred.append(pred_accel)
        acceleration_actual.append(actual_accel)
    
    # Convert to numpy arrays
    if acceleration_pred:
        acceleration_pred = np.array(acceleration_pred)
        acceleration_actual = np.array(acceleration_actual)
        
        # Calculate mean acceleration rates
        mean_pred_accel = np.mean(acceleration_pred, axis=0)
        mean_actual_accel = np.mean(acceleration_actual, axis=0)
        
        # Plot acceleration rates
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, horizons-1), mean_pred_accel, 'o-', label='Predicted Acceleration')
        plt.plot(range(1, horizons-1), mean_actual_accel, 'o-', label='Actual Acceleration')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title(f"{model_type.capitalize()} Model - Citation Acceleration Patterns")
        plt.xlabel("Time Horizon Transition")
        plt.ylabel("Mean Acceleration (Δ Growth Rate)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(temporal_dir, 'acceleration_patterns.png'))
        plt.close()
    
    # 3. Time since publication effect
    # Analyze how paper age affects prediction accuracy
    time_indices = []
    horizon_errors_by_time = [[] for _ in range(horizons)]
    
    for state_id in val_state_ids:
        if state_id not in predictions:
            continue
        
        state = states[state_id]
        time_index = state.time_index  # Time since publication
        pred = predictions[state_id]
        target = citation_targets[state_id]
        
        time_indices.append(time_index)
        for h in range(horizons):
            error = abs(pred[h] - target[h])
            horizon_errors_by_time[h].append((time_index, error))
    
    # Create scatter plot of error vs time for each horizon
    plt.figure(figsize=(15, 10))
    
    for h in range(horizons):
        plt.subplot(2, 2, h+1)
        
        # Extract data for this horizon
        time_data = [t for t, _ in horizon_errors_by_time[h]]
        error_data = [e for _, e in horizon_errors_by_time[h]]
        
        # Create scatter plot
        plt.scatter(time_data, error_data, alpha=0.5)
        
        # Add trend line
        if time_data:
            from scipy.stats import linregress
            slope, intercept, r_value, _, _ = linregress(time_data, error_data)
            x_trend = np.array([min(time_data), max(time_data)])
            y_trend = slope * x_trend + intercept
            plt.plot(x_trend, y_trend, 'r-', label=f'Trend (r={r_value:.2f})')
        
        plt.title(f"Horizon {h+1}: Error vs Time Since Publication")
        plt.xlabel("Time Index (Quarters Since Publication)")
        plt.ylabel("Prediction Error")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(temporal_dir, 'error_vs_time.png'))
    plt.close()
    
    # 4. Long-term vs short-term prediction accuracy
    # Group papers by initial citation count and analyze long vs short-term accuracy
    citation_bins = [(0, 5), (6, 10), (11, 20), (21, 50), (51, float('inf'))]
    bin_labels = ['0-5', '6-10', '11-20', '21-50', '51+']
    
    # Initialize containers for relative errors by bin and horizon
    rel_errors_by_bin = {label: [[] for _ in range(horizons)] for label in bin_labels}
    
    for state_id in val_state_ids:
        if state_id not in predictions:
            continue
        
        state = states[state_id]
        citation_count = state.citation_count
        pred = predictions[state_id]
        target = citation_targets[state_id]
        
        # Determine which bin this paper belongs to
        bin_idx = next((i for i, (lower, upper) in enumerate(citation_bins) 
                        if lower <= citation_count <= upper), None)
        
        if bin_idx is not None:
            bin_label = bin_labels[bin_idx]
            
            # Calculate relative error for each horizon
            for h in range(horizons):
                rel_error = abs(pred[h] - target[h]) / max(1, target[h])
                rel_errors_by_bin[bin_label][h].append(rel_error)
    
    # Calculate mean relative error for each bin and horizon
    mean_rel_errors = {}
    for bin_label in bin_labels:
        mean_rel_errors[bin_label] = []
        for h in range(horizons):
            errors = rel_errors_by_bin[bin_label][h]
            if errors:
                mean_rel_errors[bin_label].append(np.mean(errors))
            else:
                mean_rel_errors[bin_label].append(None)
    
    # Plot mean relative error by citation bin for each horizon
    plt.figure(figsize=(12, 8))
    
    x = range(1, horizons+1)
    for bin_label in bin_labels:
        errors = mean_rel_errors[bin_label]
        if any(e is not None for e in errors):
            # Filter out None values
            valid_x = [x_val for x_val, e in zip(x, errors) if e is not None]
            valid_errors = [e for e in errors if e is not None]
            plt.plot(valid_x, valid_errors, 'o-', label=f'{bin_label} citations')
    
    plt.title(f"{model_type.capitalize()} Model - Relative Error by Citation Count and Horizon")
    plt.xlabel("Time Horizon")
    plt.ylabel("Mean Relative Error")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(temporal_dir, 'error_by_citation_and_horizon.png'))
    plt.close()
    
    # Save temporal analysis metrics
    temporal_metrics = {
        'horizon_mae': horizon_mean_errors,
        'horizon_std': horizon_std_errors,
        'bias': mean_biases,
        'relative_bias': mean_relative_biases,
        'growth_rates': {
            'predicted': mean_pred_growth.tolist() if isinstance(mean_pred_growth, np.ndarray) else mean_pred_growth,
            'actual': mean_actual_growth.tolist() if isinstance(mean_actual_growth, np.ndarray) else mean_actual_growth
        }
    }
    
    # Add acceleration data if available
    if 'mean_pred_accel' in locals():
        temporal_metrics['acceleration_rates'] = {
            'predicted': mean_pred_accel.tolist() if isinstance(mean_pred_accel, np.ndarray) else mean_pred_accel,
            'actual': mean_actual_accel.tolist() if isinstance(mean_actual_accel, np.ndarray) else mean_actual_accel
        }
    
    with open(os.path.join(temporal_dir, 'temporal_metrics.json'), 'w') as f:
        json.dump(temporal_metrics, f, indent=2)
    
    print(f"Temporal pattern analysis complete. Results saved to {temporal_dir}")
    return temporal_metrics

def analyze_network_features(model, data, model_results, output_dir, model_type):
    print(f"Running network feature analysis for {model_type} model...")
    network_dir = os.path.join(output_dir, 'network')
    
    if data['citation_network'] is None:
        print("Citation network data not available. Skipping network analysis.")
        return None
    
    citation_network = data['citation_network']
    print("DEBUG: Citation network keys sample (first 20):", list(citation_network.keys())[:20])
    states = data['states']
    predictions = model_results['predictions']
    citation_targets = data['citation_targets']
    val_state_ids = data['val_states']
    field_targets = data['field_targets']
    id_to_field = data['id_to_field']
    
    # Check if mapping exists
    if 'state_id_to_arxiv_id' not in data:
        print("ERROR: state_id_to_arxiv_id mapping not found in data. Cannot proceed with network analysis.")
        return None
    # print(f"DEBUG: Using state_id_to_arxiv_id mapping with {len(data['state_id_to_arxiv_id'])} entries")
    # print("DEBUG: Mapping sample (first 20):", {k: v for k, v in list(data['state_id_to_arxiv_id'].items())[:20]})
    
    reference_counts = []
    citation_counts = []
    predicted_citations = []
    actual_citations = []
    reference_diversity = []
    paper_ids = []
    field_ids = []
    
    processed_count = 0
    skipped_count = 0
    debug_limit = 30  # Limit debug prints to first 30 iterations
    debug_count = 0
    
    for state_id in val_state_ids:
        if state_id not in predictions:
            continue
        
        state = states[state_id]
        paper_id = data['state_id_to_arxiv_id'].get(state_id)
        if paper_id is None:
            if debug_count < debug_limit:
                print(f"WARNING: No paper_id mapped for state_id={state_id}, falling back to state_id")
            paper_id = str(state.state_id)
        
        if debug_count < debug_limit:
            print(f"DEBUG: state_id={state_id}, mapped paper_id={paper_id}")
        
        if paper_id not in citation_network:
            skipped_count += 1
            if debug_count < debug_limit:
                print(f"DEBUG: paper_id={paper_id} not in citation_network, skipping")
            debug_count += 1
            continue
        
        processed_count += 1
        if debug_count < debug_limit:
            debug_count += 1
        
        ref_count = len(citation_network[paper_id]['references'])
        cit_count = len(citation_network[paper_id]['citations'])
        
        reference_counts.append(ref_count)
        citation_counts.append(cit_count)
        predicted_citations.append(np.mean(predictions[state_id]))
        actual_citations.append(np.mean(citation_targets[state_id]))
        reference_diversity.append(state.reference_diversity)
        paper_ids.append(paper_id)
        field_ids.append(field_targets.get(state_id))
    
    # print(f"DEBUG: Number of papers processed = {processed_count}")
    # print(f"DEBUG: Number of papers skipped = {skipped_count}")
    # print(f"DEBUG: reference_counts sample (first 30): {reference_counts[:30]}")
    # print(f"DEBUG: citation_counts sample (first 30): {citation_counts[:30]}")
    # print(f"DEBUG: predicted_citations sample (first 30): {predicted_citations[:30]}")
    # print(f"DEBUG: actual_citations sample (first 30): {actual_citations[:30]}")
    # print(f"DEBUG: reference_diversity sample (first 30): {reference_diversity[:30]}")
    
    # Convert lists to numpy arrays
    reference_counts = np.array(reference_counts)
    citation_counts = np.array(citation_counts)
    predicted_citations = np.array(predicted_citations)
    actual_citations = np.array(actual_citations)
    reference_diversity = np.array(reference_diversity)
    
    if len(reference_counts) == 0 or len(predicted_citations) == 0:
        print("DEBUG: reference_counts or predicted_citations array is empty or too short for correlation (predicted).")
    if len(reference_diversity) == 0 or len(actual_citations) == 0:
        print("DEBUG: reference_diversity or actual_citations array is empty or too short for correlation (diversity actual).")
    if len(reference_diversity) == 0 or len(predicted_citations) == 0:
        print("DEBUG: reference_diversity or predicted_citations array is empty or too short for correlation (diversity predicted).")
    
    # 1. Base analysis: relationship between reference count and citations
    plt.figure(figsize=(10, 6))
    plt.scatter(reference_counts, actual_citations, alpha=0.5, label='Actual')
    plt.scatter(reference_counts, predicted_citations, alpha=0.5, label='Predicted')
    plt.title(f"{model_type.capitalize()} Model - References vs Citations")
    plt.xlabel("Number of References")
    plt.ylabel("Citation Count")
    plt.legend()
    plt.savefig(os.path.join(network_dir, 'references_vs_citations.png'))
    plt.close()
    
    # Calculate correlations with error handling
    try:
        ref_actual_corr = np.corrcoef(reference_counts, actual_citations)[0, 1]
    except Exception as e:
        print(f"DEBUG: Error calculating ref_actual_corr: {e}")
        ref_actual_corr = float('nan')
    try:
        ref_pred_corr = np.corrcoef(reference_counts, predicted_citations)[0, 1]
    except Exception as e:
        print(f"DEBUG: Error calculating ref_pred_corr: {e}")
        ref_pred_corr = float('nan')
    
    # 2. Reference diversity analysis
    plt.figure(figsize=(10, 6))
    plt.scatter(reference_diversity, actual_citations, alpha=0.5, label='Actual')
    plt.scatter(reference_diversity, predicted_citations, alpha=0.5, label='Predicted')
    plt.title(f"{model_type.capitalize()} Model - Reference Diversity vs Citations")
    plt.xlabel("Reference Diversity")
    plt.ylabel("Citation Count")
    plt.legend()
    plt.savefig(os.path.join(network_dir, 'diversity_vs_citations.png'))
    plt.close()
    
    try:
        diversity_actual_corr = np.corrcoef(reference_diversity, actual_citations)[0, 1]
    except Exception as e:
        print(f"DEBUG: Error calculating diversity_actual_corr: {e}")
        diversity_actual_corr = float('nan')
    try:
        diversity_pred_corr = np.corrcoef(reference_diversity, predicted_citations)[0, 1]
    except Exception as e:
        print(f"DEBUG: Error calculating diversity_pred_corr: {e}")
        diversity_pred_corr = float('nan')
    
    # 3. Field citation patterns
    field_data = {}
    for i, field_id in enumerate(field_ids):
        if field_id is None:
            continue
        field_name = id_to_field.get(field_id, f"Field {field_id}")
        if field_name not in field_data:
            field_data[field_name] = {
                'ref_counts': [],
                'cit_counts': [],
                'ref_diversity': [],
                'actual_citations': [],
                'predicted_citations': []
            }
        field_data[field_name]['ref_counts'].append(reference_counts[i])
        field_data[field_name]['cit_counts'].append(citation_counts[i])
        field_data[field_name]['ref_diversity'].append(reference_diversity[i])
        field_data[field_name]['actual_citations'].append(actual_citations[i])
        field_data[field_name]['predicted_citations'].append(predicted_citations[i])
    
    field_stats = {}
    for field_name, values in field_data.items():
        if len(values['ref_counts']) < 5:
            continue
        field_stats[field_name] = {
            'paper_count': len(values['ref_counts']),
            'mean_references': float(np.mean(values['ref_counts'])),
            'mean_citations': float(np.mean(values['cit_counts'])),
            'mean_diversity': float(np.mean(values['ref_diversity'])),
            'mean_actual_future_citations': float(np.mean(values['actual_citations'])),
            'mean_predicted_future_citations': float(np.mean(values['predicted_citations'])),
            'ref_citation_correlation': float(np.corrcoef(values['ref_counts'], values['actual_citations'])[0, 1]) if len(values['ref_counts']) > 1 else 0
        }
    
    if field_stats:
        sorted_fields = sorted(field_stats.keys(), key=lambda f: field_stats[f]['mean_actual_future_citations'], reverse=True)
        plt.figure(figsize=(14, 8))
        x = np.arange(len(sorted_fields))
        width = 0.35
        plt.bar(x - width/2, [field_stats[f]['mean_references'] for f in sorted_fields], width, label='Mean References', color='skyblue')
        plt.bar(x + width/2, [field_stats[f]['mean_actual_future_citations'] for f in sorted_fields], width, label='Mean Future Citations', color='salmon')
        plt.title(f"{model_type.capitalize()} Model - References and Citations by Field")
        plt.xlabel("Research Field")
        plt.ylabel("Count")
        plt.xticks(x, sorted_fields, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(network_dir, 'field_citation_patterns.png'))
        plt.close()
        
        plt.figure(figsize=(14, 8))
        plt.bar(x - width/2, [field_stats[f]['mean_diversity'] for f in sorted_fields], width, label='Mean Reference Diversity', color='lightgreen')
        plt.bar(x + width/2, [field_stats[f]['ref_citation_correlation'] for f in sorted_fields], width, label='Ref-Citation Correlation', color='purple')
        plt.title(f"{model_type.capitalize()} Model - Diversity and Correlation by Field")
        plt.xlabel("Research Field")
        plt.ylabel("Value")
        plt.xticks(x, sorted_fields, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(network_dir, 'field_diversity_patterns.png'))
        plt.close()
    
    # 4. Emerging research field detection (using temporal data)
    if len(paper_ids) > 50:
        try:
            has_temporal_data = all(hasattr(states[sid], 'time_index') for sid in val_state_ids if sid in states)
            if has_temporal_data:
                time_periods = {}
                for i, state_id in enumerate(val_state_ids):
                    state = states[state_id]
                    pid = str(state.paper_id) if hasattr(state, 'paper_id') else str(state.state_id)
                    # Use fallback as above to try and get the original arXiv id
                    state_dict = vars(state)
                    if pid.isdigit() or (isinstance(pid, str) and '.' not in pid):
                        if 'arxiv_id' in state_dict:
                            pid = state_dict['arxiv_id']
                    if pid not in paper_ids:
                        continue
                    time_index = state.time_index
                    field_id = field_targets.get(state_id)
                    if field_id is None:
                        continue
                    field_name = id_to_field.get(field_id, f"Field {field_id}")
                    if time_index not in time_periods:
                        time_periods[time_index] = {}
                    if field_name not in time_periods[time_index]:
                        time_periods[time_index][field_name] = {'count': 0, 'citations': [], 'growth_rate': 0}
                    time_periods[time_index][field_name]['count'] += 1
                    if i < len(actual_citations):
                        time_periods[time_index][field_name]['citations'].append(actual_citations[i])
                
                for time_index, fields in time_periods.items():
                    for field_name, data in fields.items():
                        data['avg_citations'] = np.mean(data['citations']) if data['citations'] else 0
                        
                sorted_times = sorted(time_periods.keys())
                field_growth = {}
                for i in range(1, len(sorted_times)):
                    prev_time = sorted_times[i-1]
                    curr_time = sorted_times[i]
                    common_fields = set(time_periods[prev_time].keys()) & set(time_periods[curr_time].keys())
                    for field in common_fields:
                        prev_data = time_periods[prev_time][field]
                        curr_data = time_periods[curr_time][field]
                        if 'avg_citations' not in prev_data or 'avg_citations' not in curr_data:
                            continue
                        count_growth = (curr_data['count'] / max(1, prev_data['count'])) - 1
                        citation_growth = (curr_data['avg_citations'] / max(1, prev_data['avg_citations'])) - 1
                        combined_growth = (count_growth + citation_growth) / 2
                        field_growth.setdefault(field, []).append(combined_growth)
                avg_field_growth = {field: np.mean(growth_rates) if growth_rates else 0 for field, growth_rates in field_growth.items()}
                if avg_field_growth:
                    sorted_fields_growth = sorted(avg_field_growth.keys(), key=lambda f: avg_field_growth[f], reverse=True)
                    num_emerging = max(1, len(sorted_fields_growth) // 3)
                    emerging_fields = sorted_fields_growth[:num_emerging]
                    
                    plt.figure(figsize=(12, 8))
                    fields_to_plot = sorted_fields_growth[:min(15, len(sorted_fields_growth))]
                    growth_values = [avg_field_growth[f] * 100 for f in fields_to_plot]
                    bars = plt.bar(fields_to_plot, growth_values)
                    for i, field in enumerate(fields_to_plot):
                        bars[i].set_color('red' if field in emerging_fields else 'blue')
                    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    plt.title(f"{model_type.capitalize()} Model - Field Growth Rates (Emerging Fields in Red)")
                    plt.xlabel("Research Field")
                    plt.ylabel("Average Growth Rate (%)")
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(network_dir, 'emerging_fields.png'))
                    plt.close()
                    
                    plt.figure(figsize=(12, 8))
                    top_emerging = emerging_fields[:min(5, len(emerging_fields))]
                    for field in top_emerging:
                        if field in field_growth and len(field_growth[field]) > 0:
                            plt.plot(range(1, len(field_growth[field]) + 1),
                                     [rate * 100 for rate in field_growth[field]],
                                     'o-', label=field)
                    plt.title(f"{model_type.capitalize()} Model - Growth Trajectory of Emerging Fields")
                    plt.xlabel("Time Period")
                    plt.ylabel("Growth Rate (%)")
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.savefig(os.path.join(network_dir, 'emerging_field_trajectories.png'))
                    plt.close()
        except Exception as e:
            print(f"Error in emerging field detection: {str(e)}")
    
    # 5. Citation network visualization
    if field_ids and len(set(field_ids) - {None}) > 1:
        try:
            field_names = sorted(set(id_to_field.get(fid, f"Field {fid}") for fid in set(field_ids) if fid is not None))
            citation_matrix = np.zeros((len(field_names), len(field_names)))
            for i, paper_id in enumerate(paper_ids):
                if i >= len(field_ids) or field_ids[i] is None:
                    continue
                source_field = id_to_field.get(field_ids[i], f"Field {field_ids[i]}")
                source_idx = field_names.index(source_field)
                if paper_id in citation_network and 'references' in citation_network[paper_id]:
                    for ref_id in citation_network[paper_id]['references'][:100]:
                        ref_idx = next((j for j, pid in enumerate(paper_ids) if pid == ref_id), None)
                        if ref_idx is not None and ref_idx < len(field_ids) and field_ids[ref_idx] is not None:
                            target_field = id_to_field.get(field_ids[ref_idx], f"Field {field_ids[ref_idx]}")
                            target_idx = field_names.index(target_field)
                            citation_matrix[source_idx, target_idx] += 1
            row_sums = citation_matrix.sum(axis=1, keepdims=True)
            normalized_matrix = np.divide(citation_matrix, row_sums, out=np.zeros_like(citation_matrix), where=row_sums != 0)
            plt.figure(figsize=(14, 12))
            sns.heatmap(normalized_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                      xticklabels=field_names, yticklabels=field_names)
            plt.title(f"{model_type.capitalize()} Model - Field-to-Field Citation Patterns")
            plt.xlabel("Cited Field")
            plt.ylabel("Citing Field")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(network_dir, 'field_citation_network.png'))
            plt.close()
            
            in_degree = citation_matrix.sum(axis=0)
            out_degree = citation_matrix.sum(axis=1)
            centrality = in_degree / (in_degree.sum() or 1)
            plt.figure(figsize=(12, 6))
            sorted_indices = np.argsort(centrality)[::-1]
            sorted_fields = [field_names[i] for i in sorted_indices]
            sorted_centrality = centrality[sorted_indices]
            plt.bar(sorted_fields, sorted_centrality)
            plt.title(f"{model_type.capitalize()} Model - Field Citation Centrality")
            plt.xlabel("Research Field")
            plt.ylabel("Centrality Score")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(network_dir, 'field_centrality.png'))
            plt.close()
            
            network_field_stats = {}
            for i, field in enumerate(field_names):
                network_field_stats[field] = {
                    'in_degree': float(in_degree[i]),
                    'out_degree': float(out_degree[i]),
                    'centrality': float(centrality[i])
                }
            with open(os.path.join(network_dir, 'field_network_stats.json'), 'w') as f:
                json.dump(network_field_stats, f, indent=2)
                
        except Exception as e:
            print(f"Error in citation network visualization: {str(e)}")
            
    # Save all network analysis metrics
    network_stats = {
        'ref_actual_correlation': float(ref_actual_corr),
        'ref_pred_correlation': float(ref_pred_corr),
        'diversity_actual_correlation': float(diversity_actual_corr),
        'diversity_pred_correlation': float(diversity_pred_corr),
        'mean_references': float(np.mean(reference_counts)) if len(reference_counts) > 0 else 0,
        'mean_citations': float(np.mean(citation_counts)) if len(citation_counts) > 0 else 0,
        'field_stats': field_stats
    }
    
    with open(os.path.join(network_dir, 'network_stats.json'), 'w') as f:
        json.dump(network_stats, f, indent=2)
    
    print(f"Network feature analysis complete. Results saved to {network_dir}")
    return network_stats

def analyze_feature_importance(model, data, model_results, output_dir, model_type):
    """Analyze feature importance in the model's predictions using a variety of techniques"""
    print(f"Running feature importance analysis for {model_type} model...")
    importance_dir = os.path.join(output_dir, 'importance')
    
    # Extract data
    states = data['states']
    citation_targets = data['citation_targets']
    val_state_ids = data['val_states']
    predictions = model_results['predictions']
    
    # 1. Correlation analysis
    # Create feature matrix
    feature_names = [
        'citation_count', 'reference_diversity', 'field_impact_factor', 
        'collaboration_info', 'time_index'
    ]
    
    # Check if collaboration_info has any non-zero values
    has_collaboration_data = False
    for state_id in val_state_ids[:100]:  # Check a sample
        if state_id in states and states[state_id].collaboration_info > 0:
            has_collaboration_data = True
            break
    
    if not has_collaboration_data:
        print("Warning: All collaboration_info values appear to be zero. This feature will be excluded from analysis.")
        feature_names.remove('collaboration_info')
    
    feature_matrix = []
    target_values = []
    pred_values = []
    
    for state_id in val_state_ids:
        if state_id not in predictions or state_id not in citation_targets:
            continue
            
        state = states[state_id]
        features = [
            state.citation_count,
            state.reference_diversity,
            state.field_impact_factor
        ]
        
        # Only add collaboration_info if it has meaningful values
        if has_collaboration_data:
            features.append(state.collaboration_info)
            
        features.append(state.time_index)
        
        # Average target and prediction across horizons
        avg_target = np.mean(citation_targets[state_id])
        avg_pred = np.mean(predictions[state_id])
        
        feature_matrix.append(features)
        target_values.append(avg_target)
        pred_values.append(avg_pred)
    
    # Calculate correlations with target values
    correlations = []
    for i, feature_name in enumerate(feature_names):
        feature_values = feature_matrix[:, i]
        corr_target = np.corrcoef(feature_values, target_values)[0, 1]
        corr_pred = np.corrcoef(feature_values, pred_values)[0, 1]
        correlations.append({
            'feature': feature_name,
            'corr_with_target': float(corr_target),
            'corr_with_prediction': float(corr_pred)
        })
    
    # Plot correlation analysis
    plt.figure(figsize=(12, 6))
    
    features = [c['feature'] for c in correlations]
    corr_targets = [c['corr_with_target'] for c in correlations]
    corr_preds = [c['corr_with_prediction'] for c in correlations]
    
    x = np.arange(len(features))
    width = 0.35
    
    plt.bar(x - width/2, corr_targets, width, label='Correlation with Actual Citations', color='blue')
    plt.bar(x + width/2, corr_preds, width, label='Correlation with Predicted Citations', color='red')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    plt.title(f"{model_type.capitalize()} Model - Feature Correlations")
    plt.xlabel("Features")
    plt.ylabel("Correlation Coefficient")
    plt.xticks(x, [f.replace('_', ' ').title() for f in features])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(importance_dir, 'feature_correlations.png'))
    plt.close()
    
    # 2. Simple feature ablation study
    # Select a subset of papers for partial dependence analysis
    num_samples = min(1000, len(val_state_ids))
    sample_ids = random.sample(val_state_ids, num_samples)
    
    # Initialize containers for partial dependence plots
    pdp_data = {}
    
    # Map feature names to state attributes for ablation
    feature_to_attr = {
        'citation_count': 'citation_count',
        'reference_diversity': 'reference_diversity',
        'field_impact_factor': 'field_impact_factor',
        'collaboration_info': 'collaboration_info'
        # time_index isn't included as we're not perturbing time
    }
    
    device = next(model.parameters()).device
    
    for feature_name, attr_name in feature_to_attr.items():
        print(f"  Calculating partial dependence for {feature_name}...")
        
        # Determine feature range for ablation
        feature_values = [getattr(states[sid], attr_name) for sid in sample_ids if sid in states]
        min_val = min(feature_values)
        max_val = max(feature_values)
        
        # Create test values spanning the feature range
        if feature_name == 'citation_count':
            # For citation count, use specific values
            test_values = [0, 1, 2, 5, 10, 20, 50, 100]
        else:
            # For other features, use evenly spaced values
            test_values = np.linspace(min_val, max_val, 10)
        
        # Initialize results container
        pdp_results = []
        
        # For each test value, modify all papers and get predictions
        for test_value in test_values:
            batch_predictions = []
            
            # Process in smaller batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(sample_ids), batch_size):
                batch_ids = sample_ids[i:i+batch_size]
                batch_states = []
                
                for sid in batch_ids:
                    if sid not in states:
                        continue
                        
                    # Create modified state
                    state = copy.deepcopy(states[sid])
                    setattr(state, attr_name, test_value)
                    batch_states.append(state)
                
                if not batch_states:
                    continue
                
                # Convert states to tensors
                batch_data = [s.to_numpy() for s in batch_states]
                batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=device).unsqueeze(1)
                
                # Forward pass through model
                with torch.no_grad():
                    _, _, citation_params, _ = model(batch_tensor)
                    
                    # Get predictions for each horizon
                    for h in range(model.horizons):
                        horizon_preds = []
                        
                        try:
                            # Handle different model types
                            if hasattr(model, 'citation_head') and isinstance(model.citation_head, MultiHeadCitationPredictor):
                                if isinstance(citation_params, tuple):
                                    params, _ = citation_params
                                    nb_dist, mean = model.citation_head.get_distribution_for_horizon(params, h)
                                else:
                                    nb_dist, mean = model.get_citation_distribution(citation_params, h)
                            else:
                                nb_dist, mean = model.get_citation_distribution(citation_params, h)
                                
                            # Transform if using log space
                            if hasattr(model, 'use_log_transform') and model.use_log_transform:
                                horizon_preds = torch.exp(mean).cpu().numpy() - 1
                            else:
                                horizon_preds = mean.cpu().numpy()
                                
                        except Exception as e:
                            print(f"Error getting predictions: {str(e)}")
                            horizon_preds = np.zeros(len(batch_states))
                            
                        batch_predictions.extend(horizon_preds)
            
            # Calculate mean prediction across all papers for this feature value
            if batch_predictions:
                mean_prediction = np.mean(batch_predictions)
                pdp_results.append({
                    'feature_value': float(test_value),
                    'mean_prediction': float(mean_prediction)
                })
        
        # Store partial dependence plot data
        pdp_data[feature_name] = pdp_results
    
    # Plot partial dependence plots
    plt.figure(figsize=(16, 12))
    
    for i, (feature_name, results) in enumerate(pdp_data.items()):
        plt.subplot(2, 2, i+1)
        
        x = [r['feature_value'] for r in results]
        y = [r['mean_prediction'] for r in results]
        
        plt.plot(x, y, 'o-', linewidth=2)
        plt.title(f"Partial Dependence: {feature_name.replace('_', ' ').title()}")
        plt.xlabel(feature_name.replace('_', ' ').title())
        plt.ylabel("Mean Predicted Citations")
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(importance_dir, 'partial_dependence_plots.png'))
    plt.close()
    
    # 3. Citation count vs prediction relationship
    citation_analysis = []
    
    for state_id in val_state_ids:
        if state_id not in predictions or state_id not in citation_targets:
            continue
            
        state = states[state_id]
        
        # Average target and prediction across horizons
        avg_target = np.mean(citation_targets[state_id])
        avg_pred = np.mean(predictions[state_id])
        
        citation_analysis.append({
            'state_id': state_id,
            'citation_count': state.citation_count,
            'target': float(avg_target),
            'prediction': float(avg_pred)
        })
    
    # Sort by citation count
    citation_analysis.sort(key=lambda x: x['citation_count'])
    
    # Plot citation count vs prediction
    plt.figure(figsize=(12, 8))
    
    # Extract data
    c_counts = [d['citation_count'] for d in citation_analysis]
    c_targets = [d['target'] for d in citation_analysis]
    c_preds = [d['prediction'] for d in citation_analysis]
    
    # Create scatter plot
    plt.scatter(c_counts, c_targets, alpha=0.5, label='Actual Citations', color='blue')
    plt.scatter(c_counts, c_preds, alpha=0.5, label='Predicted Citations', color='red')
    
    # Add trend lines
    from scipy.ndimage import gaussian_filter1d
    
    # Sort data for trend lines
    sorted_indices = np.argsort(c_counts)
    sorted_counts = np.array([c_counts[i] for i in sorted_indices])
    sorted_targets = np.array([c_targets[i] for i in sorted_indices])
    sorted_preds = np.array([c_preds[i] for i in sorted_indices])
    
    # Apply smoothing
    smoothed_targets = gaussian_filter1d(sorted_targets, sigma=5)
    smoothed_preds = gaussian_filter1d(sorted_preds, sigma=5)
    
    plt.plot(sorted_counts, smoothed_targets, color='blue', linewidth=2)
    plt.plot(sorted_counts, smoothed_preds, color='red', linewidth=2)
    
    plt.title(f"{model_type.capitalize()} Model - Citation Count vs Predictions")
    plt.xlabel("Paper Citation Count")
    plt.ylabel("Average Future Citations")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(importance_dir, 'citation_count_vs_prediction.png'))
    plt.close()
    
    # Save feature importance data
    importance_data = {
        'correlations': correlations,
        'partial_dependence': pdp_data
    }
    
    with open(os.path.join(importance_dir, 'feature_importance.json'), 'w') as f:
        json.dump(importance_data, f, indent=2)
    
    print(f"Feature importance analysis complete. Results saved to {importance_dir}")
    return importance_data

def analyze_error_cases(model, data, model_results, output_dir, model_type):
    """Analyze cases with highest prediction errors"""
    print(f"Running error case analysis for {model_type} model...")
    error_dir = os.path.join(output_dir, 'error')
    os.makedirs(error_dir, exist_ok=True)
    
    # Extract data
    predictions = model_results['predictions']
    citation_targets = data['citation_targets']
    val_state_ids = data['val_states']
    states = data['states']
    state_id_to_arxiv_id = data['state_id_to_arxiv_id']  # Add mapping
    
    # Calculate error for each paper
    error_data = []
    
    for state_id in val_state_ids:
        if state_id not in predictions:
            continue
        
        pred = predictions[state_id]
        target = citation_targets[state_id]
        
        # Mean absolute error across horizons
        mae = np.mean(np.abs(np.array(pred) - np.array(target)))  # Ensure NumPy arrays for subtraction
        
        # Relative error
        mean_target = np.mean(target)
        relative_error = mae / (mean_target + 1)  # Add 1 to handle zero citations
        
        paper_id = state_id_to_arxiv_id.get(state_id, str(state_id))
        error_data.append({
            'state_id': state_id,
            'paper_id': paper_id,
            'citation_count': states[state_id].citation_count,
            'predictions': pred if isinstance(pred, list) else pred.tolist(),
            'targets': target,
            'mae': float(mae),
            'relative_error': float(relative_error)
        })
    
    # Sort by error (both absolute and relative)
    error_data_abs = sorted(error_data, key=lambda x: x['mae'], reverse=True)
    error_data_rel = sorted(error_data, key=lambda x: x['relative_error'], reverse=True)
    
    # Save top 100 worst prediction cases
    with open(os.path.join(error_dir, 'worst_absolute_errors.json'), 'w') as f:
        json.dump(error_data_abs[:100], f, indent=2)
    
    with open(os.path.join(error_dir, 'worst_relative_errors.json'), 'w') as f:
        json.dump(error_data_rel[:100], f, indent=2)
    
    # Plot error distribution by citation count
    citation_counts = [data['citation_count'] for data in error_data]
    maes = [data['mae'] for data in error_data]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(citation_counts, maes, alpha=0.5)
    plt.title(f"{model_type.capitalize()} Model - Error vs Citation Count")
    plt.xlabel("Citation Count")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(error_dir, 'error_vs_citation.png'))
    plt.close()
    
    # Analyze error patterns by field
    field_errors = defaultdict(list)
    for data_point in error_data:
        state_id = data_point['state_id']
        if state_id in data['field_targets']:
            field_id = data['field_targets'][state_id]
            field_name = data['id_to_field'].get(field_id, f"Field {field_id}")
            field_errors[field_name].append(data_point['mae'])
    
    # Calculate mean error by field
    field_mean_errors = {field: float(np.mean(errors)) for field, errors in field_errors.items()}
    
    # Plot error by field
    plt.figure(figsize=(12, 6))
    fields = list(field_mean_errors.keys())
    mean_errors = list(field_mean_errors.values())
    sorted_indices = np.argsort(mean_errors)[::-1]  # Sort by descending error
    
    plt.bar([fields[i] for i in sorted_indices], [mean_errors[i] for i in sorted_indices])
    plt.title(f"{model_type.capitalize()} Model - Mean Error by Research Field")
    plt.xlabel("Research Field")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(error_dir, 'error_by_field.png'))
    plt.close()
    
    print(f"Error case analysis complete. Results saved to {error_dir}")
    return {
        'top_absolute_errors': error_data_abs[:5],
        'top_relative_errors': error_data_rel[:5],
        'field_errors': field_mean_errors
    }
    
def compare_models(high_model, low_model, data, output_dir, threshold):
    """Compare high and low citation models on boundary cases"""
    print("Running model comparison analysis...")
    comparison_dir = os.path.join(output_dir, 'comparison')
    
    # Only run if both models are provided
    if high_model is None or low_model is None:
        print("Both models required for comparison. Skipping.")
        return None
    
    # Find boundary cases - papers with citations near the threshold
    boundary_range = (threshold - 5, threshold + 5)
    boundary_states = {}
    
    for state_id, state in data['states'].items():
        if boundary_range[0] <= state.citation_count <= boundary_range[1]:
            boundary_states[state_id] = state
    
    print(f"Found {len(boundary_states)} boundary cases with {boundary_range[0]}-{boundary_range[1]} citations")
    
    # Run both models on boundary cases
    device = next(high_model.parameters()).device
    
    # Process in smaller batches to avoid memory issues
    batch_size = 16
    state_ids = list(boundary_states.keys())
    
    high_preds = {}
    low_preds = {}
    
    with torch.no_grad():
        for i in range(0, len(state_ids), batch_size):
            batch_ids = state_ids[i:i+batch_size]
            batch_states = [boundary_states[sid].to_numpy() for sid in batch_ids]
            state_tensor = torch.tensor(batch_states, dtype=torch.float32, device=device).unsqueeze(1)
            
            # High model predictions
            _, _, high_citation_params, _ = high_model(state_tensor)
            
            # Low model predictions
            _, _, low_citation_params, _ = low_model(state_tensor)
            
            # Process predictions for both models
            for j, sid in enumerate(batch_ids):
                # High model
                high_preds_horizon = []
                for h in range(high_model.horizons):
                    _, mean = high_model.get_citation_distribution(high_citation_params, h)
                    # Transform predictions if using log
                    if hasattr(high_model, 'use_log_transform') and high_model.use_log_transform:
                        pred = torch.exp(mean).cpu().item() - 1
                    else:
                        pred = mean.cpu().item()
                    high_preds_horizon.append(pred)
                high_preds[sid] = high_preds_horizon
                
                # Low model
                low_preds_horizon = []
                for h in range(low_model.horizons):
                    # Handle MultiHeadCitationPredictor for low model if present
                    if hasattr(low_model, 'citation_head') and isinstance(low_model.citation_head, MultiHeadCitationPredictor):
                        if isinstance(low_citation_params, tuple):
                            params, _ = low_citation_params
                            nb_dist, mean = low_model.citation_head.get_distribution_for_horizon(params, h)
                        else:
                            nb_dist, mean = low_model.get_citation_distribution(low_citation_params, h)
                    else:
                        nb_dist, mean = low_model.get_citation_distribution(low_citation_params, h)
                    
                    # Transform predictions if using log
                    if hasattr(low_model, 'use_log_transform') and low_model.use_log_transform:
                        pred = torch.exp(mean).cpu().item() - 1
                    else:
                        pred = mean.cpu().item()
                    low_preds_horizon.append(pred)
                low_preds[sid] = low_preds_horizon
    
    # Calculate metrics for each model on boundary cases
    targets = {sid: data['citation_targets'][sid] for sid in boundary_states if sid in data['citation_targets']}
    
    high_predictions = [high_preds[sid] for sid in targets]
    low_predictions = [low_preds[sid] for sid in targets]
    target_values = list(targets.values())
    
    high_metrics = CitationMetrics().compute(torch.tensor(high_predictions), torch.tensor(target_values))
    low_metrics = CitationMetrics().compute(torch.tensor(low_predictions), torch.tensor(target_values))
    
    # Save comparison metrics
    comparison_metrics = {
        'boundary_range': boundary_range,
        'num_boundary_cases': len(targets),
        'high_model': high_metrics,
        'low_model': low_metrics
    }
    
    with open(os.path.join(comparison_dir, 'boundary_comparison.json'), 'w') as f:
        json.dump(comparison_metrics, f, indent=2)
    
    # Plot comparison of MAE by citation count
    citation_counts = [boundary_states[sid].citation_count for sid in targets]
    high_errors = np.abs(np.array(high_predictions) - np.array(target_values)).mean(axis=1)
    low_errors = np.abs(np.array(low_predictions) - np.array(target_values)).mean(axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(citation_counts, high_errors, alpha=0.7, label='High Model', color='blue')
    plt.scatter(citation_counts, low_errors, alpha=0.7, label='Low Model', color='red')
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    plt.title('Model Performance Comparison on Boundary Cases')
    plt.xlabel('Citation Count')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig(os.path.join(comparison_dir, 'boundary_comparison.png'))
    plt.close()
    
    # Plot prediction differences
    prediction_diffs = np.array(high_predictions) - np.array(low_predictions)
    mean_diffs = np.mean(prediction_diffs, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(citation_counts, mean_diffs, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    plt.title('High Model vs Low Model Prediction Differences')
    plt.xlabel('Citation Count')
    plt.ylabel('Mean Prediction Difference (High - Low)')
    plt.legend()
    plt.savefig(os.path.join(comparison_dir, 'prediction_differences.png'))
    plt.close()
    
    print(f"Model comparison analysis complete. Results saved to {comparison_dir}")
    return comparison_metrics

def analyze_research_impact(model, data, model_results, output_dir, model_type):
    """Analyze factors that lead to research impact"""
    print(f"Running research impact analysis for {model_type} model...")
    impact_dir = os.path.join(output_dir, 'impact')
    os.makedirs(impact_dir, exist_ok=True)  # Ensure directory exists
    
    # Extract data
    states = data['states']
    citation_targets = data['citation_targets']
    predictions = model_results['predictions']
    actions = model_results.get('actions', {})  # Safely get actions, default to empty dict
    
    # Collect features that might correlate with impact
    impact_data = []
    
    for state_id, state in states.items():
        if state_id not in predictions or state_id not in citation_targets:
            continue
        
        # Use actual future citations as impact measure
        future_citations = citation_targets[state_id]
        max_future_citations = max(future_citations)
        
        # Get model's action for this paper
        if state_id in actions:
            action = actions[state_id]
            
            # Extract state and action features using dict access for actions
            impact_data.append({
                'state_id': state_id,
                'citation_count': state.citation_count,
                'reference_diversity': state.reference_diversity,
                'field_impact_factor': state.field_impact_factor,
                'collaboration_info': state.collaboration_info,
                'novelty_level': action.get('novelty_level', 0),  # Fixed: Dict access with default
                'collaboration_strategy': action.get('collaboration_strategy', 0),  # Fixed: Dict access with default
                'max_future_citations': float(max_future_citations)  # Ensure serializable
            })
    
    if not impact_data:
        print("WARNING: No impact data collected. Check if actions are available in model_results.")
        return {}
    
    # Convert to numpy arrays for analysis
    impact_array = np.array([(d['reference_diversity'], d['field_impact_factor'], 
                             d['collaboration_info'], d['novelty_level'], 
                             d['collaboration_strategy'], d['max_future_citations']) 
                            for d in impact_data])
    
    # Calculate correlations
    correlation_matrix = np.corrcoef(impact_array.T)
    feature_names = ['Reference Diversity', 'Field Impact Factor', 'Collaboration Score', 
                    'Novelty Level', 'Collaboration Strategy', 'Future Citations']
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
               xticklabels=feature_names, yticklabels=feature_names)
    plt.title(f"{model_type.capitalize()} Model - Feature Correlations with Impact")
    plt.tight_layout()
    plt.savefig(os.path.join(impact_dir, 'impact_correlations.png'))
    plt.close()
    
    # Identify high-impact papers (top 10%)
    impact_threshold = np.percentile([d['max_future_citations'] for d in impact_data], 90)
    high_impact = [d for d in impact_data if d['max_future_citations'] >= impact_threshold]
    low_impact = [d for d in impact_data if d['max_future_citations'] < impact_threshold]
    
    # Compare features between high and low impact papers
    comparison_features = ['reference_diversity', 'field_impact_factor', 
                         'collaboration_info', 'novelty_level', 'collaboration_strategy']
    
    # Plot comparison of means
    high_means = [np.mean([d[feature] for d in high_impact]) for feature in comparison_features]
    low_means = [np.mean([d[feature] for d in low_impact]) for feature in comparison_features]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(comparison_features))
    width = 0.35
    
    plt.bar(x - width/2, high_means, width, label='High Impact Papers')
    plt.bar(x + width/2, low_means, width, label='Low Impact Papers')
    
    plt.xlabel('Features')
    plt.ylabel('Mean Value')
    plt.title(f"{model_type.capitalize()} Model - Feature Comparison by Impact Level")
    plt.xticks(x, [' '.join(feature.split('_')).title() for feature in comparison_features])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid
    plt.tight_layout()
    plt.savefig(os.path.join(impact_dir, 'high_low_impact_comparison.png'))
    plt.close()
    
    # Save impact statistics
    impact_stats = {
        'impact_threshold': float(impact_threshold),  # Ensure serializable
        'high_impact_count': len(high_impact),
        'low_impact_count': len(low_impact),
        'feature_correlations': {
            feature_names[i]: float(correlation_matrix[i, -1]) for i in range(len(feature_names)-1)  # Ensure serializable
        },
        'high_impact_means': {feature: float(np.mean([d[feature] for d in high_impact])) for feature in comparison_features},
        'low_impact_means': {feature: float(np.mean([d[feature] for d in low_impact])) for feature in comparison_features}
    }
    
    with open(os.path.join(impact_dir, 'impact_statistics.json'), 'w') as f:
        json.dump(impact_stats, f, indent=2)
    
    print(f"Research impact analysis complete. Results saved to {impact_dir}")
    return impact_stats

def analyze_model_robustness(model, data, model_results, output_dir, model_type):
    """Analyze model robustness across different data subsets"""
    print(f"Running model robustness analysis for {model_type} model...")
    robustness_dir = os.path.join(output_dir, 'robustness')
    
    # Extract data
    states = data['states']
    citation_targets = data['citation_targets']
    val_state_ids = data['val_states']
    predictions = model_results['predictions']
    
    # Define different partitioning criteria for analysis
    partitions = {
        'citation_count': [
            (0, 5, 'Very Low Citations'),
            (6, 20, 'Low Citations'),
            (21, 50, 'Medium Citations'),
            (51, 100, 'High Citations'),
            (101, float('inf'), 'Very High Citations')
        ],
        'time_index': [
            (0, 4, 'Recent Papers'),
            (5, 8, 'Moderate Age'),
            (9, float('inf'), 'Older Papers')
        ],
        'reference_diversity': [
            (0, 0.3, 'Low Diversity'),
            (0.3, 0.7, 'Medium Diversity'),
            (0.7, 1.0, 'High Diversity')
        ]
    }
    
    # Calculate metrics for each partition
    results = {}
    
    for partition_key, ranges in partitions.items():
        partition_results = {}
        
        for lower, upper, label in ranges:
            # Filter states based on partition criteria
            subset_ids = [
                sid for sid in val_state_ids 
                if sid in predictions and
                lower <= getattr(states[sid], partition_key) <= upper
            ]
            
            # Skip if no data in this partition
            if not subset_ids:
                partition_results[label] = {
                    'count': 0,
                    'mae': None,
                    'rmse': None
                }
                continue
            
            # Collect predictions and targets
            subset_preds = [predictions[sid] for sid in subset_ids]
            subset_targets = [citation_targets[sid] for sid in subset_ids]
            
            # Calculate metrics
            metrics = CitationMetrics().compute(
                torch.tensor(subset_preds),
                torch.tensor(subset_targets)
            )
            
            partition_results[label] = {
                'count': len(subset_ids),
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'spearman': metrics['spearman']
            }
        
        results[partition_key] = partition_results
    
    # Save results
    with open(os.path.join(robustness_dir, 'robustness_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot metrics by partition
    for partition_key, partition_results in results.items():
        plt.figure(figsize=(12, 6))
        
        # Extract data for plotting
        labels = []
        mae_values = []
        rmse_values = []
        counts = []
        
        for label, metrics in partition_results.items():
            if metrics['count'] > 0:
                labels.append(label)
                mae_values.append(metrics['mae'])
                rmse_values.append(metrics['rmse'])
                counts.append(metrics['count'])
        
        # Plot metrics
        x = np.arange(len(labels))
        width = 0.35
        
        ax1 = plt.subplot(111)
        bars1 = ax1.bar(x - width/2, mae_values, width, label='MAE', color='blue')
        bars2 = ax1.bar(x + width/2, rmse_values, width, label='RMSE', color='red')
        
        # Add count annotation
        for i, bar in enumerate(bars1):
            ax1.annotate(f'n={counts[i]}', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
        
        ax1.set_xlabel(partition_key.replace('_', ' ').title())
        ax1.set_ylabel('Error Metric')
        ax1.set_title(f"{model_type.capitalize()} Model - Performance by {partition_key.replace('_', ' ').title()}")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(robustness_dir, f'robustness_{partition_key}.png'))
        plt.close()
    
    print(f"Model robustness analysis complete. Results saved to {robustness_dir}")
    return results

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = setup_output_directory(args.folder, args.model_type)
    
    print(f"Analysis results will be saved to: {output_dir}")
    
    # Determine which models to load
    models_to_load = []
    if args.model_type == 'low' or args.model_type == 'both':
        models_to_load.append('low')
    if args.model_type == 'high' or args.model_type == 'both':
        models_to_load.append('high')
    
    # Load models and data
    models = {}
    model_data = {}
    model_config = {}
    
    for model_type in models_to_load:
        try:
            models[model_type], model_data[model_type], model_config[model_type] = load_model_and_data(args, model_type)
            print(f"Successfully loaded {model_type} model")
            
            # Add this new code to fix collaboration information
            print(f"Attempting to fix collaboration information for {model_type} model...")
            fixed = fix_collaboration_info(model_data[model_type], model_type)
            if fixed:
                print(f"Successfully fixed collaboration information for {model_type} model")
            else:
                print(f"Warning: Could not fix collaboration information for {model_type} model. Analysis may show zero values for collaboration metrics.")
                
        except Exception as e:
            print(f"Error loading {model_type} model: {str(e)}")
            models[model_type] = None
    
    # Set device for running models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move models to device
    for model_type, model in models.items():
        if model is not None:
            models[model_type] = model.to(device)
    
    # Determine which analyses to run
    analyses_to_run = args.analyses
    if 'all' in analyses_to_run:
        analyses_to_run = [
            'prediction', 'field', 'strategy', 'temporal', 
            'network', 'importance', 'error', 'comparison', 
            'impact', 'robustness'
        ]
    
    # Run analyses for each model
    for model_type in models_to_load:
        if models[model_type] is None:
            print(f"Skipping analyses for {model_type} model since it could not be loaded")
            continue
        
        print(f"\n{'='*20} Analyzing {model_type.upper()} model {'='*20}\n")
        
        # Filter data if needed
        if args.model_type == 'both':
            # For 'both' mode, we still analyze each model with its appropriate data subset
            filtered_states, filtered_citation_targets, filtered_field_targets = filter_by_citation_count(
                model_data[model_type]['states'],
                model_data[model_type]['citation_targets'],
                model_data[model_type]['field_targets'],
                args.threshold,
                model_type
            )
            
            # Create filtered data
            filtered_data = {**model_data[model_type]}
            filtered_data['states'] = filtered_states
            filtered_data['citation_targets'] = filtered_citation_targets
            filtered_data['field_targets'] = filtered_field_targets
            
            active_data = filtered_data
        else:
            # For single model analysis, use full data
            active_data = model_data[model_type]
        
        # Run model on dataset to get predictions
        model_results = run_model_on_dataset(models[model_type], active_data['states'], device)
        
        # Run individual analyses
        for analysis in analyses_to_run:
            try:
                if analysis == 'prediction':
                    analyze_citation_prediction(models[model_type], active_data, model_results, output_dir, model_type, args.threshold)
                
                elif analysis == 'field':
                    analyze_field_classification(models[model_type], active_data, model_results, output_dir, model_type)
                
                elif analysis == 'strategy':
                    analyze_rl_strategy(models[model_type], active_data, model_results, output_dir, model_type)
                
                elif analysis == 'temporal':
                    analyze_temporal_patterns(models[model_type], active_data, model_results, output_dir, model_type)
                
                elif analysis == 'network':
                    analyze_network_features(models[model_type], active_data, model_results, output_dir, model_type)
                
                elif analysis == 'importance':
                    analyze_feature_importance(models[model_type], active_data, model_results, output_dir, model_type)
                
                elif analysis == 'error':
                    analyze_error_cases(models[model_type], active_data, model_results, output_dir, model_type)
                
                elif analysis == 'impact':
                    analyze_research_impact(models[model_type], active_data, model_results, output_dir, model_type)
                
                elif analysis == 'robustness':
                    analyze_model_robustness(models[model_type], active_data, model_results, output_dir, model_type)
            
            except Exception as e:
                print(f"Error running {analysis} analysis for {model_type} model: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Run model comparison if both models are loaded
    if 'comparison' in analyses_to_run and args.model_type == 'both':
        if models['high'] is not None and models['low'] is not None:
            compare_models(models['high'], models['low'], model_data['high'], output_dir, args.threshold)
    
    print(f"\nAnalysis complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
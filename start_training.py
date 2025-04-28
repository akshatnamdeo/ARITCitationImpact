import math
import os
import copy
import pickle
import random
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from torch.serialization import add_safe_globals

# Import ARIT components
from arit_citations import CitationCalibrationLayer, CitationScalingLayer, EnhancedRangeAwareCitationLoss, MultiHeadCitationPredictor, RangeAwareCitationLoss, analyze_citation_distribution, analyze_moe_experts, pretrain_citation_model, pretrain_moe_citation_model
from arit_config import ARITConfig
from arit_types import ARITAction
from arit_environment import ARITState, ARITTransitions, ImprovedRewardCalculator, RewardCalculator, ARITEnvironment
from arit_model import CitationGraphProcessor
from arit_model import ARITModel
from arit_training import ARITTrainer
from arit_evaluation import CitationMetrics, ResultsVisualizer

def load_arit_data(processed_dir="./arit_data/processed", citation_cap=500):
    print(f"Loading ARIT data with citation cap at {citation_cap}...")
    
    # Load standard data files
    with open(os.path.join(processed_dir, "train_states.pkl"), 'rb') as f:
        train_states_raw = pickle.load(f)
    with open(os.path.join(processed_dir, "val_states.pkl"), 'rb') as f:
        val_states_raw = pickle.load(f)
    with open(os.path.join(processed_dir, "transitions.pkl"), 'rb') as f:
        transitions_dict = pickle.load(f)
    
    # Load citation network data if available
    citation_network = None
    citation_network_path = os.path.join(processed_dir, "citation_network.pkl")
    external_papers_path = os.path.join(processed_dir, "external_papers.pkl")
    
    if os.path.exists(citation_network_path):
        print("Loading citation network data...")
        with open(citation_network_path, 'rb') as f:
            citation_network = pickle.load(f)
        print(f"Loaded citation network with {len(citation_network)} papers")
        
        if os.path.exists(external_papers_path):
            with open(external_papers_path, 'rb') as f:
                external_papers = pickle.load(f)
            print(f"Loaded data for {len(external_papers)} external papers")
    
    print(f"Loaded {len(train_states_raw)} training states, {len(val_states_raw)} validation states")
    
    # Apply citation cap
    for state in train_states_raw + val_states_raw:
        state['citation_count'] = min(state['citation_count'], citation_cap)
        if isinstance(state['future_citations'], list):
            state['future_citations'] = [min(cit, citation_cap) for cit in state['future_citations']]
    
    # Create ARITState objects
    arit_states = {}
    for state in train_states_raw + val_states_raw:
        # Extract network data if available
        network_data = state.get('network_data', {})
        
        # Extract fields
        primary_category = state.get('primary_category', None)
        field_target = state.get('field_target', None)
        
        arit_states[state['state_id']] = ARITState(
            content_embedding=state['content_embedding'],
            field_centroid=state['field_centroid'],
            reference_diversity=state['reference_diversity'],
            citation_count=state['citation_count'],
            field_impact_factor=state['field_impact_factor'],
            collaboration_info=state['collaboration_info'],
            time_index=state['time_index'],
            state_id=state['state_id'],
            future_citations=state['future_citations'],
            network_data=network_data,
            primary_category=primary_category,
            field_target=field_target
        )
    
    # Create transitions
    arit_transitions = ARITTransitions(arit_states, transitions_dict)
    
    # Extract targets
    citation_targets = {state['state_id']: state['future_citations'] 
                       for state in train_states_raw + val_states_raw}
    field_targets = {state['state_id']: state['field_target'] 
                    for state in train_states_raw + val_states_raw 
                    if 'field_target' in state}
    
    return {
        'states': arit_states,
        'transitions': arit_transitions,
        'train_states': [s['state_id'] for s in train_states_raw],
        'val_states': [s['state_id'] for s in val_states_raw],
        'citation_targets': citation_targets,
        'field_targets': field_targets,
        'citation_network': citation_network  # Include citation network
    }

def split_data_by_citation_range(data, threshold=20, min_citations=1):
    """
    Split data into low-citation and high-citation sets, excluding papers with fewer
    than min_citations.
    
    Args:
        data: Original data dictionary
        threshold: Citation count threshold (default: 20)
        min_citations: Minimum citation count to include (default: 1)
        
    Returns:
        low_data: Data dictionary for papers with citations between min_citations and threshold
        high_data: Data dictionary for papers with citations > threshold
    """
    # Initialize data dictionaries
    low_data = {
        'states': {},
        'train_states': [],
        'val_states': [],
        'citation_targets': {},
        'field_targets': {}
    }
    
    high_data = {
        'states': {},
        'train_states': [],
        'val_states': [],
        'citation_targets': {},
        'field_targets': {}
    }
    
    # Copy the transitions and network data
    if 'transitions' in data:
        low_data['transitions'] = data['transitions']
        high_data['transitions'] = data['transitions']
    
    if 'citation_network' in data:
        low_data['citation_network'] = data['citation_network']
        high_data['citation_network'] = data['citation_network']
    
    # Track skipped papers
    skipped_count = 0
    
    # Sort papers into appropriate datasets
    for state_id in data['states']:
        state = data['states'][state_id]
        citation_count = state.citation_count
        
        # Skip papers with fewer than min_citations
        if citation_count < min_citations:
            skipped_count += 1
            continue
        
        # Determine which dataset this paper belongs to
        if citation_count <= threshold:
            target_data = low_data
        else:
            target_data = high_data
        
        # Add to the appropriate dataset
        target_data['states'][state_id] = state
        
        if state_id in data['train_states']:
            target_data['train_states'].append(state_id)
        
        if state_id in data['val_states']:
            target_data['val_states'].append(state_id)
        
        if state_id in data['citation_targets']:
            target_data['citation_targets'][state_id] = data['citation_targets'][state_id]
        
        if state_id in data['field_targets']:
            target_data['field_targets'][state_id] = data['field_targets'][state_id]
    
    # Print dataset statistics
    print(f"Original dataset: {len(data['states'])} papers")
    print(f"Skipped papers (< {min_citations} citations): {skipped_count}")
    print(f"Low-citation dataset ({min_citations}-{threshold}): {len(low_data['states'])} papers")
    print(f"High-citation dataset (>{threshold}): {len(high_data['states'])} papers")
    
    return low_data, high_data

def train_dual_models(config, data, device, results_dir):
    """
    Train two specialized models: one for low-citation papers and one for high-citation papers.
    """
    # Split data into low and high citation datasets
    threshold = 20
    low_data, high_data = split_data_by_citation_range(data, threshold, min_citations=1)
    
    # Verify the split is clean
    low_range = [state.citation_count for state in low_data['states'].values()]
    high_range = [state.citation_count for state in high_data['states'].values()]
    
    print(f"\nLow model citation range: min={min(low_range)}, max={max(low_range)}, mean={sum(low_range)/len(low_range):.2f}")
    print(f"High model citation range: min={min(high_range)}, max={max(high_range)}, mean={sum(high_range)/len(high_range):.2f}")
    
    assert max(low_range) <= threshold, f"Low data contains citations > {threshold}!"
    assert min(high_range) > threshold, f"High data contains citations ≤ {threshold}!"
    
    # Set up results directories
    low_results_dir = os.path.join(results_dir, 'low_citation_model')
    high_results_dir = os.path.join(results_dir, 'high_citation_model')
    os.makedirs(low_results_dir, exist_ok=True)
    os.makedirs(high_results_dir, exist_ok=True)
    
    # Configure and train low-citation model
    print("\n======= TRAINING LOW-CITATION MODEL (≤20) =======\n")
    low_config = copy.deepcopy(config)
    high_config = copy.deepcopy(config)

    # CONFIGURATION FOR LOW-CITATION MODEL
    low_config.model_config["num_layers"] = 4
    low_config.model_config["d_model"] = 512
    low_config.model_config["dropout"] = 0.2
    
    # Create and train low-citation model
    first_state = next(iter(low_data['states'].values()))
    state_dim = len(first_state.to_numpy())
    
    # Create a model with our architecture
    low_model, low_optimizer, low_scheduler = setup_model_and_optimizer(low_config, state_dim, device)
    
    # Add our calibration layer to the model
    low_model.citation_calibration = CitationCalibrationLayer(num_ranges=3).to(device)
    
    # Use our custom loss
    custom_loss = EnhancedRangeAwareCitationLoss()
    
    low_scaler = torch.amp.GradScaler() if low_config.training_config.get("use_mixed_precision", False) else None
    
    # Pre-train low-citation model with enhanced loss
    low_pretrain_metrics, low_best_state = pretrain_citation_model(
        model=low_model,
        config=low_config,
        data=low_data,
        optimizer=low_optimizer,
        device=device,
        results_dir=low_results_dir,
        epochs=25,
        batch_size=32,
        scaler=low_scaler,
        use_log_transform=True,
        custom_loss=custom_loss
    )
    
    # Configure and train high-citation model
    print("\n======= TRAINING HIGH-CITATION MODEL (>20) =======\n")
    
    # Create and train high-citation model
    high_model, high_optimizer, high_scheduler = setup_model_and_optimizer(high_config, state_dim, device)
    high_scaler = torch.amp.GradScaler() if high_config.training_config.get("use_mixed_precision", False) else None
    
    # Pre-train high-citation model
    high_pretrain_metrics, high_best_state = pretrain_citation_model(
        model=high_model,
        config=high_config,
        data=high_data,
        optimizer=high_optimizer,
        device=device,
        results_dir=high_results_dir,
        epochs=6,
        batch_size=64,
        scaler=high_scaler,
        use_log_transform=True
    )
    
    # Save both models
    torch.save({
        'model_state_dict': low_model.state_dict(),
        'threshold': 20,
        'config': low_config.to_dict(),
        'metrics': low_pretrain_metrics
    }, os.path.join(results_dir, 'low_citation_model.pt'))
    
    torch.save({
        'model_state_dict': high_model.state_dict(),
        'threshold': 20,
        'config': high_config.to_dict(),
        'metrics': high_pretrain_metrics
    }, os.path.join(results_dir, 'high_citation_model.pt'))
    
    return low_model, high_model

def predict_citations_dual_model(low_model, high_model, states, device, threshold=20):
    """
    Make citation predictions using both models.
    
    Args:
        low_model: Model trained on low-citation papers
        high_model: Model trained on high-citation papers
        states: Dictionary of states to predict for
        device: Device to run models on
        threshold: Citation threshold for model selection
    
    Returns:
        Dictionary mapping state_id to citation predictions
    """
    low_model.eval()
    high_model.eval()
    
    predictions = {}
    
    with torch.no_grad():
        for state_id, state in states.items():
            state_np = state.to_numpy()
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            # Determine which model to use based on citation count
            if state.citation_count <= threshold:
                model = low_model
            else:
                model = high_model
            
            # Get prediction from the selected model
            _, _, citation_params, _ = model(state_tensor)
            
            citation_preds = []
            for h in range(model.horizons):
                nb_dist, mean = model.get_citation_distribution(citation_params, h)
                pred_mean = mean.cpu().item()
                citation_preds.append(pred_mean)
            
            predictions[state_id] = citation_preds
    
    return predictions


def validate_dual_models(low_model, high_model, val_states, citation_targets, field_targets, device, threshold=20):
    """
    Validate using both specialized models.
    
    Args:
        low_model: Model trained on low-citation papers
        high_model: Model trained on high-citation papers
        val_states: Validation states
        citation_targets: Dictionary mapping state_id to citation targets
        field_targets: Dictionary mapping state_id to field targets
        device: Device to run models on
        threshold: Citation threshold for model selection
        
    Returns:
        citation_metrics: Overall citation prediction metrics
        range_metrics: Metrics broken down by citation range
    """
    # Get predictions
    predictions = predict_citations_dual_model(
        low_model, high_model, val_states, device, threshold
    )
    
    # Convert to format expected by CitationMetrics
    all_predictions = []
    all_targets = []
    
    for state_id in citation_targets:
        if state_id in predictions:
            all_predictions.append(predictions[state_id])
            all_targets.append(citation_targets[state_id])
    
    # Compute metrics
    citation_metrics = CitationMetrics().compute(
        torch.tensor(all_predictions), 
        torch.tensor(all_targets)
    )
    
    # Compute metrics by citation range
    citation_ranges = [
        (0, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 500)
    ]
    
    range_metrics = {}
    for low, high in citation_ranges:
        range_preds = []
        range_targets = []
        
        for state_id in citation_targets:
            if state_id in predictions and state_id in val_states:
                # Use the current citation count to determine range
                citation_count = val_states[state_id].citation_count
                
                if low <= citation_count <= high:
                    range_preds.append(predictions[state_id])
                    range_targets.append(citation_targets[state_id])
        
        if range_preds:
            range_metrics[f"{low}-{high}"] = CitationMetrics().compute(
                torch.tensor(range_preds),
                torch.tensor(range_targets)
            )
            # Add count information
            range_metrics[f"{low}-{high}"]["count"] = len(range_preds)
    
    return citation_metrics, range_metrics

def setup_environment(config, data):
    """Set up the ARIT environment with citation network."""
    print("Setting up environment...")
    
    reward_calculator = ImprovedRewardCalculator(config)
    
    environment = ARITEnvironment(
        config=config,
        states=data['states'],
        transitions=data['transitions'],
        reward_calculator=reward_calculator,
        citation_network=data.get('citation_network')  # Pass citation network
    )
    
    return environment

def optimize_config_for_4060ti(config):
    """Optimize configuration for RTX 4060 Ti."""
    print("Optimizing configuration for RTX 4060 Ti...")
    
    # Batch size
    config.training_config["batch_size"] = 32
    
    # Add gradient accumulation for effective larger batch size
    config.training_config["gradient_accumulation_steps"] = 2
    
    config.training_config["ppo_epochs"] = 4
    
    # Enable mixed precision for memory efficiency
    config.training_config["use_mixed_precision"] = True
    
    if config.model_config["d_model"] > 512:
        config.model_config["d_model"] = 512
        
    config.training_config["warmup_steps"] = 500
    
    return config

def setup_model_and_optimizer(config, state_dim, device, is_low_citation=False, use_moe=False, num_experts=3):
    print("Setting up model and optimizer...")
    input_dim = state_dim
    if state_dim % config.model_config["n_heads"] != 0:
        d_model = (state_dim // config.model_config["n_heads"]) * config.model_config["n_heads"]
        config.model_config["d_model"] = d_model
        print(f"Adjusted d_model to {d_model} for head divisibility")
    
    # Create the model based on the selected approach
    if use_moe:
        print(f"Creating Mixture of Experts model with {num_experts} experts")
        model = ARITModel(config, input_dim=input_dim, 
                          use_multi_head_predictor=False,
                          use_moe=True, 
                          num_experts=num_experts).to(device)
    elif is_low_citation:
        # Use the MultiHeadCitationPredictor for low-citation models
        print("Creating low citation model with MultiHeadCitationPredictor")
        model = ARITModel(config, input_dim=input_dim, 
                         use_multi_head_predictor=True).to(device)
    else:
        print("Creating standard model with ImprovedNBCitationHead")
        model = ARITModel(config, input_dim=input_dim).to(device)
    
    optimizer = Adam(
        model.parameters(),
        lr=config.training_config["learning_rate"],
        weight_decay=config.training_config["weight_decay"]
    )
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=50,  # Total epochs
        eta_min=1e-7  # Minimum LR (not zero)
    )
    return model, optimizer, scheduler

def debug_citation_prediction(model, data, device, threshold=20, num_samples=10):
    """
    Debug helper function for citation prediction that doesn't modify any shared code.
    Shows clear comparisons of predictions vs targets in both log and original space.
    
    Args:
        model: The trained model to debug
        data: The data dictionary containing states, citation_targets, etc.
        device: The device to run inference on
        threshold: Citation threshold for high vs low analysis
        num_samples: Number of samples to analyze
    """
    model.eval()
    
    # Get sample papers
    val_state_ids = random.sample(data['val_states'], min(num_samples, len(data['val_states'])))
    
    # Create lists to store results
    results = []
    
    print("\n===== CITATION PREDICTION DEBUG =====")
    print(f"Analyzing {len(val_state_ids)} random validation samples\n")
    
    with torch.no_grad():
        for state_id in val_state_ids:
            state = data['states'][state_id]
            citation_target = data['citation_targets'][state_id]
            
            # Convert state to tensor
            state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            # Get model predictions
            _, _, citation_params, _ = model(state_tensor)
            
            # Get predictions for each horizon
            citation_preds = []
            log_citation_preds = []
            
            for h in range(model.horizons):
                _, mean = model.get_citation_distribution(citation_params, h)
                
                # Store log space value
                log_pred = mean.item()
                log_citation_preds.append(log_pred)
                
                # Convert to original space if needed
                if hasattr(model, 'use_log_transform') and model.use_log_transform:
                    orig_pred = math.exp(log_pred) - 1
                else:
                    orig_pred = log_pred
                
                citation_preds.append(orig_pred)
            
            # Convert targets to log space for comparison
            log_targets = [math.log(1 + t) for t in citation_target]
            
            # Record all the information
            results.append({
                'state_id': state_id,
                'citation_count': state.citation_count,
                'is_high_citation': state.citation_count > threshold,
                'orig_preds': citation_preds,
                'log_preds': log_citation_preds,
                'orig_targets': citation_target,
                'log_targets': log_targets
            })
    
    # Display results with clear formatting
    print(f"{'ID':<8} {'Count':<6} {'Space':<6} {'Pred H1':<10} {'Target H1':<10} {'Pred H2':<10} {'Target H2':<10}")
    print("-" * 70)
    
    for result in results:
        # Print original space comparison
        print(f"{result['state_id']:<8} {result['citation_count']:<6} {'orig':<6} " +
              f"{result['orig_preds'][0]:<10.2f} {result['orig_targets'][0]:<10.2f} " +
              f"{result['orig_preds'][1]:<10.2f} {result['orig_targets'][1]:<10.2f}")
        
        # Print log space comparison
        print(f"{'':<8} {'':<6} {'log':<6} " +
              f"{result['log_preds'][0]:<10.2f} {result['log_targets'][0]:<10.2f} " +
              f"{result['log_preds'][1]:<10.2f} {result['log_targets'][1]:<10.2f}")
        print("-" * 70)
    
    # Calculate and display average metrics
    high_papers = [r for r in results if r['is_high_citation']]
    low_papers = [r for r in results if not r['is_high_citation']]
    
    print("\nSUMMARY STATISTICS:")
    print(f"High citation papers (>{threshold}): {len(high_papers)}")
    print(f"Low citation papers (≤{threshold}): {len(low_papers)}")
    
    if high_papers:
        high_log_diff = [abs(p - t) for r in high_papers for p, t in zip(r['log_preds'], r['log_targets'])]
        high_orig_diff = [abs(p - t) for r in high_papers for p, t in zip(r['orig_preds'], r['orig_targets'])]
        print(f"High papers - Avg log space MAE: {sum(high_log_diff)/len(high_log_diff):.4f}")
        print(f"High papers - Avg orig space MAE: {sum(high_orig_diff)/len(high_orig_diff):.4f}")
    
    if low_papers:
        low_log_diff = [abs(p - t) for r in low_papers for p, t in zip(r['log_preds'], r['log_targets'])]
        low_orig_diff = [abs(p - t) for r in low_papers for p, t in zip(r['orig_preds'], r['orig_targets'])]
        print(f"Low papers - Avg log space MAE: {sum(low_log_diff)/len(low_log_diff):.4f}")
        print(f"Low papers - Avg orig space MAE: {sum(low_orig_diff)/len(low_orig_diff):.4f}")

def validate_with_citation_targets(model, val_states, citation_targets, field_targets, device, use_log_transform=True):
    model.eval()
    all_citation_preds = []
    all_citation_targets = []
    all_field_preds = []
    all_field_targets = []
    
    with torch.no_grad():
        num_predictions = 0
        sample_predictions = []
        sample_targets = []
        
        for state_id in citation_targets.keys():
            state = val_states[state_id]
            state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            _, _, citation_params, field_logits = model(state_tensor)
            
            citation_preds = []
            for h in range(model.horizons):
                nb_dist, mean = model.get_citation_distribution(citation_params, h, debug=False)  # Ensure debug=True
                if use_log_transform:
                    # Apply a log-space cap before exponentiating
                    LOG_SPACE_MAX = 6.2
                    capped_mean = min(mean.item(), LOG_SPACE_MAX)
                    pred_mean_raw = math.exp(capped_mean) - 1
                    # print(f"Before and after: {capped_mean}, {pred_mean_raw}")
                    # Debug transformation
                    # print(f"[validate_with_citation_targets] State {state_id}, Horizon {h} - "
                    #     f"Raw mean: {mean.item():.4f}, Capped: {capped_mean:.4f}, Transformed pred: {pred_mean_raw:.4f}")
                    if pred_mean_raw == float('inf') or pred_mean_raw != pred_mean_raw:
                        print(f"[validate_with_citation_targets] WARNING: Inf/NaN in transformed pred: {pred_mean_raw}")
                else:
                    pred_mean_raw = mean.item()
                
                pred_mean_raw = min(pred_mean_raw, 10000)
                citation_preds.append(pred_mean_raw)
            
            all_citation_preds.append(citation_preds)
            all_citation_targets.append(citation_targets[state_id])
            
            if len(sample_predictions) < 5 and state_id in citation_targets:
                sample_predictions.append(citation_preds)
                sample_targets.append(citation_targets[state_id])
                num_predictions += 1
                print(f"Validation sample {num_predictions}: Predictions={citation_preds}, Targets={citation_targets[state_id]}")
            
            field_pred = torch.argmax(field_logits, dim=1).cpu().item()
            all_field_preds.append(field_pred)
            all_field_targets.append(field_targets[state_id])
    
    citation_metrics = CitationMetrics().compute(torch.tensor(all_citation_preds), torch.tensor(all_citation_targets))
    field_acc = sum(p == t for p, t in zip(all_field_preds, all_field_targets)) / len(all_field_targets)
    return {**citation_metrics, 'field_accuracy': field_acc}, torch.tensor(all_citation_preds), torch.tensor(all_citation_targets)

# In fine_tune_citation_head in start_training.py
def fine_tune_citation_head(model, data, device, epochs=10):
    model.train()
    optimizer = torch.optim.Adam(
        [p for n, p in model.named_parameters() if 'citation_head' in n],
        lr=5e-5
    )
    criterion = nn.HuberLoss(delta=1.0)
    
    states = list(data['states'].values())
    citation_targets = data['citation_targets']
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(states), 32):
            batch_states = states[i:i+32]
            batch_ids = [s.state_id for s in batch_states]
            
            # Optimize tensor creation: Convert list of NumPy arrays to single NumPy array first
            state_array = np.array([s.to_numpy() for s in batch_states], dtype=np.float32)
            state_tensor = torch.from_numpy(state_array).to(device).unsqueeze(1)
            
            target_tensor = torch.tensor([citation_targets[sid] for sid in batch_ids], dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            _, _, citation_params, _ = model(state_tensor)
            
            log_target = torch.log1p(target_tensor)
            batch_loss = torch.tensor(0.0, device=device)
            for h in range(model.horizons):
                _, mean = model.get_citation_distribution(citation_params, h)
                loss_h = criterion(mean, log_target[:, h])
                batch_loss += loss_h
            
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        
        avg_loss = total_loss / (len(states) / 32)
        print(f"Fine-tune Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")



def main():
    """
    This script supports multiple flags to control which phases to run
    and which model(s) to use. The flags are:

    1) --pretrain-high        (just pretraining, high model only)
    2) --pretrain-low         (just pretraining, low model only)
    3) --pretrain-both        (just pretraining, both models)
    4) --rl-high              (just RL training, high model only)
    5) --rl-low               (just RL training, low model only)
    6) --rl-both              (just RL training, both models)
    7) --validate-high        (just validating, high model only)
    8) --validate-low         (just validating, low model only)
    9) --validate-both        (just validating, both models)
    10) --run-all-high        (run all phases with high model only)
    11) --run-all-low         (run all phases with low model only)
    12) --run-all-both        (run all phases with both models)
    13) --folder <foldername> (the folder name from which to load/save models)

    Examples:
      * python start_training.py --folder my_results --rl-high
      * python start_training.py --folder results_XXXX --validate-both
      * python start_training.py --folder results_XXXX --pretrain-both --rl-both
      * python start_training.py --run-all-both --folder results_XXXX
    """
    import numpy.core.multiarray
    import numpy
    add_safe_globals([numpy.core.multiarray.scalar, numpy.dtype, numpy.dtypes.Float64DType])
    
    # Parse command-line arguments
    args = sys.argv[1:]

    # Phase flags
    pretrain_high = "--pretrain-high" in args
    pretrain_low = "--pretrain-low" in args
    pretrain_both = "--pretrain-both" in args

    rl_high = "--rl-high" in args
    rl_low = "--rl-low" in args
    rl_both = "--rl-both" in args

    validate_high = "--validate-high" in args
    validate_low = "--validate-low" in args
    validate_both = "--validate-both" in args

    run_all_high = "--run-all-high" in args
    run_all_low = "--run-all-low" in args
    run_all_both = "--run-all-both" in args
    
    pretrain_moe = "--pretrain-moe" in args
    rl_moe = "--rl-moe" in args
    validate_moe = "--validate-moe" in args
    run_all_moe = "--run-all-moe" in args

    if "--folder" in args:
        idx = args.index("--folder")
        if idx + 1 < len(args):
            load_from_folder = args[idx + 1]
        else:
            print("Error: --folder flag provided but no folder name specified.")
            sys.exit(1)
    else:
        load_from_folder = None

    if run_all_high:
        pretrain_high = True
        rl_high = True
        validate_high = True
    if run_all_low:
        pretrain_low = True
        rl_low = True
        validate_low = True
    if run_all_both:
        pretrain_both = True
        rl_both = True
        validate_both = True
    if run_all_moe:
        pretrain_moe = True
        rl_moe = True
        validate_moe = True

    # Check if any relevant phase flag was provided
    if not any([
        pretrain_high, pretrain_low, pretrain_both, pretrain_moe,
        rl_high, rl_low, rl_both, rl_moe,
        validate_high, validate_low, validate_both, validate_moe,
        run_all_high, run_all_low, run_all_both, run_all_moe
    ]):
        print("Error: No phase flag provided (pretraining/rl/validation/run-all). Exiting.")
        sys.exit(1)

    # COMMON SETUP
    results_dir = os.path.join('./arit_data', f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created new results directory: {results_dir}")

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = ARITConfig()
    config = optimize_config_for_4060ti(config)
    data = load_arit_data(citation_cap=500)

    first_state = next(iter(data['states'].values()))
    state_dim = len(first_state.to_numpy())
    print(f"State dimension: {state_dim}")

    threshold = 20  # used for the dual-model approach

    low_model = None
    high_model = None
    low_config = copy.deepcopy(config)
    high_config = copy.deepcopy(config)
    low_pretrained = False
    high_pretrained = False
    moe_model = None
    moe_pretrained = False

    # UTILITY: Load existing models from a folder if we skip pretraining
    def try_load_low_model_if_none():
        """
        If 'low_model' is None and a folder is provided, try to load:
        1) final_low_citation_model.pt
        2) low_citation_model.pt
        If found, load it and set low_pretrained = True
        """
        nonlocal low_model, low_pretrained, low_config, threshold
        if load_from_folder and low_model is None:
            # Check final first
            low_model_path = os.path.join("./arit_data", load_from_folder, "final_low_citation_model.pt")
            if not os.path.exists(low_model_path):
                # fallback to normal
                low_model_path = os.path.join("./arit_data", load_from_folder, "low_citation_model.pt")

            if os.path.exists(low_model_path):
                print(f"Loading low-citation model from: {low_model_path}")
                import numpy.core.multiarray
                import numpy
                from torch.serialization import add_safe_globals
                add_safe_globals([numpy.core.multiarray.scalar, numpy.dtype, numpy.dtypes.Float64DType])
                checkpoint = torch.load(low_model_path, map_location=device)

                # If config is in checkpoint, use it
                if 'config' in checkpoint:
                    low_config = ARITConfig.from_dict(checkpoint['config'])

                # Create model with correct architecture (use_multi_head_predictor=True for low citation model)
                tmp_model, _, _ = setup_model_and_optimizer(low_config, state_dim, device, is_low_citation=True)
                tmp_model.citation_calibration = CitationCalibrationLayer(num_ranges=3).to(device)
                
                # Set use_log_transform attribute for correct citation prediction
                tmp_model.use_log_transform = True
                
                try:
                    # First try direct load
                    tmp_model.load_state_dict(checkpoint['model_state_dict'])
                except RuntimeError as e:
                    print(f"Warning: Direct state_dict loading failed. Attempting flexible loading: {str(e)}")
                    
                    # Flexible loading - only load matching keys
                    model_dict = tmp_model.state_dict()
                    pretrained_dict = checkpoint['model_state_dict']
                    
                    # Filter out mismatched keys
                    compatible_dict = {k: v for k, v in pretrained_dict.items() 
                                    if k in model_dict and model_dict[k].shape == v.shape}
                    
                    # Update the model with compatible weights
                    model_dict.update(compatible_dict)
                    tmp_model.load_state_dict(model_dict, strict=False)
                    
                    print(f"Loaded {len(compatible_dict)}/{len(pretrained_dict)} layers from checkpoint")
                
                low_model = tmp_model
                low_pretrained = True

                if 'threshold' in checkpoint:
                    threshold = checkpoint['threshold']
            else:
                print("No existing low-citation model found in folder. (Skipping load)")
            
    def try_load_high_model_if_none():
        """
        If 'high_model' is None and a folder is provided, try to load:
         1) final_high_citation_model.pt
         2) high_citation_model.pt
        If found, load it and set high_pretrained = True
        """
        nonlocal high_model, high_pretrained, high_config, threshold
        if load_from_folder and high_model is None:
            # Check final first
            high_model_path = os.path.join("./arit_data", load_from_folder, "final_high_citation_model.pt")
            if not os.path.exists(high_model_path):
                # fallback
                high_model_path = os.path.join("./arit_data", load_from_folder, "high_citation_model.pt")

            if os.path.exists(high_model_path):
                print(f"Loading high-citation model from: {high_model_path}")
                import numpy.core.multiarray
                import numpy
                from torch.serialization import add_safe_globals
                add_safe_globals([numpy.core.multiarray.scalar, numpy.dtype, numpy.dtypes.Float64DType])
                checkpoint = torch.load(high_model_path, map_location=device)

                if 'config' in checkpoint:
                    high_config = ARITConfig.from_dict(checkpoint['config'])

                tmp_model, _, _ = setup_model_and_optimizer(high_config, state_dim, device)
                tmp_model.load_state_dict(checkpoint['model_state_dict'])
                high_model = tmp_model
                high_pretrained = True

                if 'threshold' in checkpoint:
                    threshold = checkpoint['threshold']
            else:
                print("No existing high-citation model found in folder. (Skipping load)")

    def try_load_moe_model_if_none():
        """
        If 'moe_model' is None and a folder is provided, try to load:
        1) final_moe_citation_model.pt
        2) moe_citation_model.pt
        If found, load it and set moe_pretrained = True
        Otherwise, do nothing.
        """
        nonlocal moe_model, moe_pretrained, config
        
        if load_from_folder and moe_model is None:
            # Check final first
            moe_model_path = os.path.join("./arit_data", load_from_folder, "final_moe_citation_model.pt")
            if not os.path.exists(moe_model_path):
                # fallback to normal
                moe_model_path = os.path.join("./arit_data", load_from_folder, "moe_citation_model.pt")
            
            if os.path.exists(moe_model_path):
                print(f"Loading MoE model from: {moe_model_path}")
                checkpoint = torch.load(moe_model_path, map_location=device)
                
                # If config is in checkpoint, use it
                if 'config' in checkpoint:
                    moe_config = ARITConfig.from_dict(checkpoint['config'])
                else:
                    moe_config = copy.deepcopy(config)
                
                # Get number of experts from checkpoint, default to 3
                num_experts = checkpoint.get('num_experts', 3)
                
                # Create model with MoE architecture
                tmp_model, _, _ = setup_model_and_optimizer(
                    moe_config, state_dim, device, use_moe=True, num_experts=num_experts
                )
                
                # Set use_log_transform attribute for correct citation prediction
                tmp_model.use_log_transform = True
                
                try:
                    # First try direct load
                    tmp_model.load_state_dict(checkpoint['model_state_dict'])
                except RuntimeError as e:
                    print(f"Warning: Direct state_dict loading failed. Attempting flexible loading: {str(e)}")
                    
                    # Flexible loading - only load matching keys
                    model_dict = tmp_model.state_dict()
                    pretrained_dict = checkpoint['model_state_dict']
                    
                    # Filter out mismatched keys
                    compatible_dict = {k: v for k, v in pretrained_dict.items() 
                                    if k in model_dict and model_dict[k].shape == v.shape}
                    
                    # Update the model with compatible weights
                    model_dict.update(compatible_dict)
                    tmp_model.load_state_dict(model_dict, strict=False)
                    
                    print(f"Loaded {len(compatible_dict)}/{len(pretrained_dict)} layers from checkpoint")
                
                moe_model = tmp_model
                moe_pretrained = True
            else:
                print("No existing MoE model found in folder. (Skipping load)")

    any_pretraining_requested = (pretrain_low or pretrain_high or pretrain_both)
    if load_from_folder and not any_pretraining_requested:
        if rl_low or validate_low or rl_both or validate_both:
            try_load_low_model_if_none()
        if rl_high or validate_high or rl_both or validate_both:
            try_load_high_model_if_none() 
        if rl_moe or validate_moe:
            try_load_moe_model_if_none()

    # PRETRAINING PHASE
    from arit_citations import pretrain_citation_model

    def pretrain_dual_models():
        nonlocal low_model, high_model, threshold, low_config, high_config
        nonlocal low_pretrained, high_pretrained

        print("\n======== PRETRAINING DUAL MODELS (LOW & HIGH) ========\n")

        low_data, high_data = split_data_by_citation_range(data, threshold, min_citations=1)

        # If a folder is specified, attempt to load existing models first
        if load_from_folder:
            try_load_low_model_if_none()
            try_load_high_model_if_none()

        # If either model is still None, train new models
        if (low_model is None) or (high_model is None):
            print("Training new dual models (low, high) because one or both are None.")
            low_model, high_model = train_dual_models(config, data, device, results_dir)
            low_pretrained = True
            high_pretrained = True
            threshold = 20

        # Validation after pretraining
        print("\n====== Validating the dual models after pretraining ======\n")
        low_val_states_dict = {k: low_data['states'][k] for k in low_data['val_states']}
        low_val_citation_targets = {k: v for k, v in low_data['citation_targets'].items() if k in low_data['val_states']}
        low_val_field_targets = {k: v for k, v in low_data['field_targets'].items() if k in low_data['val_states']}

        high_val_states_dict = {k: high_data['states'][k] for k in high_data['val_states']}
        high_val_citation_targets = {k: v for k, v in high_data['citation_targets'].items() if k in high_data['val_states']}
        high_val_field_targets = {k: v for k, v in high_data['field_targets'].items() if k in high_data['val_states']}

        print("Validating Low-Citation Model (≤20):")
        low_val_metrics, low_range_metrics = validate_dual_models(
            low_model, high_model, low_val_states_dict,
            low_val_citation_targets, low_val_field_targets,
            device, threshold
        )
        print("\nLow-citation model validation metrics:")
        for key, value in low_val_metrics.items():
            print(f"  {key}: {value:.4f}")
        print("\nMetrics by citation range (low model):")
        for range_name, metrics in low_range_metrics.items():
            print(f"  Range {range_name}: (Count: {metrics.get('count', 0)})")
            for key, value in metrics.items():
                if key != 'count':
                    print(f"    {key}: {value:.4f}")

        print("\nValidating High-Citation Model (>20):")
        high_val_metrics, high_range_metrics = validate_dual_models(
            low_model, high_model, high_val_states_dict,
            high_val_citation_targets, high_val_field_targets,
            device, threshold
        )
        print("\nHigh-citation model validation metrics:")
        for key, value in high_val_metrics.items():
            print(f"  {key}: {value:.4f}")
        print("\nMetrics by citation range (high model):")
        for range_name, metrics in high_range_metrics.items():
            print(f"  Range {range_name}: (Count: {metrics.get('count', 0)})")
            for key, value in metrics.items():
                if key != 'count':
                    print(f"    {key}: {value:.4f}")

        # Save final
        torch.save({
            'model_state_dict': low_model.state_dict(),
            'threshold': threshold,
            'config': low_config.to_dict(),
            'metrics': low_val_metrics,
            'date_trained': datetime.now().isoformat()
        }, os.path.join(results_dir, 'final_low_citation_model.pt'))

        torch.save({
            'model_state_dict': high_model.state_dict(),
            'threshold': threshold,
            'config': high_config.to_dict(),
            'metrics': high_val_metrics,
            'date_trained': datetime.now().isoformat()
        }, os.path.join(results_dir, 'final_high_citation_model.pt'))

    def pretrain_single_model(which="low"):
        """
        Pretrain a single model (low or high) with data split at threshold=20.
        If which="low", trains on papers with ≤20 citations.
        If which="high", trains on papers with >20 citations.
        With separate configurations for each model type.
        """
        nonlocal low_model, high_model, low_pretrained, high_pretrained
        
        # Split data into low and high citation datasets
        threshold = 20
        low_data, high_data = split_data_by_citation_range(data, threshold, min_citations=1)
        
        # Print data statistics
        print(f"Original dataset: {len(data['states'])} papers")
        print(f"Low-citation dataset (≤{threshold}): {len(low_data['states'])} papers")
        print(f"High-citation dataset (>{threshold}): {len(high_data['states'])} papers")
        
        if which == "low":
            # ============ LOW CITATION MODEL PRETRAINING ============
            print("\n======== PRETRAINING LOW MODEL (≤20 CITATIONS) ========\n")
            if low_model is not None:
                print("Low model is already loaded. Skipping re-training.")
                return
            
            # Configure specific settings for low citation model
            config_to_use = low_config
            model_type_str = "LOW citation (≤20)"
            training_data = low_data
            
            print(f"Filtered to {len(training_data['train_states'])} training states and {len(training_data['val_states'])} validation states")
            
            # Create model with appropriate settings for low citation
            model, _, _ = setup_model_and_optimizer(
                config_to_use, 
                state_dim, 
                device, 
                is_low_citation=True  # Use multi-head predictor for low citation
            )
            
            # Add citation calibration layer specifically for low model
            model.citation_calibration = CitationCalibrationLayer(num_ranges=3).to(device)
            
            # Low model specific training parameters
            pretrain_epochs = 15
            pretrain_batch_size = 64
            pretrain_lr = 1e-5  # Reduced learning rate for stability (from 3e-5)
            
            # Create optimizer with enhanced regularization for low model
            pretrain_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=pretrain_lr,
                weight_decay=config_to_use.training_config["weight_decay"] * 4.0  # Double weight decay for regularization
            )
            
            # Use custom loss with enhanced weighting for problematic ranges
            custom_loss = EnhancedRangeAwareCitationLoss(
                range_weights={
                    (0, 5): 2.0,
                    (6, 10): 1.0,
                    (11, 20): 1.5
                }
            )
            
        else:  # which == "high"
            # ============ HIGH CITATION MODEL PRETRAINING ============
            print("\n======== PRETRAINING HIGH MODEL (>20 CITATIONS) ========\n")
            if high_model is not None:
                print("High model is already loaded. Skipping re-training.")
                return
            
            # Configure specific settings for high citation model
            config_to_use = high_config
            model_type_str = "HIGH citation (>20)"
            training_data = high_data
            
            print(f"Filtered to {len(training_data['train_states'])} training states and {len(training_data['val_states'])} validation states")
            
            # Create model for high citation (standard setup)
            model, _, _ = setup_model_and_optimizer(
                config_to_use, 
                state_dim, 
                device, 
                is_low_citation=False  # Use standard predictor for high citation
            )
            
            # High model specific training parameters
            pretrain_epochs = 15
            pretrain_batch_size = 64
            pretrain_lr = 3e-5
            
            # Create optimizer with standard regularization for high model
            pretrain_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=pretrain_lr,
                weight_decay=config_to_use.training_config["weight_decay"]
            )
            
            # No custom loss for high model
            custom_loss = None
        
        # Common setup after model-specific configuration
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {param_count:,} trainable parameters")
        
        # Create scheduler
        pretrain_scheduler = CosineAnnealingLR(
            pretrain_optimizer, 
            T_max=pretrain_epochs, 
            eta_min=1e-6
        )
        
        print(f"\n======= STARTING CITATION AND FIELD PRE-TRAINING ({model_type_str.upper()}) =======\n")
        print(f"Training {model_type_str}")
        print(f"Model type: {model_type_str.split()[0]} citation")
        
        # Train the model
        if which == "low":
            # Define a custom function to apply after each batch
            def after_batch_hook(model, optimizer):
                # Apply gradient clipping specifically for low model
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        else:
            # No special processing for high model
            after_batch_hook = None
        
        # Same pretrain function for both models but with custom hook
        pretrain_metrics, best_pretrained_state = pretrain_citation_model(
            model=model,
            config=config_to_use,
            data=training_data,
            optimizer=pretrain_optimizer,
            device=device,
            results_dir=results_dir,
            epochs=pretrain_epochs,
            batch_size=pretrain_batch_size,
            scaler=None,
            use_log_transform=True,
            custom_loss=custom_loss,
            which=which,
            after_batch_hook=after_batch_hook
        )
        
        # Save the model
        model_type = "low" if which == "low" else "high"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config_to_use.to_dict(),
            'metrics': pretrain_metrics,
            'date_trained': datetime.now().isoformat(),
            'threshold': threshold
        }, os.path.join(results_dir, f'final_{model_type}_citation_model.pt'))
        
        # Assign to the appropriate global variable
        if which == "low":
            low_model = model
            low_pretrained = True
        else:
            high_model = model
            high_pretrained = True
        
        # Validation on the filtered subset
        val_state_ids = training_data['val_states']
        val_states_filtered = {k: training_data['states'][k] for k in val_state_ids}
        val_citation_targets = {k: v for k, v in training_data['citation_targets'].items() if k in val_state_ids}
        val_field_targets = {k: v for k, v in training_data['field_targets'].items() if k in val_state_ids}
        
        # Additional check to ensure citation range filtering is applied correctly
        val_states_filtered = {
            k: v for k, v in val_states_filtered.items() 
            if (v.citation_count <= threshold if which == "low" else v.citation_count > threshold)
        }
        val_citation_targets = {k: v for k, v in val_citation_targets.items() if k in val_states_filtered}
        val_field_targets = {k: v for k, v in val_field_targets.items() if k in val_states_filtered}
        
        # Run validation
        val_metrics, predictions, targets = validate_with_citation_targets(
            model=model,
            val_states=val_states_filtered,
            citation_targets=val_citation_targets,
            field_targets=val_field_targets,
            device=device
        )
        
        debug_citation_prediction(model, data, device, threshold=20, num_samples=5)
        
        print(f"\n{model_type.capitalize()}-citation model validation metrics after pretraining:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    def pretrain_moe_model():
        """
        Train a single Mixture of Experts model using the entire dataset.
        """
        nonlocal moe_model, moe_pretrained
        
        print("\n======== PRETRAINING MIXTURE OF EXPERTS MODEL ========\n")
        
        # Configure specific settings for MoE model
        moe_config = copy.deepcopy(config)
        moe_config.model_config["num_layers"] = 4
        moe_config.model_config["d_model"] = 512
        moe_config.model_config["dropout"] = 0.15
        
        # Create the MoE model
        if moe_model is not None:
            print("MoE model is already loaded. Skipping re-training.")
            return
        
        # Number of experts
        num_experts = 3
        
        model, optimizer, scheduler = setup_model_and_optimizer(
            moe_config, 
            state_dim, 
            device, 
            use_moe=True,
            num_experts=num_experts
        )
        
        # Add the CitationScalingLayer to the model
        model.citation_scaling = CitationScalingLayer(num_ranges=5).to(device)
        
        # Override the get_citation_distribution method to include the scaling layer
        original_get_citation_distribution = model.get_citation_distribution
        
        def new_get_citation_distribution(self, citation_params, horizon_idx, debug=False):
            # Get the original distribution and mean
            nb_dist, mean = original_get_citation_distribution(citation_params, horizon_idx, debug)
            
            # Apply the citation scaling layer
            citation_counts = getattr(self, '_temp_citation_counts', None)
            
            # Debug output
            if debug:
                print("\n----- DEBUG: CitationScalingLayer -----")
                print(f"  Mean before scaling: {mean.shape}")
                if mean.shape[0] == 1:
                    print(f"  Value before scaling: {mean.item():.4f}")
                else:
                    print(f"  Values before scaling: min={mean.min().item():.4f}, max={mean.max().item():.4f}")
            
            # Apply scaling
            if mean.dim() == 1:
                mean_reshaped = mean.unsqueeze(1)
                scaled_mean = self.citation_scaling(mean_reshaped, citation_counts).squeeze(1)
            else:
                scaled_mean = self.citation_scaling(mean, citation_counts)
            
            # Debug output for after scaling
            if debug:
                if scaled_mean.shape[0] == 1:
                    print(f"  Value after scaling: {scaled_mean.item():.4f}")
                else:
                    print(f"  Values after scaling: min={scaled_mean.min().item():.4f}, max={scaled_mean.max().item():.4f}")
                print("-----------------------------------------")
            
            return nb_dist, scaled_mean
        
        # Replace the method
        import types
        model.get_citation_distribution = types.MethodType(new_get_citation_distribution, model)
        
        original_forward = model.forward
        
        def forward_with_citation_counts(self, x, citation_count=None):
            # Store citation_count for use in get_citation_distribution
            self._temp_citation_counts = citation_count
            
            # Call the original forward method
            return original_forward(x, citation_count=citation_count)
        
        model.forward = types.MethodType(forward_with_citation_counts, model)
        
        # Set up results directory
        moe_results_dir = os.path.join(results_dir, 'moe_model')
        os.makedirs(moe_results_dir, exist_ok=True)
        
        # MoE training parameters
        pretrain_epochs = 40
        pretrain_batch_size = 64
        pretrain_lr = 2e-5
        
        # Update optimizer with appropriate learning rate
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=pretrain_lr,
            weight_decay=moe_config.training_config["weight_decay"]
        )
        
        pretrain_metrics, best_pretrained_state = pretrain_moe_citation_model(
            model=model,
            config=moe_config,
            data=data,
            optimizer=optimizer,
            device=device,
            results_dir=moe_results_dir,
            epochs=pretrain_epochs,
            batch_size=pretrain_batch_size,
            scaler=None,
            use_log_transform=True
        )
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': moe_config.to_dict(),
            'metrics': pretrain_metrics,
            'date_trained': datetime.now().isoformat(),
            'num_experts': num_experts
        }, os.path.join(results_dir, 'moe_citation_model.pt'))
        
        # Analyze the expert utilization
        expert_weights = analyze_moe_experts(model, data, device, moe_results_dir)
        
        # Store model for later use
        moe_model = model
        moe_pretrained = True
        
        # Run validation
        val_state_ids = data['val_states']
        val_states_filtered = {k: data['states'][k] for k in val_state_ids}
        val_citation_targets = {k: v for k, v in data['citation_targets'].items() if k in val_state_ids}
        val_field_targets = {k: v for k, v in data['field_targets'].items() if k in val_state_ids}
        
        val_metrics, predictions, targets = validate_with_citation_targets(
            model=model,
            val_states=val_states_filtered,
            citation_targets=val_citation_targets,
            field_targets=val_field_targets,
            device=device
        )
        
        print("\nMoE model validation metrics after pretraining:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        return val_metrics

    # Actually run pretraining if requested
    if pretrain_low:
        # Try to load from folder if it exists
        try_load_low_model_if_none()
        pretrain_single_model("low")

    if pretrain_high:
        try_load_high_model_if_none()
        pretrain_single_model("high")

    if pretrain_both:
        pretrain_dual_models()

    # ------------------------------------------------------------------
    # RL TRAINING PHASE
    # ------------------------------------------------------------------
    def run_rl_for_low_model():
        print("\n======= STARTING RL TRAINING WITH LOW-CITATION MODEL =======\n")
        # Try to load if still None
        try_load_low_model_if_none()
        if low_model is None:
            print("Error: Attempting to RL-train low model, but it is None. Make sure it's pretrained or loaded.")
            return
        
        def diagnose_citation_head():
            print("\n=== Diagnosing MultiHeadCitationPredictor ===")
            
            # Check if the citation head is a MultiHeadCitationPredictor
            has_multi_head = hasattr(low_model, 'citation_head') and isinstance(low_model.citation_head, MultiHeadCitationPredictor)
            print(f"Model has MultiHeadCitationPredictor: {has_multi_head}")
            
            if has_multi_head:
                # Get a sample state for testing
                state_id = next(iter(data['val_states']))
                state = data['states'][state_id]
                print(f"Testing with state ID {state_id}, citation count {state.citation_count}")
                
                state_np = state.to_numpy()
                state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    # Forward pass through the model
                    _, _, citation_params, _ = low_model(state_tensor)
                    
                    print(f"Citation params type: {type(citation_params)}")
                    if isinstance(citation_params, tuple):
                        params, range_probs = citation_params
                        print(f"Range probabilities shape: {range_probs.shape}")
                        print(f"Range probabilities: {range_probs.cpu().numpy()}")
                        selected_range = torch.argmax(range_probs, dim=1).item()
                        print(f"Selected range: {selected_range}")
                    
                    # Try to make predictions directly
                    predictions = []
                    for h in range(low_model.horizons):
                        try:
                            if isinstance(citation_params, tuple):
                                # Try get_distribution_for_horizon if it exists
                                if hasattr(low_model.citation_head, 'get_distribution_for_horizon'):
                                    nb_dist, mean = low_model.citation_head.get_distribution_for_horizon(params, h)
                                    raw_pred = mean.cpu().item()
                                    print(f"Horizon {h} - Raw mean: {raw_pred}")
                                    
                                    exp_pred = torch.exp(mean).cpu().item() - 1
                                    print(f"Horizon {h} - exp(mean)-1: {exp_pred}")
                                    
                                    pred = exp_pred if low_model.use_log_transform else raw_pred
                                    predictions.append(pred)
                                else:
                                    print(f"get_distribution_for_horizon not found")
                            else:
                                # Standard citation prediction
                                nb_dist, mean = low_model.get_citation_distribution(citation_params, h)
                                pred = torch.exp(mean).cpu().item() - 1 if low_model.use_log_transform else mean.cpu().item()
                                predictions.append(pred)
                        except Exception as e:
                            print(f"Error predicting for horizon {h}: {str(e)}")
                    
                    print(f"Predictions: {predictions}")
                    print(f"Targets: {data['citation_targets'][state_id]}")
            
            print("=== Diagnosis Complete ===\n")

        diagnose_citation_head()
        # Check if we have a MultiHeadCitationPredictor
        has_multi_head = hasattr(low_model, 'citation_head') and isinstance(low_model.citation_head, MultiHeadCitationPredictor)
        print(f"Using MultiHeadCitationPredictor: {has_multi_head}")

        environment = setup_environment(config, data)
        if low_pretrained:
            print("Freezing citation prediction components in low model (since it's pre-trained).")
            for name, param in low_model.named_parameters():
                if 'citation_head' in name or 'citation_calibration' in name:
                    param.requires_grad = False

        # Use lower learning rate for low citation model
        low_lr = config.training_config["learning_rate"] / 200
        
        # Use different optimizer settings for low citation model
        low_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, low_model.parameters()),
            lr=low_lr,
            weight_decay=config.training_config["weight_decay"] * 2
        )
        
        # Use cosine annealing for low model
        low_scheduler = CosineAnnealingLR(
            low_optimizer,
            T_max=config.training_config["num_epochs"],
            eta_min=config.training_config["learning_rate"] / 1000
        )

        # Adjust configuration for low citation model
        low_rl_config = copy.deepcopy(config)
        low_rl_config.training_config["citation_weight"] = 0.02
        low_rl_config.training_config["clip_grad_norm"] = 0.25
        low_rl_config.training_config["policy_weight"] = 1.5
        low_rl_config.training_config["ppo_epochs"] = 3

        # Define custom validation function for low citation model
        def validate_low_citation_model(model, val_states, citation_targets, field_targets, device, threshold=20):
            print("\n=== Running low citation model validation with detailed debugging ===")
            model.eval()
            all_citation_preds = []
            all_citation_targets = []
            all_field_preds = []
            all_field_targets = []
            total_papers = len(citation_targets)
            filtered_papers = 0
            # Check model properties
            has_multi_head = hasattr(model, 'citation_head') and isinstance(model.citation_head, MultiHeadCitationPredictor)
            print(f"Model has MultiHeadCitationPredictor: {has_multi_head}")
            if has_multi_head:
                print(f"Citation head type: {type(model.citation_head).__name__}")
                print(f"Citation head attributes: {dir(model.citation_head)}")
            
            # Track detailed stats for debugging
            range_selections = []
            raw_outputs = []
            
            # Process only a subset of papers for detailed debugging
            debug_sample_size = 10
            debug_counter = 0
            
            with torch.no_grad():
                for state_id in citation_targets.keys():
                    state = val_states[state_id]
                    
                    if state.citation_count > threshold or max(citation_targets[state_id]) > threshold:
                        continue
                    filtered_papers += 1
                    
                    # Detailed debugging for a few samples
                    is_debug_sample = debug_counter < debug_sample_size
                    if is_debug_sample:
                        print(f"\n--- DEBUG SAMPLE {debug_counter+1} ---")
                        print(f"Paper ID: {state_id}")
                        print(f"Actual citation count: {state.citation_count}")
                        print(f"Future citations target: {citation_targets[state_id]}")
                    
                    state_np = state.to_numpy()
                    state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    
                    # Forward pass
                    _, _, citation_params, field_logits = model(state_tensor)
                    
                    # Citation predictions - handle MultiHeadCitationPredictor
                    citation_preds = []
                    
                    if has_multi_head:
                        if is_debug_sample:
                            print("Using MultiHeadCitationPredictor for prediction")
                            # Check the type and shape of citation_params
                            print(f"Citation params type: {type(citation_params)}")
                            if isinstance(citation_params, tuple):
                                print(f"Citation params is a tuple of length {len(citation_params)}")
                                print(f"  First element shape: {citation_params[0].shape}")
                                print(f"  Second element shape: {citation_params[1].shape}")
                                
                                # Print range probabilities for debugging
                                range_probs = citation_params[1]
                                print(f"  Range probabilities: {range_probs.cpu().numpy()}")
                                range_selections.append(torch.argmax(range_probs, dim=1).item())
                            else:
                                print(f"Citation params shape: {citation_params.shape}")
                        
                        # Extract parameters and range probabilities
                        if isinstance(citation_params, tuple) and len(citation_params) == 2:
                            params, range_probs = citation_params
                            
                            # Track selected range
                            selected_range = torch.argmax(range_probs, dim=1).item()
                            if is_debug_sample:
                                print(f"  Selected range index: {selected_range}")
                        else:
                            # Handle unexpected format
                            if is_debug_sample:
                                print("  Warning: Unexpected citation_params format")
                            params = citation_params
                        
                        try:
                            # Try using the model's method
                            if is_debug_sample:
                                print("  Attempting to use model.citation_head.get_distribution_for_horizon")
                                
                            for h in range(model.horizons):
                                # Try to get distribution from multi-head predictor
                                try:
                                    nb_dist, mean = model.citation_head.get_distribution_for_horizon(params, h)
                                    
                                    if is_debug_sample:
                                        print(f"  Horizon {h} - Raw mean: {mean.cpu().item()}")
                                        if hasattr(model, 'use_log_transform'):
                                            print(f"  Model has use_log_transform={model.use_log_transform}")
                                    
                                    # Convert from log space if needed
                                    if hasattr(model, 'use_log_transform') and model.use_log_transform:
                                        pred_mean_raw = torch.exp(mean).cpu().item() - 1
                                        if is_debug_sample:
                                            print(f"  Horizon {h} - After log transform: {pred_mean_raw}")
                                    else:
                                        pred_mean_raw = mean.cpu().item()
                                        
                                    # Cap at threshold for low model
                                    pred_mean_raw = min(pred_mean_raw, 20.0)
                                    citation_preds.append(pred_mean_raw)
                                    
                                    if is_debug_sample:
                                        print(f"  Horizon {h} - Final prediction: {pred_mean_raw}")
                                        
                                except Exception as e:
                                    print(f"  Error in get_distribution_for_horizon: {str(e)}")
                                    # Fallback to direct access
                                    citation_preds.append(0.0)
                        except Exception as e:
                            print(f"Error processing citations: {str(e)}")
                            # Add fallback predictions
                            citation_preds = [0.0] * model.horizons
                    else:
                        # Standard handling
                        if is_debug_sample:
                            print("Using standard citation prediction")
                            
                        for h in range(model.horizons):
                            try:
                                nb_dist, mean = model.get_citation_distribution(citation_params, h)
                                # Convert from log space if needed
                                if hasattr(model, 'use_log_transform') and model.use_log_transform:
                                    pred_mean_raw = torch.exp(mean).cpu().item() - 1
                                else:
                                    pred_mean_raw = mean.cpu().item()
                                    
                                pred_mean_raw = min(pred_mean_raw, 20.0)  # Cap at threshold for low model
                                citation_preds.append(pred_mean_raw)
                                
                                if is_debug_sample:
                                    print(f"  Horizon {h} - Final prediction: {pred_mean_raw}")
                            except Exception as e:
                                print(f"  Error in get_citation_distribution: {str(e)}")
                                citation_preds.append(0.0)
                    
                    # Store predictions and targets
                    all_citation_preds.append(citation_preds)
                    all_citation_targets.append(citation_targets[state_id])
                    
                    # Field predictions
                    field_pred = torch.argmax(field_logits, dim=1).cpu().item()
                    all_field_preds.append(field_pred)
                    all_field_targets.append(field_targets[state_id])
                    
                    # Increment debug counter
                    if is_debug_sample:
                        debug_counter += 1
                        raw_outputs.append({
                            'state_id': state_id,
                            'citation_count': state.citation_count,
                            'citation_preds': citation_preds,
                            'citation_targets': citation_targets[state_id]
                        })
            
            if len(all_citation_preds) == 0:
                print("Warning: No low citation papers found in validation set!")
                return {"mae": float('nan'), "rmse": float('nan'), "r2": float('nan')}, None, None
            
            # Print range selection statistics if available
            if range_selections:
                from collections import Counter
                range_counts = Counter(range_selections)
                print("\nRange selection distribution:")
                for range_idx, count in range_counts.items():
                    print(f"  Range {range_idx}: {count} samples ({count/len(range_selections)*100:.1f}%)")
            
            # Compute metrics
            citation_metrics = CitationMetrics().compute(torch.tensor(all_citation_preds), torch.tensor(all_citation_targets))
            field_acc = (sum(p == t for p, t in zip(all_field_preds, all_field_targets)) / len(all_field_targets) 
                        if all_field_targets else 0)
            metrics = {**citation_metrics, 'field_accuracy': field_acc}
            
            # Debug output
            print("\n---- Citation Validation Debug Summary ----")
            print(f"\nFiltered validation set: {filtered_papers}/{total_papers} papers ({filtered_papers/total_papers*100:.1f}%) with citations ≤ {threshold}")
            print(f"Number of predictions: {len(all_citation_preds)}")
            all_preds_flat = [p for sublist in all_citation_preds for p in sublist]
            print(f"Prediction stats: min={min(all_preds_flat) if all_preds_flat else 0:.4f}, " + 
                f"max={max(all_preds_flat) if all_preds_flat else 0:.4f}, " + 
                f"mean={sum(all_preds_flat)/len(all_preds_flat) if all_preds_flat else 0:.4f}")
            
            print("\nSample predictions vs targets:")
            for i in range(min(5, len(all_citation_preds))):
                print(f"  Sample {i+1}:")
                print(f"    Predictions: {[f'{p:.4f}' for p in all_citation_preds[i]]}")
                print(f"    Targets: {all_citation_targets[i]}")
            
            print(f"\nFinal metrics - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, Field Acc: {metrics['field_accuracy']:.4f}")
            
            return metrics, torch.tensor(all_citation_preds), torch.tensor(all_citation_targets)
                
        # Setup custom trainer for low citation model
        class LowCitationARITTrainer(ARITTrainer):
            def validate_epoch(self, epoch):
                """Override to use our custom validation function"""
                val_metrics, _, _ = validate_low_citation_model(
                    self.model, 
                    self.val_states,
                    self.val_citation_targets,
                    self.val_field_targets,
                    self.device, 
                    threshold=threshold
                )
                
                # Log metrics
                val_log = {f"val_{k}": v for k, v in val_metrics.items()}
                self.metrics["validation"].append({
                    "epoch": epoch,
                    **val_log
                })
                
                return val_metrics
            def validate_citation(self):
                """Override the parent method to filter for low citation papers"""
                self.model.eval()
                all_predictions = []
                all_targets = []
                all_field_preds = []
                all_field_targets = []
                
                # Debugging variables
                pred_magnitudes = []
                
                # Filtering threshold
                threshold = 20  # Use the same threshold for filtering
                
                # Track filtering statistics
                total_papers = len(self.val_citation_targets)
                filtered_papers = 0
                
                # Batch processing
                batch_size = 32
                state_ids = list(self.val_citation_targets.keys())
                
                with torch.no_grad():
                    for i in range(0, len(state_ids), batch_size):
                        batch_ids = state_ids[i:i + batch_size]
                        
                        # Filter batch_ids to only include low citation papers
                        filtered_batch_ids = []
                        for sid in batch_ids:
                            state = self.val_states[sid]
                            # Only include papers where both current and future citations are below threshold
                            if state.citation_count <= threshold and max(self.val_citation_targets[sid]) <= threshold:
                                filtered_batch_ids.append(sid)
                                filtered_papers += 1
                        
                        if not filtered_batch_ids:
                            continue  # Skip this batch if no papers match our criteria
                        
                        states = [self.val_states[sid].to_numpy() for sid in filtered_batch_ids]
                        state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device).unsqueeze(1)
                        _, _, citation_params, field_logits = self.model(state_tensor)
                        
                        batch_predictions = []
                        for h in range(self.model.horizons):
                            nb_dist, mean = self.model.get_citation_distribution(citation_params, h)
                            batch_predictions.append(mean)
                        
                        # Stack predictions [batch_size, horizons]
                        batch_predictions = torch.stack(batch_predictions, dim=1)  # [batch_size, horizons]
                        
                        # Transform predictions if using log
                        if self.use_log_transform:
                            batch_predictions = torch.expm1(batch_predictions)  # Convert back to original space
                        
                        # Collect predictions and targets
                        for j, sid in enumerate(filtered_batch_ids):
                            preds = batch_predictions[j].cpu().tolist()
                            all_predictions.append(preds)
                            pred_magnitudes.extend(preds)
                            
                            targets = self.val_citation_targets[sid]
                            all_targets.append(targets)
                            
                            field_pred = torch.argmax(field_logits[j], dim=0).cpu().item()
                            all_field_preds.append(field_pred)
                            all_field_targets.append(self.val_field_targets[sid])
                
                # Debugging summary
                print("\n---- Citation Validation Debug Summary ----")
                print(f"Filtered validation set: {filtered_papers}/{total_papers} papers ({filtered_papers/total_papers*100:.1f}%) with citations ≤ {threshold}")
                print(f"Number of predictions: {len(all_predictions)}")
                print(f"Prediction stats: min={min(pred_magnitudes) if pred_magnitudes else 0:.4f}, max={max(pred_magnitudes) if pred_magnitudes else 0:.4f}, "
                    f"mean={sum(pred_magnitudes)/len(pred_magnitudes) if pred_magnitudes else 0:.4f}")
                
                # Sample predictions vs targets
                print("\nSample predictions vs targets:")
                for i in range(min(5, len(all_predictions))):
                    print(f"  Sample {i+1}:")
                    print(f"    Predictions: {[round(p, 4) for p in all_predictions[i]]}")
                    print(f"    Targets: {all_targets[i]}")
                
                # Check if we have any valid predictions
                if not all_predictions:
                    print("Warning: No low citation papers found in validation set!")
                    return float('nan'), float('nan'), 0.0
                    
                # Compute metrics
                predictions_tensor = torch.tensor(all_predictions, device=self.device)
                targets_tensor = torch.tensor(all_targets, device=self.device)
                
                diff = predictions_tensor - targets_tensor
                mae = torch.mean(torch.abs(diff)).item()
                rmse = torch.sqrt(torch.mean(diff**2)).item()
                field_acc = sum(p == t for p, t in zip(all_field_preds, all_field_targets)) / len(all_field_targets) if all_field_targets else 0.0
                
                print(f"Final metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, Field Acc: {field_acc:.4f}")
                return mae, rmse, field_acc

        # Setup trainer with our custom subclass
        if has_multi_head:
            trainer_class = LowCitationARITTrainer
        else:
            trainer_class = ARITTrainer
            
        low_trainer = trainer_class(
            model=low_model,
            config=low_rl_config,
            env=environment,
            optimizer=low_optimizer,
            scheduler=low_scheduler,
            device=device,
            scaler=None,
            citation_targets=data['citation_targets'],
            field_targets=data['field_targets'],
            val_states=data['states'],
            val_citation_targets={k: v for k, v in data['citation_targets'].items() if k in data['val_states']},
            val_field_targets={k: v for k, v in data['field_targets'].items() if k in data['val_states']},
            use_log_transform=True
        )
        
        # Perform RL training
        low_training_metrics = low_trainer.train()

        # Save results
        rl_results_dir = os.path.join(results_dir, 'rl_training')
        os.makedirs(rl_results_dir, exist_ok=True)
        low_rl_dir = os.path.join(rl_results_dir, 'low_model')
        os.makedirs(low_rl_dir, exist_ok=True)

        with open(os.path.join(low_rl_dir, 'training_metrics.pkl'), 'wb') as f:
            pickle.dump(low_training_metrics, f)

        # Plot training curves
        visualizer = ResultsVisualizer()
        plt.figure(figsize=(12, 8))
        visualizer.plot_training_curves(low_training_metrics)
        plt.savefig(os.path.join(low_rl_dir, 'training_curves.png'))
        plt.close()

        # Run final validation
        print("\nRunning final low citation model validation...")
        final_metrics, final_predictions, final_targets = validate_low_citation_model(
            model=low_model,
            val_states=data['states'],
            citation_targets={k: v for k, v in data['citation_targets'].items() if k in data['val_states']},
            field_targets={k: v for k, v in data['field_targets'].items() if k in data['val_states']},
            device=device,
            threshold=threshold
        )

        print("\nLow model RL validation metrics (final):")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save trained model
        torch.save({
            'model_state_dict': low_model.state_dict(),
            'optimizer_state_dict': low_optimizer.state_dict(),
            'config': low_rl_config.to_dict(),
            'metrics': final_metrics,
            'date_trained': datetime.now().isoformat(),
            'is_multi_head': has_multi_head,
            'threshold': threshold
        }, os.path.join(low_rl_dir, 'trained_low_model.pt'))

        return final_metrics

    def run_rl_for_high_model():
        print("\n======= STARTING RL TRAINING WITH HIGH-CITATION MODEL =======\n")
        try_load_high_model_if_none()
        if high_model is None:
            print("Error: Attempting to RL-train high model, but it is None. Make sure it's pretrained or loaded.")
            return

        environment = setup_environment(config, data)

        for name, param in high_model.named_parameters():
            if 'citation_head' in name or 'policy_head' in name or 'value_head' in name:
                param.requires_grad = True
            elif 'transformer' in name:
                if 'layer' in name and ('2' in name or '3' in name):  # Unfreeze last 2 layers
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        high_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, high_model.parameters()),
            lr=config.training_config["learning_rate"] / 100,  # Base LR: 1e-6
            weight_decay=config.training_config["weight_decay"]
        )

        # Fix 6: Add Learning Rate Decay
        from torch.optim.lr_scheduler import StepLR
        high_scheduler = StepLR(
            high_optimizer,
            step_size=10,  # Decrease LR every 10 epochs
            gamma=0.7      # Multiply by 0.7 each time
        )
        high_rl_config = copy.deepcopy(config)
        high_rl_config.training_config["citation_coeff"] = 0.5
        high_rl_config.training_config["policy_weight"] = 3.0
        high_rl_config.training_config["field_weight"] = 0.3
        high_rl_config.training_config["clip_grad_norm"] = 0.5 

        fine_tune_citation_head(high_model, data, device)
        high_trainer = ARITTrainer(
            model=high_model,
            config=high_rl_config,
            env=environment,
            optimizer=high_optimizer,
            scheduler=high_scheduler,
            device=device,
            scaler=None,
            citation_targets=data['citation_targets'],
            field_targets=data['field_targets'],
            val_states=data['states'],
            val_citation_targets={k: v for k, v in data['citation_targets'].items() if k in data['val_states']},
            val_field_targets={k: v for k, v in data['field_targets'].items() if k in data['val_states']},
            use_log_transform=True
        )

        # Modify train() to apply scheduler step
        # original_train = high_trainer.train
        # def custom_train():
        #     metrics = original_train()
        #     for epoch in range(high_rl_config.training_config["num_epochs"]):
        #         high_scheduler.step()  # Apply LR decay after each epoch
        #         print(f"Epoch {epoch+1}: Learning Rate = {high_scheduler.get_last_lr()[0]:.8f}")
        #     return metrics
        # high_trainer.train = custom_train

        high_training_metrics = high_trainer.train()

        rl_results_dir = os.path.join(results_dir, 'rl_training')
        os.makedirs(rl_results_dir, exist_ok=True)
        high_rl_dir = os.path.join(rl_results_dir, 'high_model')
        os.makedirs(high_rl_dir, exist_ok=True)

        with open(os.path.join(high_rl_dir, 'training_metrics.pkl'), 'wb') as f:
            pickle.dump(high_training_metrics, f)

        visualizer = ResultsVisualizer()
        plt.figure(figsize=(12, 8))
        visualizer.plot_training_curves(high_training_metrics)
        plt.savefig(os.path.join(high_rl_dir, 'training_curves.png'))
        plt.close()

        val_state_ids = data['val_states']
        val_citation_targets = {k: v for k, v in data['citation_targets'].items() if k in val_state_ids}
        val_field_targets = {k: v for k, v in data['field_targets'].items() if k in val_state_ids}

        high_val_metrics, high_predictions, high_targets = validate_with_citation_targets(
            model=high_model,
            val_states=data['states'],
            citation_targets=val_citation_targets,
            field_targets=val_field_targets,
            device=device
        )
        print("\nHigh model RL validation metrics:")
        for key, value in high_val_metrics.items():
            print(f"  {key}: {value:.4f}")

        torch.save({
            'model_state_dict': high_model.state_dict(),
            'optimizer_state_dict': high_optimizer.state_dict(),
            'config': high_rl_config.to_dict(),
            'metrics': high_val_metrics,
            'date_trained': datetime.now().isoformat()
        }, os.path.join(high_rl_dir, 'trained_high_model.pt'))

        return high_val_metrics

    def run_rl_for_dual_models():
        print("\n======= STARTING RL TRAINING FOR DUAL MODELS =======\n")
        # Ensure both are loaded
        try_load_low_model_if_none()
        try_load_high_model_if_none()

        if (low_model is None) or (high_model is None):
            print("Error: Attempting RL with dual models, but one or both are None. Make sure both are pretrained/loaded.")
            return

        low_val_metrics = run_rl_for_low_model()
        high_val_metrics = run_rl_for_high_model()

        print("\n======= STARTING RL TRAINING WITH COMBINED MODEL APPROACH =======\n")

        class CombinedModelWrapper:
            def __init__(self, low_model, high_model, threshold=20):
                self.low_model = low_model
                self.high_model = high_model
                self.threshold = threshold
                self.horizons = low_model.horizons

                self.parameters = lambda: list(low_model.parameters()) + list(high_model.parameters())

            def to(self, device):
                self.low_model = self.low_model.to(device)
                self.high_model = self.high_model.to(device)
                return self

            def train(self):
                self.low_model.train()
                self.high_model.train()

            def eval(self):
                self.low_model.eval()
                self.high_model.eval()

            def forward(self, x, citation_data=None, mask=None, batch_idx=None):
                if isinstance(x, dict):
                    states = x.get('states')
                    citation_counts = x.get('citation_counts')
                else:
                    states = x
                    citation_counts = None

                batch_size = states.shape[0]
                policy_outputs = None
                values = None
                citation_params = None
                field_logits = None

                for i in range(batch_size):
                    state = states[i:i+1]
                    citation_count = citation_counts[i] if citation_counts is not None else 0
                    model = self.low_model if citation_count <= self.threshold else self.high_model

                    policy_out, value_out, c_param, f_logit = model(
                        state,
                        None if citation_data is None else {k: v[i:i+1] for k, v in citation_data.items()},
                        None if mask is None else mask[i:i+1],
                        batch_idx
                    )
                    if policy_outputs is None:
                        policy_outputs = torch.zeros((batch_size,) + policy_out.shape[1:], device=policy_out.device)
                        values = torch.zeros((batch_size,) + value_out.shape[1:], device=value_out.device)
                        citation_params = torch.zeros((batch_size,) + c_param.shape[1:], device=c_param.device)
                        field_logits = torch.zeros((batch_size,) + f_logit.shape[1:], device=f_logit.device)

                    policy_outputs[i] = policy_out
                    values[i] = value_out
                    citation_params[i] = c_param
                    field_logits[i] = f_logit

                return policy_outputs, values, citation_params, field_logits

            def get_citation_distribution(self, params, horizon_idx):
                # placeholder
                return None, None

        class CombinedModelTrainer(ARITTrainer):
            def __init__(self, low_model, high_model, threshold=20, **kwargs):
                self.low_model = low_model
                self.high_model = high_model
                self.threshold = threshold
                combined_model = CombinedModelWrapper(low_model, high_model, threshold)
                super().__init__(model=combined_model, **kwargs)
                self.low_optimizer = None
                self.high_optimizer = None

            def train_step(self, batch):
                states = batch['states']
                citation_counts = [s.citation_count for s in states]

                low_indices = [i for i, c in enumerate(citation_counts) if c <= self.threshold]
                high_indices = [i for i, c in enumerate(citation_counts) if c > self.threshold]

                if not low_indices or not high_indices:
                    return 0.0

                low_batch = {k: [v[i] for i in low_indices] for k, v in batch.items()}
                low_loss = super().train_step(low_batch, model=self.low_model, optimizer=self.low_optimizer)

                high_batch = {k: [v[i] for i in high_indices] for k, v in batch.items()}
                high_loss = super().train_step(high_batch, model=self.high_model, optimizer=self.high_optimizer)

                return (low_loss + high_loss) / 2

        environment = setup_environment(config, data)
        combined_trainer = CombinedModelTrainer(
            low_model=low_model,
            high_model=high_model,
            threshold=threshold,
            config=config,
            env=environment,
            optimizer=None,
            scheduler=None,
            device=device,
            scaler=None,
            citation_targets=data['citation_targets'],
            field_targets=data['field_targets'],
            val_states=data['states'],
            val_citation_targets={k: v for k, v in data['citation_targets'].items() if k in data['val_states']},
            val_field_targets={k: v for k, v in data['field_targets'].items() if k in data['val_states']},
            use_log_transform=True
        )
        combined_trainer.low_optimizer = torch.optim.Adam(
            low_model.parameters(),
            lr=config.training_config["learning_rate"] / 1000,
            weight_decay=config.training_config["weight_decay"]
        )
        combined_trainer.high_optimizer = torch.optim.Adam(
            high_model.parameters(),
            lr=config.training_config["learning_rate"] / 1000,
            weight_decay=config.training_config["weight_decay"]
        )
        combined_training_metrics = combined_trainer.train()

        rl_results_dir = os.path.join(results_dir, 'rl_training')
        combined_rl_dir = os.path.join(rl_results_dir, 'combined_model')
        os.makedirs(combined_rl_dir, exist_ok=True)

        with open(os.path.join(combined_rl_dir, 'training_metrics.pkl'), 'wb') as f:
            pickle.dump(combined_training_metrics, f)

        visualizer = ResultsVisualizer()
        plt.figure(figsize=(12, 8))
        visualizer.plot_training_curves(combined_training_metrics)
        plt.savefig(os.path.join(combined_rl_dir, 'training_curves.png'))
        plt.close()

        # Validate combined approach
        def validate_combined_model(low_model, high_model, val_states, citation_targets, field_targets, device, threshold=20):
            low_model.eval()
            high_model.eval()
            all_citation_preds = []
            all_citation_targets = []
            all_field_preds = []
            all_field_targets = []

            with torch.no_grad():
                for state_id in citation_targets.keys():
                    state = val_states[state_id]
                    state_np = state.to_numpy()
                    state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    if state.citation_count <= threshold:
                        model_ = low_model
                    else:
                        model_ = high_model

                    _, _, citation_params, field_logits = model_(state_tensor)
                    citation_preds = []
                    for h in range(model_.horizons):
                        nb_dist, mean = model_.get_citation_distribution(citation_params, h)
                        pred_mean_raw = mean.cpu().item() if mean is not None else 0
                        citation_preds.append(pred_mean_raw)

                    all_citation_preds.append(citation_preds)
                    all_citation_targets.append(citation_targets[state_id])

                    field_pred = torch.argmax(field_logits, dim=1).cpu().item()
                    all_field_preds.append(field_pred)
                    all_field_targets.append(field_targets[state_id])

            citation_metrics = CitationMetrics().compute(
                torch.tensor(all_citation_preds), torch.tensor(all_citation_targets)
            )
            field_acc = sum(p == t for p, t in zip(all_field_preds, all_field_targets)) / len(all_field_targets)
            metrics_ = {**citation_metrics, 'field_accuracy': field_acc}
            return metrics_, torch.tensor(all_citation_preds), torch.tensor(all_citation_targets)

        val_state_ids = data['val_states']
        val_citation_targets = {k: v for k, v in data['citation_targets'].items() if k in val_state_ids}
        val_field_targets = {k: v for k, v in data['field_targets'].items() if k in val_state_ids}

        combined_val_metrics, combined_predictions, combined_targets = validate_combined_model(
            low_model, high_model,
            data['states'],
            val_citation_targets,
            val_field_targets,
            device,
            threshold=threshold
        )
        print("\nCombined models RL validation metrics:")
        for key, value in combined_val_metrics.items():
            print(f"  {key}: {value:.4f}")

        torch.save({
            'low_model_state_dict': low_model.state_dict(),
            'high_model_state_dict': high_model.state_dict(),
            'threshold': threshold,
            'config': config.to_dict(),
            'metrics': combined_val_metrics,
            'date_trained': datetime.now().isoformat()
        }, os.path.join(combined_rl_dir, 'combined_trained_model.pt'))

        # Compare all
        print("\n======= COMPARING RL TRAINING APPROACHES =======\n")
        if low_val_metrics:
            print(f"Low model - RMSE: {low_val_metrics['rmse']:.4f}, "
                  f"MAE: {low_val_metrics['mae']:.4f}, "
                  f"Spearman: {low_val_metrics['spearman']:.4f}")
        if high_val_metrics:
            print(f"High model - RMSE: {high_val_metrics['rmse']:.4f}, "
                  f"MAE: {high_val_metrics['mae']:.4f}, "
                  f"Spearman: {high_val_metrics['spearman']:.4f}")
        print(f"Combined model - RMSE: {combined_val_metrics['rmse']:.4f}, "
              f"MAE: {combined_val_metrics['mae']:.4f}, "
              f"Spearman: {combined_val_metrics['spearman']:.4f}")

        best_approach = "combined"
        best_rmse = combined_val_metrics['rmse']
        if low_val_metrics and (low_val_metrics['rmse'] < best_rmse):
            best_approach = "low"
            best_rmse = low_val_metrics['rmse']
        if high_val_metrics and (high_val_metrics['rmse'] < best_rmse):
            best_approach = "high"
            best_rmse = high_val_metrics['rmse']
        print(f"\nBest approach based on RMSE: {best_approach.upper()} model")

    def run_rl_for_moe_model():
        """
        Run RL training for the Mixture of Experts model.
        """
        print("\n======= STARTING RL TRAINING WITH MIXTURE OF EXPERTS MODEL =======\n")
        
        # Try to load if still None
        try_load_moe_model_if_none()
        if moe_model is None:
            print("Error: Attempting to RL-train MoE model, but it is None. Make sure it's pretrained or loaded.")
            return
        
        environment = setup_environment(config, data)
        
        # Only unfreeze certain parts of the model
        for name, param in moe_model.named_parameters():
            # Always train policy and value networks
            if 'policy_net' in name or 'value_net' in name:
                param.requires_grad = True
            # Keep citation prediction components frozen as they were pretrained
            elif 'citation_head' in name:
                param.requires_grad = False
            # Partially unfreeze transformer
            elif 'transformer' in name:
                # Unfreeze only the last couple of layers
                if 'layer' in name and any(f'layer.{i}' in name for i in [2, 3]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Use an optimized learning rate for RL
        moe_lr = config.training_config["learning_rate"] / 100
        
        # Use adam optimizer
        moe_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, moe_model.parameters()),
            lr=moe_lr,
            weight_decay=config.training_config["weight_decay"]
        )
        
        # Use cosine annealing scheduler
        moe_scheduler = CosineAnnealingLR(
            moe_optimizer,
            T_max=config.training_config["num_epochs"],
            eta_min=moe_lr / 10
        )
        
        # Adjust configuration for MoE model
        moe_rl_config = copy.deepcopy(config)
        moe_rl_config.training_config["clip_grad_norm"] = 0.2
        moe_rl_config.training_config["policy_weight"] = 1.2
        moe_rl_config.training_config["ppo_epochs"] = 4
        
        # Standard trainer for MoE model
        moe_trainer = ARITTrainer(
            model=moe_model,
            config=moe_rl_config,
            env=environment,
            optimizer=moe_optimizer,
            scheduler=moe_scheduler,
            device=device,
            scaler=None,
            citation_targets=data['citation_targets'],
            field_targets=data['field_targets'],
            val_states=data['states'],
            val_citation_targets={k: v for k, v in data['citation_targets'].items() if k in data['val_states']},
            val_field_targets={k: v for k, v in data['field_targets'].items() if k in data['val_states']},
            use_log_transform=True
        )
        
        # Perform RL training
        moe_training_metrics = moe_trainer.train()
        
        # Save results
        rl_results_dir = os.path.join(results_dir, 'rl_training')
        os.makedirs(rl_results_dir, exist_ok=True)
        moe_rl_dir = os.path.join(rl_results_dir, 'moe_model')
        os.makedirs(moe_rl_dir, exist_ok=True)
        
        with open(os.path.join(moe_rl_dir, 'training_metrics.pkl'), 'wb') as f:
            pickle.dump(moe_training_metrics, f)
        
        # Plot training curves
        visualizer = ResultsVisualizer()
        plt.figure(figsize=(12, 8))
        visualizer.plot_training_curves(moe_training_metrics)
        plt.savefig(os.path.join(moe_rl_dir, 'training_curves.png'))
        plt.close()
        
        # Run final validation
        print("\nRunning final MoE model validation...")
        final_metrics, final_predictions, final_targets = validate_with_citation_targets(
            model=moe_model,
            val_states=data['states'],
            citation_targets={k: v for k, v in data['citation_targets'].items() if k in data['val_states']},
            field_targets={k: v for k, v in data['field_targets'].items() if k in data['val_states']},
            device=device
        )
        
        print("\nMoE model RL validation metrics (final):")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save trained model
        torch.save({
            'model_state_dict': moe_model.state_dict(),
            'optimizer_state_dict': moe_optimizer.state_dict(),
            'config': moe_rl_config.to_dict(),
            'metrics': final_metrics,
            'date_trained': datetime.now().isoformat()
        }, os.path.join(moe_rl_dir, 'trained_moe_model.pt'))
        
        # Analyze expert weights after RL training
        analyze_moe_experts(moe_model, data, device, moe_rl_dir)
        
        return final_metrics

    # RL logic
    if rl_low:
        run_rl_for_low_model()
    if rl_high:
        run_rl_for_high_model()
    if rl_both:
        run_rl_for_dual_models()

    # ------------------------------------------------------------------
    # VALIDATION PHASE
    # ------------------------------------------------------------------
    def validate_only(model_type="low"):
        """
        Validate a model that was presumably trained or loaded.
        We can do 'low', 'high', or 'both'.
        """
        val_state_ids = data['val_states']
        val_citation_targets = {k: v for k, v in data['citation_targets'].items() if k in val_state_ids}
        val_field_targets = {k: v for k, v in data['field_targets'].items() if k in val_state_ids}

        if model_type == "low":
            try_load_low_model_if_none()
            if low_model is None:
                print("Error: No low model found for validation.")
                return

            print("\n=== Validating low model ===\n")
            val_metrics, predictions, targets = validate_with_citation_targets(
                model=low_model,
                val_states=data['states'],
                citation_targets=val_citation_targets,
                field_targets=val_field_targets,
                device=device
            )
            print("Low model validation metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

        elif model_type == "high":
            try_load_high_model_if_none()
            if high_model is None:
                print("Error: No high model found for validation.")
                return

            print("\n=== Validating high model ===\n")
            val_metrics, predictions, targets = validate_with_citation_targets(
                model=high_model,
                val_states=data['states'],
                citation_targets=val_citation_targets,
                field_targets=val_field_targets,
                device=device
            )
            print("High model validation metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

        elif model_type == "both":
            try_load_low_model_if_none()
            try_load_high_model_if_none()
            if (low_model is None) or (high_model is None):
                print("Error: No dual models for validation.")
                return

            print("\n=== Validating dual models approach ===\n")

            def validate_combined_model(low_model, high_model, val_states, citation_targets, field_targets, device, threshold=20):
                low_model.eval()
                high_model.eval()
                all_citation_preds = []
                all_citation_targets_ = []
                all_field_preds = []
                all_field_targets_ = []

                with torch.no_grad():
                    for sid, cit_tgt in citation_targets.items():
                        state = val_states[sid]
                        state_np = state.to_numpy()
                        state_tensor = torch.tensor(state_np, dtype=torch.float32,
                                                    device=device).unsqueeze(0).unsqueeze(0)
                        if state.citation_count <= threshold:
                            model_ = low_model
                        else:
                            model_ = high_model

                        _, _, citation_params, field_logits = model_(state_tensor)
                        citation_preds = []
                        for h in range(model_.horizons):
                            nb_dist, mean_ = model_.get_citation_distribution(citation_params, h)
                            pred_mean_raw = mean_.cpu().item() if mean_ is not None else 0
                            citation_preds.append(pred_mean_raw)

                        all_citation_preds.append(citation_preds)
                        all_citation_targets_.append(cit_tgt)

                        field_pred = torch.argmax(field_logits, dim=1).cpu().item()
                        all_field_preds.append(field_pred)
                        all_field_targets_.append(field_targets[sid])

                citation_metrics = CitationMetrics().compute(
                    torch.tensor(all_citation_preds), torch.tensor(all_citation_targets_)
                )
                field_acc = sum(p == t for p, t in zip(all_field_preds, all_field_targets_)) / len(all_field_targets_)
                metrics_ = {**citation_metrics, 'field_accuracy': field_acc}
                return metrics_

            combined_val_metrics = validate_combined_model(
                low_model,
                high_model,
                data['states'],
                val_citation_targets,
                val_field_targets,
                device,
                threshold=threshold
            )
            print("Combined dual model validation metrics:")
            for k, v in combined_val_metrics.items():
                print(f"  {k}: {v:.4f}")

    def validate_moe_model():
        """
        Validate the Mixture of Experts model.
        """
        print("\n=== Validating Mixture of Experts model ===\n")
        try_load_moe_model_if_none()
        if moe_model is None:
            print("Error: No MoE model found for validation.")
            return
        
        val_state_ids = data['val_states']
        val_citation_targets = {k: v for k, v in data['citation_targets'].items() if k in val_state_ids}
        val_field_targets = {k: v for k, v in data['field_targets'].items() if k in val_state_ids}
        
        # Standard validation
        val_metrics, predictions, targets = validate_with_citation_targets(
            model=moe_model,
            val_states=data['states'],
            citation_targets=val_citation_targets,
            field_targets=val_field_targets,
            device=device
        )
        
        print("MoE model validation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Add expert analysis
        expert_weights = analyze_moe_experts(moe_model, data, device, results_dir)
        
        return val_metrics

    # Validation logic
    if validate_low:
        validate_only("low")
    if validate_high:
        validate_only("high")
    if validate_both:
        validate_only("both")
    if pretrain_moe:
        try_load_moe_model_if_none()
        pretrain_moe_model()

    if rl_moe:
        run_rl_for_moe_model()

    if validate_moe:
        validate_moe_model()

    print(f"\nAll requested phases have completed. Results are in: {results_dir}")


if __name__ == "__main__":
    main()
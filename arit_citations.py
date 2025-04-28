import random
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from arit_evaluation import CitationMetrics

# Constants for citation transformation
LOG_TRANSFORM = True  # Whether to use log transformation
MAX_CITATION = 5000  # Cap extremely high citation counts

def transform_citations(citations):
    """Transform citation counts to a more model-friendly distribution"""
    if isinstance(citations, (list, tuple, np.ndarray)):
        citations = np.array(citations)
        # Cap extremely high values
        citations = np.minimum(citations, MAX_CITATION)
        if LOG_TRANSFORM:
            return np.log1p(citations)  # log(1+x) to handle zeros
        return citations
    else:
        # Handle single value
        citation = min(float(citations), MAX_CITATION)
        if LOG_TRANSFORM:
            return math.log1p(citation)
        return citation

def inverse_transform_citations(transformed_citations):
    """Convert transformed citations back to original scale"""
    if isinstance(transformed_citations, (list, tuple, np.ndarray)):
        transformed_citations = np.array(transformed_citations)
        if LOG_TRANSFORM:
            return np.expm1(transformed_citations)  # exp(x)-1 is inverse of log1p
        return transformed_citations
    else:
        # Handle single value
        if LOG_TRANSFORM:
            return math.expm1(float(transformed_citations))
        return float(transformed_citations)

class CitationDataset(Dataset):
    def __init__(self, states, targets, transform=True):
        self.state_ids = list(states.keys())
        self.states = states
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.state_ids)
        
    def __getitem__(self, idx):
        sid = self.state_ids[idx]
        state = self.states[sid]
        target = self.targets[sid]
        
        # Transform targets if needed
        if self.transform:
            if isinstance(target, list):
                target = [transform_citations(t) for t in target]
            else:
                target = transform_citations(target)
        
        return state.to_numpy(), target

def collate_fn(batch, device):
    """Custom collate function for citation data"""
    states = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Convert to tensors
    states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    
    # If using log transformation, targets can be float
    if LOG_TRANSFORM:
        targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32, device=device)
    else:
        # Otherwise keep as long for the NegativeBinomial distribution
        targets_tensor = torch.tensor(np.array(targets), dtype=torch.long, device=device)
    
    return states_tensor, targets_tensor

# Direct regression with MSE loss
class DirectCitationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_horizons):
        super().__init__()
        self.num_horizons = num_horizons
        
        # Simpler architecture for direct prediction
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_horizons)
        )
        
        # Initialize with small weights
        self._initialize_params()
    
    def _initialize_params(self):
        """Initialize parameters with better defaults for regression"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass for citation prediction.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tensor of shape [batch_size, num_horizons] with citation predictions
        """
        return self.network(x)  # Shape: [batch_size, num_horizons]

# Improved NegativeBinomial prediction head
class ImprovedNBCitationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_horizons):
        super().__init__()
        self.num_horizons = num_horizons
        
        self.common_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Separate alpha and beta parameter networks for better stability
        self.alpha_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_horizons),
            nn.ReLU()  # This ensures alpha is always positive
        )
        
        self.beta_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_horizons),
        )
        
        # Add a small positive bias to beta parameters
        for m in self.beta_network.modules():
            if isinstance(m, nn.Linear) and m.out_features == num_horizons:
                # Initialize with a small positive bias
                nn.init.constant_(m.bias, 0.1)
        
        self._initialize_params()
    
    def _initialize_params(self):
        """Initialize parameters with better defaults for the NB distribution"""
        # Initialize alpha outputs to generate total_counts around 1-5 after softplus
        for m in self.alpha_network.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.num_horizons:  # Last layer
                    nn.init.zeros_(m.weight)  # Start with zeros
                    nn.init.constant_(m.bias, 0.5)  # Slight positive bias
        
        # Initialize beta outputs to generate probs around 0.5 after sigmoid
        for m in self.beta_network.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.num_horizons:  # Last layer
                    nn.init.zeros_(m.weight)  # Start with zeros
                    nn.init.zeros_(m.bias)    # No bias for 0.5 probability
    
    def forward(self, x):
        """
        Forward pass for citation prediction.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tensor of shape [batch_size, 2*num_horizons] where:
            - First num_horizons values are alpha parameters (for total_count)
            - Last num_horizons values are beta parameters (for probs)
        """
        common = self.common_encoder(x)
        
        # Generate alpha and beta parameters with tighter constraint on alpha
        alphas = self.alpha_network(common)
        
        # Apply lower bound to alpha to prevent zero or near-zero values
        alphas = alphas + 0.1
        
        # Process beta values
        betas = self.beta_network(common)
        
        # Concatenate along the last dimension
        params = torch.cat([alphas, betas], dim=-1)  # [batch_size, 2*num_horizons]
        
        return params

class WeightedNegativeBinomialLoss(nn.Module):
    def __init__(self, alpha=2.0):
        """
        Weighted Negative Binomial Loss that gives higher importance to papers with more citations.
        
        Args:
            alpha: Exponent controlling how much to weight higher citation counts.
                  Higher alpha means more weight to higher citation papers.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, nb_dist, target):
        """
        Compute weighted negative log-likelihood loss for Negative Binomial distribution.
        
        Args:
            nb_dist: torch.distributions.NegativeBinomial object
            target: Tensor of target citation counts [batch_size]
        Returns:
            Loss tensor [batch_size]
        """
        # Ensure target is float and on same device as distribution parameters
        target = target.float().to(nb_dist.mean.device)
        
        # Compute negative log-likelihood
        log_prob = nb_dist.log_prob(target)
        
        log_prob = torch.where(torch.isnan(log_prob) | torch.isinf(log_prob), 
                             torch.full_like(log_prob, -1e10), log_prob)
        
        # Calculate weights based on target values
        # Use target values raised to power alpha, normalized to sum to batch_size
        weights = torch.pow(target + 1.0, self.alpha)  # Adding 1 to handle zeros
        weights = weights * (target.size(0) / weights.sum())  # Normalize
        
        # Apply weights to negative log-likelihood
        weighted_nll = -log_prob * weights
        
        return weighted_nll.mean()  # Return mean of weighted negative log-likelihood

def analyze_citation_distribution(data, results_dir):
    """
    Analyze and visualize the citation count distribution.
    
    Args:
        data: Dictionary with citation data
        results_dir: Directory to save visualizations
    """
    print("\n--- Analyzing Citation Distribution ---")
    
    # Extract all citation targets
    all_citations = []
    citation_horizons = []
    
    # Count papers by citation range
    citation_ranges = [(0, 5), (6, 10), (11, 20),  # Low citation ranges (≤20)
        (21, 50), (51, 100), (101, 500), (501, 1000), (1001, float('inf'))]
    range_counts = {f"{low}-{high if high != float('inf') else '∞'}": 0 for low, high in citation_ranges}
    
    for targets in data['citation_targets'].values():
        if isinstance(targets, list):
            all_citations.extend(targets)
            if not citation_horizons:
                citation_horizons = list(range(len(targets)))
                
            # Count papers in each citation range
            for i, citations in enumerate(targets):
                for low, high in citation_ranges:
                    if low <= citations <= high:
                        range_counts[f"{low}-{high if high != float('inf') else '∞'}"] += 1
                        break
        else:
            # Handle other formats if needed
            print("Warning: Unexpected citation target format")
    
    # Convert to numpy array for analysis
    all_citations = np.array(all_citations)
    
    # Print statistics
    print(f"Total citation values: {len(all_citations)}")
    print(f"Mean: {np.mean(all_citations):.2f}")
    print(f"Median: {np.median(all_citations):.2f}")
    print(f"Std Dev: {np.std(all_citations):.2f}")
    print(f"Min: {np.min(all_citations):.2f}")
    print(f"Max: {np.max(all_citations):.2f}")
    print(f"90th percentile: {np.percentile(all_citations, 90):.2f}")
    print(f"95th percentile: {np.percentile(all_citations, 95):.2f}")
    print(f"99th percentile: {np.percentile(all_citations, 99):.2f}")
    
    # Print paper counts by citation range
    print("\nPapers by citation range:")
    for range_str, count in range_counts.items():
        print(f"  {range_str}: {count} papers")
    
    # Plot histogram with standard scale
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(all_citations, bins=50)
    plt.title('Citation Distribution')
    plt.xlabel('Citation Count')
    plt.ylabel('Frequency')
    
    # Plot histogram with log scale on y-axis
    plt.subplot(2, 2, 2)
    plt.hist(all_citations, bins=50)
    plt.yscale('log')
    plt.title('Citation Distribution (Log Y-scale)')
    plt.xlabel('Citation Count')
    plt.ylabel('Frequency (log scale)')
    
    # Plot transformed (log) histogram
    plt.subplot(2, 2, 3)
    log_citations = np.log1p(all_citations)
    plt.hist(log_citations, bins=50)
    plt.title('Log-transformed Citation Distribution')
    plt.xlabel('Log(1+Citations)')
    plt.ylabel('Frequency')
    
    # Plot CDF
    plt.subplot(2, 2, 4)
    plt.hist(all_citations, bins=50, cumulative=True, density=True)
    plt.title('Cumulative Distribution')
    plt.xlabel('Citation Count')
    plt.ylabel('Cumulative Probability')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'citation_distribution_analysis.png'))
    plt.close()
    
    return all_citations

class NegativeBinomialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, nb_dist, target):
        """
        Compute negative log-likelihood loss for Negative Binomial distribution.
        
        Args:
            nb_dist: torch.distributions.NegativeBinomial object
            target: Tensor of target citation counts [batch_size]
        Returns:
            Loss tensor [batch_size]
        """
        # Ensure target is float and on same device as distribution parameters
        target = target.float().to(nb_dist.mean.device)
        # Compute negative log-likelihood
        log_prob = nb_dist.log_prob(target)
        # Handle NaN/Inf by replacing with a large value
        log_prob = torch.where(torch.isnan(log_prob) | torch.isinf(log_prob), 
                             torch.full_like(log_prob, -1e10), log_prob)
        return -log_prob.mean()  # Negative log-likelihood (minimize)

class RangeAwareCitationLoss(nn.Module):
    def __init__(self, range_weights=None):
        super().__init__()
        self.base_loss = WeightedNegativeBinomialLoss(alpha=1.2)
        # Define weights for different citation ranges
        self.range_weights = range_weights or {
            (0, 5): 1.0,
            (6, 10): 2.5,
            (11, 20): 1.5,
        }
    
    def forward(self, nb_dist, target):
        # Get base loss
        base_loss = self.base_loss(nb_dist, target)
        
        # Apply range-specific weights
        batch_weights = torch.ones_like(target)
        for (low, high), weight in self.range_weights.items():
            # Create mask for this range
            range_mask = (target >= low) & (target <= high)
            batch_weights[range_mask] = weight
        
        # Apply weights to individual losses
        weighted_loss = base_loss * batch_weights
        return weighted_loss.mean()

class HybridCitationLoss(nn.Module):
    def __init__(self, range_weights=None):
        super().__init__()
        self.base_loss = WeightedNegativeBinomialLoss(alpha=1.2)
        self.range_weights = range_weights or {
            (0, 5): 1.2,   # Slightly increased weight
            (6, 10): 3.0,  # Increased weight for problematic range
            (11, 20): 1.8, 
        }
        self.mse_loss = nn.MSELoss()
        
    def forward(self, nb_dist, target, mean=None):
        # NB Loss with range weighting
        base_loss = self.base_loss(nb_dist, target)
        batch_weights = torch.ones_like(target)
        for (low, high), weight in self.range_weights.items():
            range_mask = (target >= low) & (target <= high)
            batch_weights[range_mask] = weight
        weighted_loss = base_loss * batch_weights
        
        # Add MSE loss on the predicted mean if provided
        mse_component = 0
        if mean is not None:
            mse_component = self.mse_loss(torch.log(mean + 1e-8), torch.log(target + 1e-8))
        
        return weighted_loss.mean(), mse_component

class MultiHeadCitationPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_horizons, use_log_transform=True):
        super().__init__()
        
        # Store config as instance attributes
        self.num_horizons = num_horizons
        self.use_log_transform = use_log_transform
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Separate prediction heads for different citation ranges
        self.range_0_5_head = ImprovedNBCitationHead(hidden_dim, hidden_dim//2, num_horizons)
        self.range_6_10_head = ImprovedNBCitationHead(hidden_dim, hidden_dim//2, num_horizons)
        self.range_11_plus_head = ImprovedNBCitationHead(hidden_dim, hidden_dim//2, num_horizons)
        
        # Range classifier to determine which head to use
        self.range_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 3),  # 3 ranges
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights with better defaults for low citation counts
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize range classifier to prefer the 0-5 range initially
        for m in self.range_classifier.modules():
            if isinstance(m, nn.Linear) and m.out_features == 3:
                # Initialize bias to prefer the 0-5 range initially
                nn.init.constant_(m.bias[0], 0.5)  # Higher bias for 0-5 range
                nn.init.constant_(m.bias[1], 0.0)  # Neutral for 6-10
                nn.init.constant_(m.bias[2], -0.5)  # Lower for 11+
    
    def forward(self, x, debug=False):
        encoded = self.encoder(x)
        
        # Get range probabilities
        range_probs = self.range_classifier(encoded)
        
        # Get predictions from each head
        pred_0_5 = self.range_0_5_head(encoded)
        pred_6_10 = self.range_6_10_head(encoded)
        pred_11_plus = self.range_11_plus_head(encoded)
        
        # Combine predictions based on range probabilities
        combined_pred = (
            range_probs[:, 0:1] * pred_0_5 +
            range_probs[:, 1:2] * pred_6_10 +
            range_probs[:, 2:3] * pred_11_plus
        )
        
        if debug and x.shape[0] > 0:
            print(f"Range probabilities: 0-5: {range_probs[0, 0].item():.4f}, " 
                  f"6-10: {range_probs[0, 1].item():.4f}, "
                  f"11+: {range_probs[0, 2].item():.4f}")
            
            # Print sample predictions from each head
            print(f"Sample predictions - 0-5 head: {pred_0_5[0, 0].item():.4f}, "
                  f"6-10 head: {pred_6_10[0, 0].item():.4f}, "
                  f"11+ head: {pred_11_plus[0, 0].item():.4f}, "
                  f"Combined: {combined_pred[0, 0].item():.4f}")
        
        return combined_pred, range_probs
    
    def get_distribution_for_horizon(self, params, horizon_idx, debug=False):
        """
        Get the citation distribution for a specific time horizon.
        
        Args:
            params: Tuple of (combined_pred, range_probs) from forward pass
            horizon_idx: Index of the horizon to get distribution for
            debug: Whether to print debug information
            
        Returns:
            Tuple of (negative_binomial_distribution, mean)
        """
        if isinstance(params, tuple) and len(params) == 2:
            combined_params, range_probs = params
        else:
            if debug:
                print("Warning: params is not a tuple in get_distribution_for_horizon")
            combined_params = params
            range_probs = None
        
        if combined_params.shape[-1] == self.num_horizons * 2:
            alphas = combined_params[:, horizon_idx]
            betas = combined_params[:, self.num_horizons + horizon_idx]
        else:
            param_width = combined_params.shape[-1] // self.num_horizons
            start_idx = horizon_idx * param_width
            alphas = combined_params[:, start_idx]
            betas = combined_params[:, start_idx + 1] if param_width > 1 else torch.sigmoid(alphas)
        
        # Improved parameter handling for low citation counts
        alphas = torch.relu(alphas) + 0.01
        # Scale total_count appropriately for low citation counts
        total_count = torch.clamp(alphas, min=0.1, max=10.0)  # Lower max for low citations
        probs = torch.sigmoid(betas)
        
        # Clamp probabilities to avoid numerical issues
        probs = torch.clamp(probs, min=0.05, max=0.95)
        
        try:
            dist = torch.distributions.NegativeBinomial(total_count=total_count, probs=probs)
            mean = total_count * (1 - probs) / probs
            
            # Clamp mean to reasonable range for low citation papers
            mean = torch.clamp(mean, min=0.0, max=20.0)
        except Exception as e:
            if debug:
                print(f"Error creating NB distribution: {str(e)}")
            mean = torch.ones_like(alphas)
            return None, mean

        if debug:
            if len(mean.shape) > 0:
                print(f"  [DEBUG] Alpha: min={alphas.min().item():.4f}, max={alphas.max().item():.4f}, avg={alphas.mean().item():.4f}")
                print(f"  [DEBUG] Beta: min={betas.min().item():.4f}, max={betas.max().item():.4f}, avg={betas.mean().item():.4f}")
                print(f"  [DEBUG] Total count: min={total_count.min().item():.4f}, max={total_count.max().item():.4f}")
                print(f"  [DEBUG] Probability p: min={probs.min().item():.4f}, max={probs.max().item():.4f}")
                print(f"  [DEBUG] Mean: min={mean.min().item():.4f}, max={mean.max().item():.4f}, avg={mean.mean().item():.4f}")
                if range_probs is not None:
                    top_experts = torch.argmax(range_probs, dim=1)
                    expert_counts = torch.bincount(top_experts, minlength=range_probs.shape[1])
                    print(f"  [DEBUG] Expert selection counts: {expert_counts.cpu().numpy()}")
                    for i in range(range_probs.shape[1]):
                        print(f"  [DEBUG] Expert {i} avg weight: {range_probs[:, i].mean().item():.4f}")
            else:
                print(f"  [DEBUG] Alpha: {alphas.item():.4f}")
                print(f"  [DEBUG] Beta: {betas.item():.4f}")
                print(f"  [DEBUG] Total count: {total_count.item():.4f}")
                print(f"  [DEBUG] Probability p: {probs.item():.4f}")
                print(f"  [DEBUG] Mean: {mean.item():.4f}")
        
        # Handle log transformation if needed
        if self.use_log_transform:
            log_mean = torch.log1p(mean)
            # Clamp log space values to appropriate range for low citations
            log_mean = torch.clamp(log_mean, min=0.0, max=3.0)  # log(21) ≈ 3.04
            if debug:
                print(f"  [DEBUG] Log-transformed mean: {log_mean[0].item() if len(log_mean.shape) > 0 else log_mean.item():.4f}")
                untransformed = torch.exp(log_mean) - 1
                print(f"  [DEBUG] Would transform back to: {untransformed[0].item() if len(untransformed.shape) > 0 else untransformed.item():.4f}")
            return dist, log_mean
        else:
            return dist, mean
        
class CitationCalibrationLayer(nn.Module):
    def __init__(self, num_ranges=3):
        super().__init__()
        # Create learnable calibration parameters per citation range
        self.range_scales = nn.Parameter(torch.ones(num_ranges))
        self.range_biases = nn.Parameter(torch.zeros(num_ranges))
        self.ranges = [(0, 5), (6, 10), (11, 20)]
    
    def forward(self, predictions, citation_counts=None):
        # Clone predictions to avoid modifying the original
        calibrated = predictions.clone()
        
        # If we have citation counts, use them to apply range-specific calibration
        if citation_counts is not None:
            for i, (low, high) in enumerate(self.ranges):
                mask = (citation_counts >= low) & (citation_counts <= high)
                if torch.any(mask):
                    calibrated[mask] = self.range_scales[i] * calibrated[mask] + self.range_biases[i]
        
        return calibrated

# Enhanced loss function for low-citation model
class EnhancedRangeAwareCitationLoss(nn.Module):
    def __init__(self, range_weights=None):
        super().__init__()
        # Use lower alpha to reduce over-weighting of higher citation counts
        self.base_loss = WeightedNegativeBinomialLoss(alpha=0.8)
        # Default weights with stronger emphasis on problematic ranges
        self.range_weights = range_weights or {
            (0, 5): 2.0,    # Increased weight for lowest range
            (6, 10): 1.0,   # Reduced weight for middle range
            (11, 20): 1.5,  # Moderate weight for higher range
        }
        # Add MSE component for better R² values
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, nb_dist, target):
        # NB Loss with range weighting
        base_loss = self.base_loss(nb_dist, target)
        
        # Get predicted mean
        mean = nb_dist.mean
        
        # Apply range-specific weights and directional penalties
        batch_weights = torch.ones_like(target)
        directional_penalty = torch.zeros_like(target)
        
        for (low, high), weight in self.range_weights.items():
            range_mask = (target >= low) & (target <= high)
            batch_weights[range_mask] = weight
            
            # Add directional penalties based on range
            if low == 0 and high == 5:
                # For 0-5 range: penalize overprediction more
                over_pred = torch.relu(mean[range_mask] - target[range_mask])
                directional_penalty[range_mask] = over_pred * 1.5
            elif low == 6 and high == 10:
                # For 6-10 range: balanced penalty
                diff = (mean[range_mask] - target[range_mask])
                directional_penalty[range_mask] = torch.abs(diff) * 0.4
            elif low == 11:
                # For 11+ range: penalize underprediction more
                under_pred = torch.relu(target[range_mask] - mean[range_mask])
                directional_penalty[range_mask] = under_pred * 2.0
        
        # Calculate MSE on log values for better numerical stability
        log_mean = torch.log(mean + 1e-8)
        log_target = torch.log(target + 1e-8)
        mse_losses = self.mse_loss(log_mean, log_target)
        
        # Combine all components into final loss
        weighted_loss = (batch_weights * base_loss).mean()
        mse_component = (batch_weights * mse_losses).mean()
        directional_component = (batch_weights * directional_penalty).mean()
        
        # Balance the different loss components
        return weighted_loss + 0.5 * mse_component + directional_component
    
class PowerCitationTransform:
    """Transform citation counts with a power-law transformation to better handle high citation papers."""
    
    def __init__(self, alpha=0.5):
        """
        Initialize with a power parameter.
        
        Args:
            alpha: Power value (0.5 = square root, 0.33 = cube root)
                  Lower values provide more aggressive scaling for high citations
        """
        self.alpha = alpha
    
    def transform(self, citations):
        """Transform citations to a more balanced scale."""
        return torch.sign(citations) * torch.pow(torch.abs(citations) + 1, self.alpha) - 1
    
    def inverse_transform(self, transformed_citations):
        """Convert back to the original citation scale."""
        return torch.sign(transformed_citations) * (torch.pow(torch.abs(transformed_citations) + 1, 1/self.alpha) - 1)

class RangeAwareMoECitationPredictor(nn.Module):
    """Citation prediction using Mixture of Experts with range-specific specialization."""
    
    def __init__(self, input_dim, horizons=4, num_experts=3, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.horizons = horizons
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Create citation range thresholds for expert specialization
        # Expert 0: 0-20 citations
        # Expert 1: 21-100 citations
        # Expert 2: 101+ citations
        self.range_thresholds = [0, 21, 101]
        
        # Expert gating network
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        
        # Create separate experts
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(num_experts)
        ])
        
        # Range classifier to predict citation range
        self.range_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
    
    def _create_expert(self):
        """Create a single expert network for citation prediction."""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.horizons)  # Alpha and beta for each horizon
        )
    
    def forward(self, x, citation_count=None):
        """
        Forward pass with range-specific expert assignment.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            citation_count: If provided, used for supervised range assignment during training.
        
        Returns:
            tuple (params, expert_weights)
        """
        batch_size = x.size(0)
        
        # Get expert weights from gate network
        expert_logits = self.gate_network(x)
        
        if self.training and citation_count is not None:
            range_targets = torch.zeros_like(expert_logits)
            for i, count in enumerate(citation_count):
                for j, threshold in enumerate(self.range_thresholds):
                    if count >= threshold:
                        target_expert = j
                    else:
                        break
                range_targets[i, target_expert] = 1.0
            
            # Debug: print target assignments and blend factor
            if hasattr(self, 'debug') and self.debug:
                print(f"[DEBUG] Range targets: {range_targets}")
            
            training_progress = 0.5
            blend_factor = torch.sigmoid(torch.tensor(5.0 * (training_progress - 0.5)))
            if hasattr(self, 'debug') and self.debug:
                print(f"[DEBUG] Blend factor: {blend_factor.item():.4f}")
            
            expert_logits = (1 - blend_factor) * range_targets + blend_factor * expert_logits
            if hasattr(self, 'debug') and self.debug:
                print(f"[DEBUG] Blended expert logits: {expert_logits}")
        
        # Apply softmax to get expert weights
        expert_weights = F.softmax(expert_logits, dim=1)
        if hasattr(self, 'debug') and self.debug:
            print(f"[DEBUG] Expert weights after softmax: {expert_weights}")
        
        # Get output from each expert
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        stacked_outputs = torch.stack(expert_outputs, dim=1)
        weights = expert_weights.unsqueeze(2)
        combined_output = (stacked_outputs * weights).sum(dim=1)
        
        if hasattr(self, 'debug') and self.debug:
            print(f"[DEBUG] Combined output shape: {combined_output.shape}")
        
        return combined_output, expert_weights
    
    def get_distribution_for_horizon(self, params_tuple, horizon_idx, debug=False):
        """
        Get the negative binomial distribution for a given horizon.
        """
        params, expert_weights = params_tuple
        alpha_idx = horizon_idx
        beta_idx = self.horizons + horizon_idx
        
        alpha = params[:, alpha_idx]
        beta = params[:, beta_idx]
        
        total_count = F.softplus(alpha) * 1.5 + 1.0
        total_count = torch.clamp(total_count, min=0.1, max=500.0)
        p = torch.sigmoid(beta) * 0.9 + 0.05
        p = torch.clamp(p, min=0.01, max=0.99)
        
        nb_dist = torch.distributions.NegativeBinomial(total_count=total_count, probs=p)
        mean = nb_dist.mean
        
        if debug:
            print(f"[DEBUG] RangeAwareMoECitationPredictor.get_distribution_for_horizon:")
            print(f"  Alpha: min={alpha.min().item():.4f}, max={alpha.max().item():.4f}, avg={alpha.mean().item():.4f}")
            print(f"  Beta: min={beta.min().item():.4f}, max={beta.max().item():.4f}, avg={beta.mean().item():.4f}")
            print(f"  Total count: min={total_count.min().item():.4f}, max={total_count.max().item():.4f}")
            print(f"  p (probability): min={p.min().item():.4f}, max={p.max().item():.4f}")
            print(f"  Mean: min={mean.min().item():.4f}, max={mean.max().item():.4f}, avg={mean.mean().item():.4f}")
            if expert_weights is not None:
                top_experts = torch.argmax(expert_weights, dim=1)
                expert_counts = torch.bincount(top_experts, minlength=expert_weights.shape[1])
                print(f"  Expert selection counts: {expert_counts.cpu().numpy()}")
                for i in range(expert_weights.shape[1]):
                    print(f"  Expert {i} avg weight: {expert_weights[:, i].mean().item():.4f}")
        
        return nb_dist, mean
        
class AdaptiveCitationTransform:
    """Citation count transformation with different parameters based on citation range."""
    
    def __init__(self, alphas=None, debug=False):
        """
        Initialize with power parameters for different ranges.
        """
        self.alphas = alphas or {
            (0, 20): 0.5,
            (21, 100): 0.4,
            (101, 500): 0.25
        }
        self.sorted_ranges = sorted(self.alphas.keys())
        self.debug = debug
    
    def _get_alpha(self, citations):
        if torch.is_tensor(citations):
            result = torch.zeros_like(citations)
            for (low, high), alpha in self.alphas.items():
                mask = (citations >= low) & (citations <= high)
                result[mask] = alpha
            return result
        else:
            for low, high in self.sorted_ranges:
                if low <= citations <= high:
                    return self.alphas[(low, high)]
            return self.alphas[self.sorted_ranges[-1]]
    
    def transform(self, citations):
        if torch.is_tensor(citations):
            alphas = self._get_alpha(citations)
            transformed = torch.sign(citations) * torch.pow(torch.abs(citations) + 1, alphas) - 1
            return transformed
        else:
            alpha = self._get_alpha(citations)
            return np.sign(citations) * (np.power(np.abs(citations) + 1, alpha) - 1)
    
    def inverse_transform(self, transformed_citations):
        if torch.is_tensor(transformed_citations):
            MAX_TRANSFORMED_VALUE = 6.2
            capped_transformed = torch.clamp(transformed_citations, max=MAX_TRANSFORMED_VALUE)
            if self.debug:
                print(f"[DEBUG] Inverse Transform - Capped transformed: {capped_transformed}")
            approx_citations = torch.sign(capped_transformed) * (torch.pow(torch.abs(capped_transformed) + 1, 1/0.4) - 1)
            if self.debug:
                print(f"[DEBUG] Inverse Transform - Approx citations: {approx_citations}")
            alphas = self._get_alpha(approx_citations)
            if self.debug:
                print(f"[DEBUG] Inverse Transform - Alphas: {alphas}")
            original = torch.sign(capped_transformed) * (torch.pow(torch.abs(capped_transformed) + 1, 1/alphas) - 1)
            MAX_CITATION = 500.0
            original_clamped = torch.clamp(original, max=MAX_CITATION)
            if self.debug:
                print(f"[DEBUG] Inverse Transform - Original (clamped): {original_clamped}")
            return original_clamped
        else:
            MAX_TRANSFORMED_VALUE = 6.2
            capped_transformed = min(transformed_citations, MAX_TRANSFORMED_VALUE)
            approx_citation = np.sign(capped_transformed) * (np.power(np.abs(capped_transformed) + 1, 1/0.4) - 1)
            alpha = self._get_alpha(approx_citation)
            original = np.sign(capped_transformed) * (np.power(np.abs(capped_transformed) + 1, 1/alpha) - 1)
            MAX_CITATION = 500.0
            return min(original, MAX_CITATION)
    
class ProgressiveLossWeighting:
    """
    Dynamically adjust loss weights for different citation ranges based on current performance.
    """
    
    def __init__(self, citation_ranges=None, initial_weights=None, adjustment_rate=0.1):
        """
        Initialize with citation ranges and initial weights.
        
        Args:
            citation_ranges: List of (min, max) ranges
            initial_weights: Initial weight for each range
            adjustment_rate: How quickly to adjust weights (0.0-1.0)
        """
        self.citation_ranges = citation_ranges or [
            (0, 5), (6, 20), (21, 50), (51, 100), (101, 500)
        ]
        self.weights = initial_weights or {
            (0, 5): 1.0,
            (6, 20): 2.0,
            (21, 50): 4.0,
            (51, 100): 8.0,
            (101, 500): 16.0
        }
        self.adjustment_rate = adjustment_rate
        
        # Track performance metrics for each range
        self.range_metrics = {range_key: {'rel_error': 1.0} for range_key in self.weights}
        
        # Initialize iteration counter
        self.iteration = 0
    
    def update_metrics(self, range_metrics):
        """
        Update performance metrics for each range.
        
        Args:
            range_metrics: Dict mapping range to metrics dict with 'rel_error'
        """
        for range_key, metrics in range_metrics.items():
            if range_key in self.range_metrics:
                self.range_metrics[range_key] = metrics
        
        # Adjust weights based on relative performance
        self._adjust_weights()
        
        # Increment iteration counter
        self.iteration += 1
    
    def _adjust_weights(self):
        """Adjust weights based on current performance metrics."""
        # Get current relative errors
        rel_errors = {range_key: metrics['rel_error'] for range_key, metrics in self.range_metrics.items()}
        
        # Calculate total error to normalize
        total_error = sum(rel_errors.values())
        if total_error == 0:
            return
        
        # Normalize errors to get the proportion of total error for each range
        error_proportions = {range_key: error / total_error for range_key, error in rel_errors.items()}
        
        # Calculate target weights proportional to error
        target_weights = {}
        for range_key, proportion in error_proportions.items():
            # Higher proportion of error = higher weight
            target_weights[range_key] = 1.0 + 15.0 * proportion  # Scale between 1.0 and 16.0
        
        for range_key in self.weights:
            current = self.weights[range_key]
            target = target_weights[range_key]
            
            # Apply adjustment with temperature annealing
            # Start with smaller adjustments, increase over time
            effective_rate = self.adjustment_rate * (1.0 - np.exp(-0.1 * self.iteration))
            
            self.weights[range_key] = current * (1 - effective_rate) + target * effective_rate
    
    def get_sample_weights(self, citation_counts):
        """
        Get weight for each sample based on its citation count.
        
        Args:
            citation_counts: Tensor of citation counts
            
        Returns:
            Tensor of same shape with weights
        """
        # Create output tensor of same shape
        weights = torch.ones_like(citation_counts)
        
        # Apply appropriate weight for each sample
        for (low, high), weight in self.weights.items():
            mask = (citation_counts >= low) & (citation_counts <= high)
            weights[mask] = weight
        
        return weights
    
    def get_range_weights_dict(self):
        """Get current weights as a dictionary."""
        return {f"{low}-{high}": self.weights[(low, high)] for low, high in self.citation_ranges}
    
    def __str__(self):
        """String representation with current weights."""
        weight_strs = [f"{low}-{high}: {self.weights[(low, high)]:.2f}" for low, high in self.citation_ranges]
        return f"Progressive Weights: {', '.join(weight_strs)}"

def debug_pretrain_validation(model, val_loader, device, use_log_transform=False, is_low_model=False, citation_transform=None):
    """Validate the model with debug output and optional citation transformation."""
    model.eval()
    all_preds = []
    all_targets = []
    all_field_preds = []
    all_field_targets = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            # Handle both 3-item and 4-item unpacking
            if len(batch_data) == 4:
                states, targets, field_targets, citation_counts = batch_data
            else:
                states, targets, field_targets = batch_data
                citation_counts = None
                
            states = states.unsqueeze(1)
            
            # Forward pass through model, providing citation_counts if available
            if citation_counts is not None:
                _, _, citation_params, field_logits = model(states, citation_count=citation_counts)
            else:
                _, _, citation_params, field_logits = model(states)
            
            # Get citation predictions for all horizons
            batch_preds = []
            for h in range(model.horizons):
                _, mean = model.get_citation_distribution(citation_params, h)
                
                # Apply appropriate transformation for the prediction
                if citation_transform is not None:
                    preds = citation_transform.inverse_transform(mean)
                elif use_log_transform:
                    if is_low_model:
                        # Lower cap for low citation model: log(20) ~ 3.0
                        LOG_SPACE_MAX = 3.0
                    else:
                        # Higher cap for high citation model: log(500) ~ 6.2
                        LOG_SPACE_MAX = 6.2
                    
                    capped_mean = torch.clamp(mean, max=LOG_SPACE_MAX)
                    preds = torch.exp(capped_mean) - 1
                    
                    # Additional clipping for low citation model
                    if is_low_model:
                        preds = torch.clamp(preds, min=0.0, max=20.0)
                else:
                    # No transformation
                    preds = mean
                    
                    # If low model, ensure predictions are within range
                    if is_low_model:
                        preds = torch.clamp(preds, min=0.0, max=20.0)
                
                batch_preds.append(preds)
            
            # Stack predictions for all horizons [batch_size, horizons]
            batch_preds = torch.stack(batch_preds, dim=1)
            
            # Debug first few batches
            if batch_idx < 3:
                # Select a few samples for debugging
                debug_indices = list(range(min(5, batch_preds.shape[0])))
                
                print(f"\n--- Debug Batch {batch_idx+1} ---")
                # If using transforms, show both the raw and transformed predictions
                if citation_transform is not None:
                    print("  Sample predictions (transformed space):", 
                          [f"{batch_preds[i, 0].item():.4f}" for i in debug_indices])
                    print("  Sample targets (original space):", 
                          [f"{targets[i, 0].item():.4f}" for i in debug_indices])
                    
                    # Calculate error in original space
                    orig_preds = citation_transform.inverse_transform(batch_preds[:, 0]).cpu().numpy()
                    orig_targets = targets[:, 0].cpu().numpy()
                    
                    rmse = np.sqrt(np.mean((orig_preds - orig_targets) ** 2))
                    mae = np.mean(np.abs(orig_preds - orig_targets))
                    print(f"  Original Space - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
                elif use_log_transform:
                    print("  Sample predictions (model output):", 
                        [f"{batch_preds[i, 0].item():.4f}" for i in debug_indices])
                    print("  Sample targets (original space):", 
                        [f"{targets[i, 0].item():.4f}" for i in debug_indices])
                    
                    if torch.max(batch_preds) > 10:
                        orig_preds = batch_preds[:, 0].cpu().numpy()
                        print("  (Values appear to be already in original space)")
                    else:
                        # Apply inverse log transform safely
                        orig_preds = torch.clamp(torch.exp(batch_preds[:, 0]) - 1, max=1e6).cpu().numpy()
                    
                    # Original targets
                    orig_targets = targets[:, 0].cpu().numpy()
                    
                    # Calculate metrics
                    rmse = np.sqrt(np.mean((orig_preds - orig_targets) ** 2))
                    mae = np.mean(np.abs(orig_preds - orig_targets))
                    print(f"  Original Space - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
                else:
                    print("  Sample predictions:", 
                          [f"{batch_preds[i, 0].item():.4f}" for i in debug_indices])
                    print("  Sample targets:", 
                          [f"{targets[i, 0].item():.4f}" for i in debug_indices])
                    
                    # Calculate error
                    preds_np = batch_preds[:, 0].cpu().numpy()
                    targets_np = targets[:, 0].cpu().numpy()
                    
                    rmse = np.sqrt(np.mean((preds_np - targets_np) ** 2))
                    mae = np.mean(np.abs(preds_np - targets_np))
                    print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            
            # Filter 0-citation papers for low model during validation
            if is_low_model:
                # Create mask for papers with citation counts > 0 and <= threshold (20)
                citation_counts = targets[:, 0]
                valid_citation_mask = (citation_counts > 0) & (citation_counts <= 20)
                
                # Apply mask to keep only valid citation papers
                if valid_citation_mask.sum() > 0:  # If any papers match the criteria
                    batch_preds = batch_preds[valid_citation_mask]
                    targets = targets[valid_citation_mask]
                    field_logits = field_logits[valid_citation_mask]
                    field_targets = field_targets[valid_citation_mask]
                else:
                    # Skip this batch if no papers match
                    continue
            
            # Get field predictions
            field_preds = torch.argmax(field_logits, dim=1)
            
            # Collect all predictions and targets
            all_preds.append(batch_preds.cpu())
            all_targets.append(targets.cpu())
            all_field_preds.append(field_preds.cpu())
            all_field_targets.append(field_targets.cpu())
    
    # Concatenate all batches
    if all_preds:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_field_preds = torch.cat(all_field_preds, dim=0)
        all_field_targets = torch.cat(all_field_targets, dim=0)
        
        # Calculate metrics
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
        mae = torch.mean(torch.abs(all_preds - all_targets)).item()
        
        # Calculate Spearman correlation
        preds_flat = all_preds.flatten().numpy()
        targets_flat = all_targets.flatten().numpy()
        
        # Handle spearman correlation (avoid division by zero)
        if len(np.unique(preds_flat)) > 1 and len(np.unique(targets_flat)) > 1:
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(preds_flat, targets_flat)
        else:
            spearman_corr = 0.0
        
        # Field accuracy
        field_acc = (all_field_preds == all_field_targets).float().mean().item()
        
        # Customize ranges based on model type
        if is_low_model:
            citation_ranges = [(0, 5), (6, 10), (11, 20)]
        else:
            citation_ranges = [(0, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 500)]
            
        range_analysis = {}
        
        for min_val, max_val in citation_ranges:
            # Create mask for papers within the citation range
            range_mask = (all_targets[:, 0] >= min_val) & (all_targets[:, 0] <= max_val)
            
            if range_mask.sum() > 0:
                range_preds = all_preds[range_mask]
                range_targets = all_targets[range_mask]
                
                # Calculate metrics for this range
                range_mae = torch.mean(torch.abs(range_preds - range_targets)).item()
                range_rmse = torch.sqrt(torch.mean((range_preds - range_targets) ** 2)).item()
                
                # Add small epsilon to avoid division by zero
                rel_error = torch.mean(torch.abs(range_preds - range_targets) / (range_targets + 1e-6)).item() * 100
                
                # Calculate average predicted/actual ratio
                pred_actual_ratio = torch.mean(range_preds / (range_targets + 1e-6)).item()
                
                range_analysis[f"{min_val}-{max_val}"] = {
                    "count": range_mask.sum().item(),
                    "mae": range_mae,
                    "rmse": range_rmse,
                    "rel_error": rel_error,
                    "pred_actual_ratio": pred_actual_ratio
                }
        
        # Print detailed metrics
        print("\n---- Validation Metrics Summary ----")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Spearman Correlation: {spearman_corr:.4f}")
        print(f"Field Accuracy: {field_acc:.4f}")
        
        print("\n---- Citation Range Analysis ----")
        for range_key, metrics in range_analysis.items():
            print(f"Range {range_key}: Count={metrics['count']}, MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, " +
                  f"Rel Error={metrics['rel_error']:.2f}%, Pred/Actual Ratio={metrics['pred_actual_ratio']:.2f}")
        
        return rmse, mae, spearman_corr, field_acc
    else:
        print("No valid batches found for evaluation!")
        return float('inf'), float('inf'), 0.0, 0.0
    
def pretrain_citation_model(model, config, data, optimizer, device, results_dir, epochs, batch_size, scaler, use_log_transform, custom_loss=None, which="low", after_batch_hook=None):
    threshold = 20

    is_low_model = (which == "low")
    # Update model type string to indicate filtering of 0-citations for low model
    if is_low_model:
        model_type_str = f"LOW citation (1-{threshold})"
    else:
        model_type_str = f"HIGH citation (>{threshold})"
    print(f"Training {model_type_str}")

    # Define a helper function for citation comparison that also removes 0 citation papers for low model
    def citation_check(sid):
        if sid not in data['citation_targets']:
            return False
            
        citation_value = max(data['citation_targets'][sid]) if isinstance(data['citation_targets'][sid], (list, tuple)) else data['citation_targets'][sid]
        if is_low_model:
            # Filter out 0 citation papers and ensure <= threshold
            return 0 < citation_value <= threshold
        else:
            return citation_value > threshold

    # Filter training states
    train_states = [data['states'][sid] for sid in data['train_states'] if citation_check(sid)]

    # Filter validation states
    val_states = [data['states'][sid] for sid in data['val_states'] if citation_check(sid)]

    print(f"Model type: {'LOW' if is_low_model else 'HIGH'} citation")
    print(f"Filtered to {len(train_states)} training states and {len(val_states)} validation states")
    
    # Analyze citation distribution for debugging (low model only)
    if is_low_model:
        citation_counts = []
        for s in train_states:
            if isinstance(data['citation_targets'][s.state_id], (list, tuple)):
                citation_counts.append(max(data['citation_targets'][s.state_id]))
            else:
                citation_counts.append(data['citation_targets'][s.state_id])
        
        from collections import Counter
        citation_dist = Counter(citation_counts)
        print("Citation distribution in training data:")
        for i in range(1, threshold+1):
            count = citation_dist.get(i, 0)
            percentage = (count / len(citation_counts)) * 100 if citation_counts else 0
            print(f"  {i} citations: {count} papers ({percentage:.1f}%)")

    def collate_fn(batch):
        states = torch.tensor([s.to_numpy() for s in batch], dtype=torch.float32, device=device)
        citation_targets = torch.tensor([data['citation_targets'][s.state_id] for s in batch], dtype=torch.float32, device=device)
        field_targets = torch.tensor([data['field_targets'][s.state_id] for s in batch], dtype=torch.long, device=device)
        return states, citation_targets, field_targets
    
    train_loader = DataLoader(train_states, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_states, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Set different parameters for low citation model
    if is_low_model:
        # For low citation model, use a more appropriate loss if none is provided
        if custom_loss is None:
            criterion_citation = nn.HuberLoss(delta=1.0)
            # Flag to indicate we're using a direct loss, not NB distribution loss
            use_direct_loss = True
        else:
            criterion_citation = custom_loss
            use_direct_loss = isinstance(custom_loss, (nn.HuberLoss, nn.MSELoss, nn.L1Loss))
            
        # For low citation model, use smaller caps and more gentle penalties
        lambda_penalty = 5.0  # Reduced from 10.0
        lambda_smooth = 0.5   # Reduced from 1.0
        lambda_var = 1.0      # Reduced from 3.0
        cap_value = 3.0       # log(20) is about 3.0 instead of 6.21
        print(f"Using specialized low citation settings: cap={cap_value}, lambda_penalty={lambda_penalty}, lambda_smooth={lambda_smooth}")
    else:
        criterion_citation = custom_loss if custom_loss is not None else NegativeBinomialLoss()
        use_direct_loss = False
        lambda_penalty = 10.0
        lambda_smooth = 1.0
        lambda_var = 3.0
        cap_value = 6.21  # log(500)
    
    criterion_field = nn.CrossEntropyLoss()
    
    best_field_acc = 0.0
    best_spearman = -1.0
    pretrain_metrics = {'train_loss': [], 'val_rmse': [], 'val_mae': [], 'val_spearman': [], 'val_field_acc': []}
    first_epoch_saved = False
    
    # Set up gradient accumulation
    grad_accum_steps = config.training_config.get("gradient_accumulation_steps", 2)
    print(f"Using gradient accumulation with {grad_accum_steps} steps")
    
    # Low model-specific adaptive quantile loss function 
    def low_model_adaptive_quantile_loss(preds, targets, base_quantile=0.5, weight=1.0):
        # Simple MSE for values <= 5
        low_mask = targets <= 5
        high_mask = targets > 5
        
        # Calculate losses separately for low and high values
        low_loss = torch.mean((preds[low_mask] - targets[low_mask])**2) if torch.any(low_mask) else 0
        
        # Use quantile loss only for higher values
        if torch.any(high_mask):
            norm_targets = (targets[high_mask] - 5) / 15  # Normalize from (5-20) to (0-1)
            adaptive_quantile = base_quantile + (0.3 * norm_targets)
            error = targets[high_mask] - preds[high_mask]
            quantile_loss = torch.zeros_like(error)
            for i in range(len(error)):
                q = adaptive_quantile[i]
                quantile_loss[i] = torch.max(q * error[i], (q - 1) * error[i])
            high_loss = torch.mean(quantile_loss)
        else:
            high_loss = 0
        
        # Weight the losses
        return weight * (0.7 * low_loss + 0.3 * high_loss)
    
    # Original adaptive quantile loss (unchanged)
    def adaptive_quantile_loss(preds, targets, base_quantile=0.8, weight=1.0):
        if torch.max(targets) > torch.min(targets):
            norm_targets = (targets - torch.min(targets)) / (torch.max(targets) - torch.min(targets))
        else:
            norm_targets = torch.zeros_like(targets)
        adaptive_quantile = base_quantile + (1.0 - base_quantile) * norm_targets
        error = targets - preds
        quantile_loss = torch.zeros_like(error)
        for i in range(len(error)):
            q = adaptive_quantile[i]
            quantile_loss[i] = torch.max(q * error[i], (q - 1) * error[i])
        return weight * torch.mean(quantile_loss)
    
    # Early stopping (only for low model)
    patience = 3 if is_low_model else 999
    no_improvement_count = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        optimizer.zero_grad()
        
        for i, (states, citation_targets, field_targets) in enumerate(train_loader):
            states = states.unsqueeze(1)
            
            # Forward pass
            policy_out, values, citation_params, field_logits = model(states)
            
            # Different path for low vs high model
            if is_low_model:
                # Specialized low citation model path
                loss_citation = 0
                log_preds = []
                
                for h in range(model.horizons):
                    nb_dist, mean = model.get_citation_distribution(citation_params, h)
                    
                    # For low model, use direct L1 loss on log space for simplicity
                    log_mean = mean  # Already in log space
                    log_target = torch.log(citation_targets[:, h] + 1.0)  # Add 1 before taking log
                    
                    if use_direct_loss:
                        # Direct loss on log values
                        h_loss = criterion_citation(log_mean, log_target)
                    else:
                        # Original NB distribution loss
                        h_loss = criterion_citation(nb_dist, citation_targets[:, h])
                    
                    # Add specialized quantile loss for low model
                    additional_loss = low_model_adaptive_quantile_loss(log_mean, log_target, base_quantile=0.5, weight=0.8)
                    
                    loss_citation += h_loss + additional_loss
                    log_preds.append(log_mean)
                
                loss_citation = loss_citation / model.horizons
                log_preds = torch.stack(log_preds, dim=1)
                
                # Apply cap penalty and smoothness penalty
                penalty = lambda_penalty * torch.relu(log_preds - cap_value).mean()
                diff = log_preds[:, 1:] - log_preds[:, :-1]
                smoothness_penalty = lambda_smooth * (diff ** 2).mean()
                loss_field = criterion_field(field_logits, field_targets)
                
                # Combine losses with simplified weights for low model
                weight_citation = config.training_config["citation_weight"] * 1.5  # Increased weight for citations
                weight_field = config.training_config["field_weight"]
                
                loss = (
                    weight_citation * (loss_citation + penalty + smoothness_penalty) +
                    weight_field * loss_field
                )
            else:
                loss_citation = 0
                loss_quantile = 0
                log_preds = []
                
                for h in range(model.horizons):
                    nb_dist, mean = model.get_citation_distribution(citation_params, h)
                    
                    h_loss = criterion_citation(nb_dist, citation_targets[:, h])
                    loss_citation += h_loss
                    
                    log_mean = torch.log(mean + 1e-8)
                    log_target = torch.log(citation_targets[:, h] + 1e-8)
                    loss_quantile += adaptive_quantile_loss(log_mean, log_target, base_quantile=0.8, weight=1.5)
                    log_preds.append(log_mean)
                
                loss_citation = loss_citation / model.horizons
                loss_quantile = loss_quantile / model.horizons
                log_preds = torch.stack(log_preds, dim=1)
                
                penalty = lambda_penalty * torch.relu(log_preds - cap_value).mean()
                diff = log_preds[:, 1:] - log_preds[:, :-1]
                smoothness_penalty = lambda_smooth * (diff ** 2).mean()
                loss_field = criterion_field(field_logits, field_targets)
                
                citation_magnitudes = torch.mean(citation_targets, dim=1)
                norm_magnitudes = citation_magnitudes / (torch.mean(citation_magnitudes) + 1e-8)
                target_variance = torch.var(torch.log(citation_targets + 1e-8))
                pred_variance = torch.var(log_preds, dim=1)
                weighted_variance_penalty = torch.mean(norm_magnitudes * torch.abs(pred_variance - target_variance))
                variance_penalty = lambda_var * weighted_variance_penalty
                
                weight_citation = torch.tensor(config.training_config["citation_weight"] * 2.0, 
                                             device=device, requires_grad=False)
                weight_field = torch.tensor(config.training_config["field_weight"], 
                                          device=device, requires_grad=False)
                
                dummy = torch.ones(1, device=device, requires_grad=True)
                
                loss = (
                    weight_citation * (loss_citation + penalty + smoothness_penalty + 
                                     variance_penalty + loss_quantile) +
                    weight_field * loss_field
                )
                
                if not loss.requires_grad:
                    print("WARNING: Loss doesn't require grad, adding dummy")
                    loss = loss + 0 * dummy
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Apply after_batch_hook if provided (for gradient clipping, etc.)
            if after_batch_hook is not None:
                after_batch_hook(model, optimizer)
            
            if (i + 1) % grad_accum_steps == 0 or (i + 1 == len(train_loader)):
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training_config["clip_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if after_batch_hook is None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training_config["clip_grad_norm"])
                    optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            batch_count += 1
        
        # Validation
        print(f"\n----- Validation for Epoch {epoch+1} -----")
        val_rmse, val_mae, val_spearman, val_field_acc = debug_pretrain_validation(
            model, val_loader, device, use_log_transform, is_low_model=is_low_model
        )
        
        avg_loss = total_loss / batch_count
        pretrain_metrics['train_loss'].append(avg_loss)
        pretrain_metrics['val_rmse'].append(val_rmse)
        pretrain_metrics['val_mae'].append(val_mae)
        pretrain_metrics['val_spearman'].append(val_spearman)
        pretrain_metrics['val_field_acc'].append(val_field_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
              f"Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, "
              f"Val Spearman: {val_spearman:.4f}, Val Field Acc: {val_field_acc:.4f}")
        print(f"Cap Penalty: {penalty.item():.4f}, Smoothness Penalty: {smoothness_penalty.item():.4f}")
        
        # Save first epoch model
        if not first_epoch_saved:
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': {
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_spearman': val_spearman,
                    'val_field_acc': val_field_acc
                },
                'use_log_transform': use_log_transform
            }, os.path.join(results_dir, 'first_epoch_model.pt'))
            first_epoch_saved = True
        
        # Save best field accuracy model
        if val_field_acc > best_field_acc:
            best_field_acc = val_field_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': {
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_spearman': val_spearman,
                    'val_field_acc': val_field_acc
                },
                'use_log_transform': use_log_transform
            }, os.path.join(results_dir, 'best_field_acc_model.pt'))
        
        # Save best citation model
        if val_spearman > best_spearman:
            best_spearman = val_spearman
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': {
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_spearman': val_spearman,
                    'val_field_acc': val_field_acc
                },
                'use_log_transform': use_log_transform
            }, os.path.join(results_dir, 'best_citation_model.pt'))
            
        # Early stopping - only for low model
        if is_low_model:
            current_val_loss = val_mae  # Use MAE as our primary metric
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                break
    
    # Load best model
    best_citation_path = os.path.join(results_dir, 'best_citation_model.pt')
    best_field_path = os.path.join(results_dir, 'best_field_acc_model.pt')
    first_epoch_path = os.path.join(results_dir, 'first_epoch_model.pt')
    
    if os.path.exists(best_citation_path):
        print("Loading best citation model (by Spearman correlation)")
        best_model_path = best_citation_path
    elif os.path.exists(best_field_path):
        print("Loading best field accuracy model")
        best_model_path = best_field_path
    else:
        print("Loading first epoch model")
        best_model_path = first_epoch_path
    
    best_state = torch.load(best_model_path, weights_only=False)
    
    return pretrain_metrics, best_state

def validate_citation_model(model, val_states, val_targets, device, citation_metrics, use_log_transform=True):
    """
    Validate the citation prediction component of the model.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for state_id, future_citations in val_targets.items():
            state = val_states[state_id]
            state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            # Get model prediction
            _, _, citation_params = model(state_tensor)
            
            # Extract predictions
            if hasattr(model, 'citation_head') and isinstance(model.citation_head, DirectCitationHead):
                # Direct regression predictions
                predictions = citation_params.cpu().numpy()[0]
            else:
                # Extract mean predictions from NB distributions
                predictions = []
                for h in range(model.horizons):
                    nb_dist = model.get_citation_distribution(citation_params, h)
                    pred_mean = nb_dist.mean.cpu()
                    predictions.append(pred_mean.item())
            
            # Extract target values
            if isinstance(future_citations, list):
                target_list = future_citations
            else:
                # Handle case where future_citations is a dict with time horizon keys
                target_list = [future_citations.get(h, 0) for h in range(1, model.horizons + 1)]
            
            # Transform targets if using log transform
            if use_log_transform:
                target_list = [transform_citations(t) for t in target_list]
            
            all_predictions.append(predictions)
            all_targets.append(target_list)
    
    # Calculate metrics
    metrics = {"mae": float('inf'), "rmse": float('inf'), "r2": -float('inf'), "spearman": 0.0}
    if all_predictions and all_targets:
        predictions_tensor = torch.tensor(all_predictions)
        targets_tensor = torch.tensor(all_targets)
        
        # Compute metrics on transformed scale
        metrics = citation_metrics.compute(predictions_tensor, targets_tensor)
        
        # For logging purposes, also compute metrics on original scale
        if use_log_transform:
            # Convert back to original scale for additional metrics
            orig_preds = torch.tensor([[inverse_transform_citations(p) for p in pred] for pred in all_predictions])
            orig_targets = torch.tensor([[inverse_transform_citations(t) for t in targ] for targ in all_targets])
            
            orig_metrics = citation_metrics.compute(orig_preds, orig_targets)
            print(f"Original scale metrics - RMSE: {orig_metrics['rmse']:.4f}, MAE: {orig_metrics['mae']:.4f}")
    
    return metrics

class MoECitationLoss(nn.Module):
    """
    Loss function specifically designed for Mixture of Experts citation model.
    Combines negative binomial loss with expert weighting regularization.
    """
    def __init__(self, alpha=0.8, expert_diversity_weight=0.1):
        super().__init__()
        self.base_loss = WeightedNegativeBinomialLoss(alpha=alpha)
        self.expert_diversity_weight = expert_diversity_weight
        self.mse_loss = nn.MSELoss()
        self.range_weights = {
            (0, 5): 1.0,
            (6, 10): 3.5,
            (11, 20): 1.8,
            (21, 50): 1.0,
            (51, 100): 1.2,
            (101, 500): 1.5
        }
        
    def forward(self, nb_dist, target, expert_weights=None):
        # Base negative binomial loss
        base_loss = self.base_loss(nb_dist, target)
        
        # Apply range-specific weights
        batch_weights = torch.ones_like(target)
        for (low, high), weight in self.range_weights.items():
            range_mask = (target >= low) & (target <= high)
            batch_weights[range_mask] = weight
        
        # Get predicted mean for MSE calculation
        mean = nb_dist.mean
        
        # Calculate MSE on log values for better numerical stability
        mse_component = self.mse_loss(torch.log(mean + 1e-8), torch.log(target + 1e-8))
        
        # Expert diversity regularization - encourage balanced expert usage
        diversity_loss = 0.0
        if expert_weights is not None:
            # Calculate diversity loss using entropy
            avg_weights = torch.mean(expert_weights, dim=0)
            # Ideal uniform distribution
            uniform_weights = torch.ones_like(avg_weights) / avg_weights.size(0)
            # KL divergence from uniform (higher means less diverse)
            entropy = -torch.sum(avg_weights * torch.log(avg_weights + 1e-10))
            max_entropy = -torch.sum(uniform_weights * torch.log(uniform_weights + 1e-10))
            diversity_loss = max_entropy - entropy
        
        weighted_loss = (batch_weights * base_loss).mean()
        
        return weighted_loss + 0.5 * mse_component + self.expert_diversity_weight * diversity_loss

class HighlyAsymmetricCitationLoss(nn.Module):
    """
    Loss function with extreme penalties for overprediction, especially for low-citation papers.
    """
    def __init__(self, 
                 range_weights={(0, 5): 10.0, (6, 20): 5.0, (21, 50): 1.0, (51, 100): 0.5, (101, 500): 0.3},
                 overprediction_multiplier=10.0,  # Much higher
                 underprediction_multiplier=0.2): # Much lower
        super().__init__()
        self.range_weights = range_weights
        self.sorted_ranges = sorted(range_weights.keys())
        self.overprediction_multiplier = overprediction_multiplier
        self.underprediction_multiplier = underprediction_multiplier
        
        # Base loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def forward(self, predictions, targets, expert_weights=None):
        batch_size = targets.shape[0]
        
        # Convert to CPU for numpy operations
        targets_np = targets.detach().cpu().numpy()
        preds_np = predictions.detach().cpu().numpy()
        
        # Determine error direction
        is_overprediction = preds_np > targets_np
        
        # Initialize weights tensor
        weights = torch.ones_like(targets)
        
        # Apply range-specific and direction-specific weights
        for i in range(batch_size):
            for horizon in range(targets.shape[1] if len(targets.shape) > 1 else 1):
                target_val = targets_np[i, horizon] if len(targets.shape) > 1 else targets_np[i]
                
                # Find range weight
                range_weight = 1.0
                for low, high in self.sorted_ranges:
                    if low <= target_val <= high:
                        range_weight = self.range_weights[(low, high)]
                        break
                
                # Get error direction
                if len(targets.shape) > 1:
                    overpred = is_overprediction[i, horizon]
                else:
                    overpred = is_overprediction[i]
                
                # Apply direction multiplier
                direction_multiplier = self.overprediction_multiplier if overpred else self.underprediction_multiplier
                
                if overpred:
                    # Add extra penalty factor for small citations (1/log scale)
                    size_factor = 1.0 / max(1.0, np.log2(target_val + 2))  # +2 to avoid division by zero
                    direction_multiplier *= size_factor
                
                # Apply combined weight
                if len(targets.shape) > 1:
                    weights[i, horizon] = range_weight * direction_multiplier
                else:
                    weights[i] = range_weight * direction_multiplier
        
        # Calculate weighted loss (more L1 for stability)
        combined_loss = 0.8 * (weights * self.l1_loss(predictions, targets)) + 0.2 * (weights * self.mse_loss(predictions, targets))
        
        # Add expert balance penalty if needed
        if expert_weights is not None:
            expert_balance_penalty = self._compute_expert_balance_penalty(expert_weights)
            return combined_loss.mean() + 0.2 * expert_balance_penalty
        
        return combined_loss.mean()
    
    def _compute_expert_balance_penalty(self, expert_weights):
        # Calculate mean usage of each expert
        mean_expert_usage = torch.mean(expert_weights, dim=0)
        
        # Calculate ideal distribution (uniform)
        num_experts = mean_expert_usage.shape[0]
        ideal_usage = torch.ones_like(mean_expert_usage) / num_experts
        
        # Calculate KL divergence
        kl_div = torch.sum(ideal_usage * torch.log(ideal_usage / (mean_expert_usage + 1e-10)))
        
        return kl_div
    
class CitationScalingLayer(nn.Module):
    """
    Add a learnable scaling layer that applies different scaling factors based on citation range.
    """
    def __init__(self, num_ranges=5):
        super().__init__()
        # Learnable scaling factors for each range, initialized conservatively
        self.range_scales = nn.Parameter(torch.tensor([0.1, 0.2, 0.4, 0.6, 0.8], dtype=torch.float32))
        self.range_biases = nn.Parameter(torch.zeros(num_ranges, dtype=torch.float32))
        
        # Define range boundaries
        self.range_boundaries = [0, 5, 20, 50, 100, 500]
    
    def forward(self, x, citation_counts=None):
        """
        Apply scaling based on citation range.
        
        Args:
            x: Log-space predictions
            citation_counts: Optional actual citation counts
        
        Returns:
            Scaled predictions
        """
        # If no citation counts provided, use exponential of predictions as estimate
        if citation_counts is None:
            # Estimate citation range from predictions
            est_citations = torch.exp(x) - 1
            citations_for_range = est_citations
        else:
            citations_for_range = citation_counts
        
        # Initialize output tensor
        scaled_x = torch.zeros_like(x)
        
        # Apply different scaling based on range
        for i in range(len(self.range_boundaries) - 1):
            low, high = self.range_boundaries[i], self.range_boundaries[i+1]
            
            # Create mask for this range
            range_mask = (citations_for_range >= low) & (citations_for_range <= high)
            
            # Apply scaling and bias for this range
            scale = torch.sigmoid(self.range_scales[i])  # Keep between 0 and 1
            bias = self.range_biases[i]
            
            # Apply transformation: x * scale + bias
            scaled_x = torch.where(range_mask, x * scale + bias, scaled_x)
        
        return scaled_x

def range_weighted_citation_loss(predictions, targets, base_weight=1.0):
    """
    Apply higher weights to high-citation papers in the loss function.
    """
    # Create weight tensor based on citation counts
    citation_weights = torch.ones_like(targets)
    
    # Define citation ranges and their weights
    ranges = [
        (0, 5, 1.0),     # Base weight for low citations
        (6, 20, 2.0),    # Double weight for medium citations
        (21, 50, 4.0),   # 4x weight for high citations
        (51, 100, 8.0),  # 8x weight for very high citations
        (101, 500, 16.0) # 16x weight for extremely high citations
    ]
    
    # Apply weights based on citation ranges
    for min_val, max_val, weight in ranges:
        mask = (targets >= min_val) & (targets <= max_val)
        citation_weights[mask] = weight
    
    # Calculate weighted MSE
    squared_error = (predictions - targets) ** 2
    weighted_squared_error = squared_error * citation_weights
    
    return base_weight * torch.mean(weighted_squared_error)

def pretrain_moe_citation_model(model, config, data, optimizer, device, results_dir, epochs=40, batch_size=64, 
                                scaler=None, use_log_transform=False, after_batch_hook=None):
    """
    Pretrain the Mixture of Experts citation model with improved handling of high citations.
    """
    print("Training Mixture of Experts citation model with enhanced high-citation handling")
    
    # Replace standard transform with adaptive transform
    citation_transform = AdaptiveCitationTransform(
        alphas={(0, 20): 0.5, (21, 100): 0.4, (101, 500): 0.25}
    )
    
    # Initialize the progressive weight adjuster
    progressive_weights = ProgressiveLossWeighting(
        citation_ranges=[(0, 5), (6, 20), (21, 50), (51, 100), (101, 500)],
        adjustment_rate=0.1
    )
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    # Set up learning rate scheduler for longer training
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    
    # Debug: Data Format Check
    print("\n----- DEBUG: Data Format Check -----")
    first_state_id = next(iter(data['states'].keys()))
    first_state = data['states'][first_state_id]
    print(f"Sample state ID: {first_state_id}")
    print(f"State type: {type(first_state)}")
    print(f"State attributes: {dir(first_state)[:10]}...")
    print(f"Citation count: {first_state.citation_count}")
    if first_state_id in data['citation_targets']:
        print(f"Citation targets: {data['citation_targets'][first_state_id]}")
    print("------------------------------------\n")
    
    # Use all training data without filtering by citation count
    train_states = [data['states'][sid] for sid in data['train_states']]
    val_states = [data['states'][sid] for sid in data['val_states']]
    
    print(f"Using {len(train_states)} training states and {len(val_states)} validation states")
    
    def collate_fn(batch):
        # Debug batch processing occasionally
        debug_this_batch = False
        if random.random() < 0.01:  # Debug approximately 1% of batches
            debug_this_batch = True
            print("\n----- DEBUG: Batch Format Check -----")
            print(f"Batch size: {len(batch)}")
            print(f"First item type: {type(batch[0])}")
            state_shape = batch[0].to_numpy().shape if hasattr(batch[0], 'to_numpy') else None
            print(f"First item numpy shape: {state_shape}")
            citation_count = batch[0].citation_count if hasattr(batch[0], 'citation_count') else None
            print(f"First item citation count: {citation_count}")
            print("------------------------------------")
        
        # Convert to numpy array first for efficiency 
        states_np = np.array([s.to_numpy() for s in batch], dtype=np.float32)
        states = torch.from_numpy(states_np).to(device)
        
        citation_targets = torch.tensor([data['citation_targets'][s.state_id] for s in batch], dtype=torch.float32, device=device)
        field_targets = torch.tensor([data['field_targets'][s.state_id] for s in batch], dtype=torch.long, device=device)
        
        # Include citation counts for range-specific expert assignment
        citation_counts = torch.tensor([s.citation_count for s in batch], dtype=torch.float32, device=device)
        
        if debug_this_batch:
            print(f"States tensor shape: {states.shape}")
            print(f"Citation targets shape: {citation_targets.shape}")
            print(f"Field targets shape: {field_targets.shape}")
            print(f"Citation counts shape: {citation_counts.shape}")
            print("------------------------------------\n")
        
        return states, citation_targets, field_targets, citation_counts
    
    train_loader = DataLoader(train_states, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_states, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Use specialized MoE loss function
    criterion_citation = HighlyAsymmetricCitationLoss(
        range_weights={(0, 5): 3.0, (6, 20): 2.0, (21, 50): 1.0, (51, 100): 0.5, (101, 500): 0.3},
        overprediction_multiplier=3.0,  # Strongly penalize overpredicting
        underprediction_multiplier=0.5   # Be more lenient on underpredicting
    )
    criterion_field = nn.CrossEntropyLoss()
    
    lambda_penalty = 10.0
    lambda_smooth = 1.0
    cap_value = 6.21  # log(500)
    
    best_field_acc = 0.0
    best_spearman = -1.0
    pretrain_metrics = {'train_loss': [], 'val_rmse': [], 'val_mae': [], 'val_spearman': [], 'val_field_acc': []}
    first_epoch_saved = False
    
    # Set up gradient accumulation
    grad_accum_steps = config.training_config.get("gradient_accumulation_steps", 2)
    print(f"Using gradient accumulation with {grad_accum_steps} steps")
    
    # Add target-aware quantile loss function
    def adaptive_quantile_loss(preds, targets, base_quantile=0.8, weight=1.0):
        if torch.max(targets) > torch.min(targets):
            norm_targets = (targets - torch.min(targets)) / (torch.max(targets) - torch.min(targets))
        else:
            norm_targets = torch.zeros_like(targets)
        adaptive_quantile = base_quantile + (1.0 - base_quantile) * norm_targets
        error = targets - preds
        quantile_loss = torch.zeros_like(error)
        for i in range(len(error)):
            q = adaptive_quantile[i]
            quantile_loss[i] = torch.max(q * error[i], (q - 1) * error[i])
        return weight * torch.mean(quantile_loss)
    
    for epoch in range(epochs):
        if hasattr(model.citation_head, 'update_epoch'):
            model.citation_head.update_epoch(epoch)
        model.train()
        total_loss = 0
        batch_count = 0
        
        optimizer.zero_grad()
        
        # Print current progressive weights
        print(f"\nCurrent citation range weights: {progressive_weights}")
        
        # Initial prediction check at beginning of training
        if epoch == 0:
            with torch.no_grad():
                print("\n----- DEBUG: Initial Prediction Check -----")
                debug_count = 0
                for val_state_id in data['val_states']:
                    if debug_count >= 5:
                        break
                    if val_state_id in data['citation_targets']:
                        state = data['states'][val_state_id]
                        target = data['citation_targets'][val_state_id]
                        
                        state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                        citation_count = torch.tensor([state.citation_count], dtype=torch.float32, device=device)
                        
                        # Pass citation_count to model for range-specific expert assignment
                        _, _, citation_params, _ = model(state_tensor, citation_count=citation_count)
                        
                        print(f"Sample {debug_count+1}:")
                        print(f"  State ID: {val_state_id}, Citation count: {state.citation_count}")
                        print(f"  Citation params type: {type(citation_params)}")
                        
                        if isinstance(citation_params, tuple):
                            print(f"  Citation params tuple length: {len(citation_params)}")
                            params, expert_weights = citation_params
                            print(f"  Expert weights: {expert_weights.detach().cpu().numpy()}")
                            print(f"  Selected expert: {torch.argmax(expert_weights, dim=1).item()}")
                        
                        # Get predictions with debug flag
                        predictions = []
                        for h in range(model.horizons):
                            nb_dist, mean = model.get_citation_distribution(citation_params, h, debug=(h==0))
                            
                            # Apply adaptive transformation
                            pred_transformed = mean.cpu().item()
                            pred_original = citation_transform.inverse_transform(
                                torch.tensor([pred_transformed])
                            ).item()
                            
                            predictions.append(pred_transformed)
                            original_predictions = citation_transform.inverse_transform(
                                torch.tensor(predictions)
                            ).tolist()
                        
                        print(f"  Transformed-space predictions: {predictions}")
                        print(f"  Original space predictions: {original_predictions}")
                        print(f"  Targets: {target}")
                        
                        debug_count += 1
                print("------------------------------------\n")
        
        for i, (states, citation_targets, field_targets, citation_counts) in enumerate(train_loader):
            if i == 0:
                print(f"Before unsqueeze, states shape: {states.shape}")
            states = states.unsqueeze(1)
            if i == 0:
                print(f"After unsqueeze, states shape: {states.shape}")
            
            # Debug during training
            debug_this_batch = (i == 0 and epoch < 2) or (i % 200 == 0 and random.random() < 0.2)
            
            # Forward pass with citation counts for range-specific expert assignment
            policy_out, values, citation_params, field_logits = model(states, citation_count=citation_counts)
            
            if debug_this_batch:
                print(f"\n----- DEBUG: Forward Pass (Epoch {epoch+1}, Batch {i+1}) -----")
                if isinstance(policy_out, tuple):
                    print(f"  Policy output is a tuple of length {len(policy_out)}")
                    for idx, item in enumerate(policy_out):
                        print(f"  Policy tuple item {idx} shape: {item.shape if hasattr(item, 'shape') else 'no shape'}")
                else:
                    print(f"  Policy output shape: {policy_out.shape}")
                
                print(f"  Values shape: {values.shape if hasattr(values, 'shape') else 'Values is not a tensor'}")
                print(f"  Citation params type: {type(citation_params)}")
                print(f"  Field logits shape: {field_logits.shape if hasattr(field_logits, 'shape') else 'Field logits is not a tensor'}")
                
                if isinstance(citation_params, tuple):
                    params, expert_weights = citation_params
                    print(f"  Expert weights shape: {expert_weights.shape}")
                    print(f"  Params shape: {params.shape}")
                print("----------------------------------------------------\n")
            
            # Calculate losses (citation, quantile, penalty, etc.)
            loss_citation = 0
            loss_quantile = 0
            transformed_preds = []
            expert_weights = None
            if isinstance(citation_params, tuple) and len(citation_params) == 2:
                combined_params, expert_weights = citation_params
            else:
                combined_params = citation_params
            
            for h in range(model.horizons):
                nb_dist, mean = model.get_citation_distribution(citation_params, h, debug=(debug_this_batch and h==0))
                if expert_weights is not None:
                    h_loss = criterion_citation(mean, citation_targets[:, h], expert_weights)
                else:
                    h_loss = criterion_citation(mean, citation_targets[:, h])

                transformed_targets = citation_transform.transform(citation_targets[:, h])
                sample_weights = progressive_weights.get_sample_weights(citation_targets[:, h])
                weighted_h_loss = h_loss * sample_weights
                loss_citation += torch.mean(weighted_h_loss)
                
                log_mean = torch.log(mean + 1e-8)
                log_target = torch.log(citation_targets[:, h] + 1e-8)
                quantile_weights = sample_weights * 0.5
                q_loss = adaptive_quantile_loss(log_mean, log_target, base_quantile=0.8, weight=1.0)
                weighted_q_loss = q_loss * quantile_weights
                loss_quantile += torch.mean(weighted_q_loss)
                
                transformed_preds.append(mean)
            
            loss_citation = loss_citation / model.horizons
            loss_quantile = loss_quantile / model.horizons
            transformed_preds = torch.stack(transformed_preds, dim=1)
            penalty = lambda_penalty * torch.relu(transformed_preds - cap_value).mean()
            diff = transformed_preds[:, 1:] - transformed_preds[:, :-1]
            smoothness_penalty = lambda_smooth * (diff ** 2).mean()
            loss_field = criterion_field(field_logits, field_targets)
            
            expert_balance_penalty = 0.0
            if expert_weights is not None:
                avg_expert_weights = torch.mean(expert_weights, dim=0)
                expert_count = avg_expert_weights.size(0)
                ideal_weight = 1.0 / expert_count
                weight_variance = torch.mean((avg_expert_weights - ideal_weight) ** 2)
                weight_entropy = -torch.sum(avg_expert_weights * torch.log(avg_expert_weights + 1e-8))
                max_entropy = torch.log(torch.tensor(expert_count, dtype=torch.float32, device=device))
                expert_balance_penalty = 0.5 * weight_variance + 0.5 * (max_entropy - weight_entropy)
                expert_balance_weight = 0.5 * (1.0 + epoch / epochs)
                expert_balance_penalty = expert_balance_weight * expert_balance_penalty
            
            citation_weight = torch.tensor(config.training_config["citation_weight"] * 2.0, 
                                             device=device, requires_grad=False)
            field_weight = torch.tensor(config.training_config["field_weight"], 
                                        device=device, requires_grad=False)
            
            loss = (
                citation_weight * (loss_citation + penalty + smoothness_penalty + loss_quantile) +
                field_weight * loss_field +
                expert_balance_penalty
            )
            
            if debug_this_batch:
                print(f"\n----- DEBUG: Loss Components (Epoch {epoch+1}, Batch {i+1}) -----")
                print(f"  Citation loss: {loss_citation.item():.4f}")
                print(f"  Penalty: {penalty.item():.4f}")
                print(f"  Smoothness penalty: {smoothness_penalty.item():.4f}")
                print(f"  Quantile loss: {loss_quantile.item():.4f}")
                print(f"  Field loss: {loss_field.item():.4f}")
                print(f"  Expert balance penalty: {expert_balance_penalty.item():.4f}")
                print(f"  Total loss: {loss.item():.4f}")
                print(f"  Progressive weights: {progressive_weights}")
                citation_counts_np = citation_counts.cpu().numpy()
                for range_key, count in zip(*np.unique(np.digitize(citation_counts_np, [0, 6, 21, 51, 101, 501]), return_counts=True)):
                    range_bins = [0, 6, 21, 51, 101, 501]
                    if range_key < len(range_bins) - 1:
                        range_str = f"{range_bins[range_key]}-{range_bins[range_key+1]-1}"
                        print(f"  Batch range {range_str}: {count} papers")
                print("----------------------------------------------------\n")
            
            if not loss.requires_grad:
                dummy = torch.ones(1, device=device, requires_grad=True)
                loss = loss + 0 * dummy
                if debug_this_batch:
                    print("  WARNING: Loss had no gradient. Added dummy gradient.")
                
            loss = loss / grad_accum_steps
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if after_batch_hook is not None:
                after_batch_hook(model, optimizer)
            
            if (i + 1) % grad_accum_steps == 0 or (i + 1 == len(train_loader)):
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training_config["clip_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if after_batch_hook is None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training_config["clip_grad_norm"])
                    optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            batch_count += 1
        
        scheduler.step()
        
        print(f"\n----- Validation for Epoch {epoch+1} -----")
        val_rmse, val_mae, val_spearman, val_field_acc = debug_pretrain_validation(
            model, val_loader, device, use_log_transform=False, 
            is_low_model=False, 
            citation_transform=citation_transform
        )
        
        range_metrics = {}
        for range_str in ["0-5", "6-20", "21-50", "51-100", "101-500"]:
            min_val, max_val = map(int, range_str.split('-'))
            range_metrics[(min_val, max_val)] = {'rel_error': 1.0}
        
        progressive_weights.update_metrics(range_metrics)
        print(f"Updated progressive weights: {progressive_weights}")
        
        if expert_weights is not None:
            avg_expert_weights = torch.mean(expert_weights, dim=0).detach().cpu().numpy()
            print(f"Expert utilization: {avg_expert_weights}")
            print(f"Top expert: {np.argmax(avg_expert_weights)} (weight: {np.max(avg_expert_weights):.4f})")
        
        avg_loss = total_loss / batch_count
        pretrain_metrics['train_loss'].append(avg_loss)
        pretrain_metrics['val_rmse'].append(val_rmse)
        pretrain_metrics['val_mae'].append(val_mae)
        pretrain_metrics['val_spearman'].append(val_spearman)
        pretrain_metrics['val_field_acc'].append(val_field_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, Val Spearman: {val_spearman:.4f}, Val Field Acc: {val_field_acc:.4f}")
        
        if not first_epoch_saved:
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': {
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_spearman': val_spearman,
                    'val_field_acc': val_field_acc
                },
                'citation_transform': {
                    'type': 'adaptive',
                    'alphas': citation_transform.alphas
                },
                'progressive_weights': progressive_weights.get_range_weights_dict(),
                'epoch': epoch
            }, os.path.join(results_dir, 'first_epoch_model.pt'))
            first_epoch_saved = True
        
        if val_field_acc > best_field_acc:
            best_field_acc = val_field_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': {
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_spearman': val_spearman,
                    'val_field_acc': val_field_acc
                },
                'citation_transform': {
                    'type': 'adaptive',
                    'alphas': citation_transform.alphas
                },
                'progressive_weights': progressive_weights.get_range_weights_dict(),
                'epoch': epoch
            }, os.path.join(results_dir, 'best_field_acc_model.pt'))
        
        if val_spearman > best_spearman:
            best_spearman = val_spearman
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': {
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_spearman': val_spearman,
                    'val_field_acc': val_field_acc
                },
                'citation_transform': {
                    'type': 'adaptive',
                    'alphas': citation_transform.alphas
                },
                'progressive_weights': progressive_weights.get_range_weights_dict(),
                'epoch': epoch
            }, os.path.join(results_dir, 'best_citation_model.pt'))
    
    best_citation_path = os.path.join(results_dir, 'best_citation_model.pt')
    best_field_path = os.path.join(results_dir, 'best_field_acc_model.pt')
    first_epoch_path = os.path.join(results_dir, 'first_epoch_model.pt')
    
    if os.path.exists(best_citation_path):
        print("Loading best citation model (by Spearman correlation)")
        best_model_path = best_citation_path
    elif os.path.exists(best_field_path):
        print("Loading best field accuracy model")
        best_model_path = best_field_path
    else:
        print("Loading first epoch model")
        best_model_path = first_epoch_path
    
    best_state = torch.load(best_model_path, weights_only=False)
    
    return pretrain_metrics, best_state

def analyze_moe_experts(model, data, device, results_dir):
    """
    Analyze how the MoE model utilizes different experts across the citation range.
    """
    print("\n--- Analyzing Mixture of Experts Utilization ---")
    
    model.eval()
    
    # Group papers by citation range
    citation_ranges = [(0, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 500)]
    range_expert_weights = {f"{low}-{high}": [] for low, high in citation_ranges}
    
    with torch.no_grad():
        for state_id, state in data['states'].items():
            if state_id in data['val_states']:  # Use validation set for analysis
                citation_count = state.citation_count
                
                # Find which range this paper belongs to
                range_key = None
                for low, high in citation_ranges:
                    if low <= citation_count <= high:
                        range_key = f"{low}-{high}"
                        break
                
                if range_key:
                    # Get model prediction with expert weights
                    state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    
                    # Add debugging information
                    if len(range_expert_weights[range_key]) == 0:
                        print(f"\nDEBUG - Processing sample for range {range_key}")
                        print(f"  State ID: {state_id}, Citation count: {citation_count}")
                        print(f"  State tensor shape: {state_tensor.shape}")
                    
                    _, _, citation_params, _ = model(state_tensor)
                    
                    # Extract expert weights if available
                    if isinstance(citation_params, tuple) and len(citation_params) == 2:
                        params, expert_weights = citation_params
                        
                        # Detach the tensor before converting to numpy
                        expert_weights_np = expert_weights.detach().cpu().numpy()[0]
                        range_expert_weights[range_key].append(expert_weights_np)
                        
                        if len(range_expert_weights[range_key]) <= 3:
                            print(f"  Expert weights for sample {len(range_expert_weights[range_key])}: {expert_weights_np}")
    
    # Calculate and print average expert weights for each range
    print("\n--- Expert Utilization by Citation Range ---")
    for range_key, weights_list in range_expert_weights.items():
        if weights_list:
            avg_weights = np.mean(weights_list, axis=0)
            print(f"Range {range_key}: {len(weights_list)} samples")
            print(f"  Average weights: {avg_weights}")
            print(f"  Top expert: {np.argmax(avg_weights)} (weight: {np.max(avg_weights):.4f})")
    
    print("MoE expert analysis complete.")
    
    return range_expert_weights


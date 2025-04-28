import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from arit_attention import CitationGAT
from arit_citations import ImprovedNBCitationHead, MultiHeadCitationPredictor
from arit_types import ARITAction



# Transformer Encoder
class TransformerEncoderLayerPreNorm(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="gelu", layer_norm_eps=1e-5):
        super().__init__()
        self.pre_ln_1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout_1 = nn.Dropout(dropout)

        self.pre_ln_2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        if activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "relu":
            self.act_fn = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x, mask=None):
        x_ln = self.pre_ln_1(x)
        attn_out, _ = self.self_attn(x_ln, x_ln, x_ln, attn_mask=mask)
        x = x + self.dropout_1(attn_out)

        x_ln = self.pre_ln_2(x)
        ff_out = self.linear2(self.dropout_2(self.act_fn(self.linear1(x_ln))))
        x = x + self.dropout_3(ff_out)
        return x

# 1. CitationIntentAttention: Specialized multi-head attention that 
#    forms intent-based queries and scores citation intent.
class CitationIntentAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, mode="review"):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.dropout = dropout
        self.mode = mode
        # Learnable intent query vector
        self.intent_query = nn.Parameter(torch.randn(1, d_model))
        # Standard multi-head attention with batch_first=True
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        # Linear layer to compute a scalar intent score
        self.intent_score_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        # Expand intent_query to each item in the batch
        intent_query = self.intent_query.expand(batch_size, 1, self.d_model)  # [batch, 1, d_model]
        # Use intent_query as query; x as key and value
        attn_output, attn_weights = self.attn(query=intent_query, key=x, value=x)
        # Compute a scalar score for citation intent
        intent_score = torch.sigmoid(self.intent_score_layer(attn_output))  # [batch, 1, 1]
        # Scale the attention output by the intent score
        output = attn_output * intent_score
        # Squeeze the sequence dimension
        return output.squeeze(1), attn_weights


# 2. TemporalAwareAttention: Applies a recency bias to attention weights,
#    decaying contributions based on temporal distance.
class TemporalAwareAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, recency_bias=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.dropout = dropout
        self.recency_bias = recency_bias
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

    def forward(self, x, time_diffs=None):
        attn_output, attn_weights = self.attn(query=x, key=x, value=x)
        if time_diffs is not None:
            if time_diffs.size(0) != x.size(0):
                time_diffs = time_diffs[:x.size(0)]
            decay = torch.exp(-self.recency_bias * time_diffs)  # Linear decay
            attn_weights = attn_weights * decay
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            output = torch.bmm(attn_weights, x)
        else:
            output = attn_output
        return output, attn_weights


# 3. FieldAwareAttention: Introduces field-specific biases by using a 
#    learned mapping from the field embedding to a bias term.
class FieldAwareAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        # Linear layer transforms field vector into bias scores
        self.field_bias_layer = nn.Linear(d_model, 1)

    def forward(self, x, field_vector):
        # x: [batch_size, seq_len, d_model]
        # field_vector: [batch_size, d_model]
        batch_size, seq_len, _ = x.size()
        # Expand field_vector to match sequence length
        field_vector_expanded = field_vector.unsqueeze(1).expand(batch_size, seq_len, self.d_model)
        # Compute bias scores for each token position
        bias = self.field_bias_layer(field_vector_expanded)  # [batch, seq_len, 1]
        bias = bias.squeeze(-1)  # [batch, seq_len]
        attn_output, attn_weights = self.attn(query=x, key=x, value=x)
        # Modulate output with a bias factor (using sigmoid activation)
        bias_factor = torch.sigmoid(bias).unsqueeze(-1)  # [batch, seq_len, 1]
        output = attn_output * bias_factor
        return output, attn_weights


# 4. ARITTransformerEncoder: Updated to integrate the custom attention layers.
#    It fuses outputs from CitationIntentAttention, TemporalAwareAttention,
#    and FieldAwareAttention using a gating mechanism.
class ARITTransformerEncoder(nn.Module):
    def __init__(self,
                 d_model=256,
                 input_dim=None,
                 n_heads=4,
                 num_layers=4,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="gelu",
                 layer_norm_eps=1e-5,
                 use_custom_attention=True,
                 gradient_checkpointing=False):
        super().__init__()
        if d_model % n_heads != 0:
            original_d_model = d_model
            d_model = (d_model // n_heads) * n_heads
            print(f"Warning: Adjusted d_model from {original_d_model} to {d_model} to be divisible by {n_heads} heads")
        if input_dim is None:
            input_dim = d_model
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.use_custom_attention = use_custom_attention
        self.gradient_checkpointing = gradient_checkpointing

        self.input_projection = nn.Linear(input_dim, d_model, bias=False)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model, eps=layer_norm_eps),
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model, eps=layer_norm_eps)

        if self.use_custom_attention:
            self.citation_intent_attn = CitationIntentAttention(d_model, n_heads, dropout=dropout, mode="review")
            self.temporal_attn = TemporalAwareAttention(d_model, n_heads, dropout=dropout, recency_bias=0.1)
            self.field_attn = FieldAwareAttention(d_model, n_heads, dropout=dropout)
            self.gate = nn.Linear(3 * d_model, d_model)

    def forward(self, x, mask=None, time_diffs=None, field_vector=None):
        import torch.utils.checkpoint as checkpoint

        x = self.input_projection(x)
        if self.use_custom_attention:
            citation_output, _ = self.citation_intent_attn(x)
            temporal_output, _ = self.temporal_attn(x, time_diffs=time_diffs)
            if field_vector is None:
                field_vector = x.mean(dim=1)
            field_output, _ = self.field_attn(x, field_vector=field_vector)
            temporal_summary = temporal_output.mean(dim=1)
            field_summary = field_output.mean(dim=1)
            fused = torch.cat([citation_output, temporal_summary, field_summary], dim=-1)
            fused = self.gate(fused)
            fused = fused.unsqueeze(1).expand_as(x)
            x = x + fused

        def layer_fn(x, layer):
            residual = x
            x = layer(x)
            return x + residual

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(layer_fn, x, layer)
            else:
                x = layer_fn(x, layer)

        x = self.final_ln(x)
        return x

# Mixture of Experts Gating Network
class ExpertGatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        
        # Gating network to decide expert weights
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: [batch_size, input_dim]
        # Returns: [batch_size, num_experts]
        return self.gate(x)

# Expert Citation Head
class ExpertCitationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_horizons, expert_id=0, init_mean_citation=None):
        super().__init__()
        self.expert_id = expert_id
        self.horizons = num_horizons
        
        # Custom architecture based on expert ID
        if expert_id == 0:  # Low-citation expert (0-5)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2 * num_horizons)
            )
        elif expert_id == 1:  # Mid-low citation expert (6-20)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.25),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 2 * num_horizons)
            )
        else:  # High-citation expert (>20)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * num_horizons)
            )
            
        # Initialize biases based on target citation range
        if init_mean_citation is not None:
            # Get the final layer
            final_layer = self.net[-1]
            
            # Calculate appropriate biases based on target citation range
            # For NB distribution with mean μ, we need total_count * (1-p)/p = μ
            log_mean = np.log1p(init_mean_citation)
            
            # Set alpha parameters (will go through softplus)
            for h in range(num_horizons):
                alpha_idx = h
                final_layer.bias.data[alpha_idx] = log_mean * 0.8
                
            # Set beta parameters (will go through sigmoid)
            # Higher p for low citations, lower p for high citations
            if expert_id == 0:  # Low citation expert
                beta_val = 0.2  # Higher p value (~0.55)
            elif expert_id == 1:  # Mid citation expert
                beta_val = 0.0  # Medium p value (~0.5)
            else:  # High citation expert
                beta_val = -0.2  # Lower p value (~0.45)
                
            for h in range(num_horizons):
                beta_idx = num_horizons + h
                final_layer.bias.data[beta_idx] = beta_val
    
    def forward(self, x):
        return self.net(x)

class RangeSpecializedMoERouter(nn.Module):
    """Router that assigns experts based on citation ranges with supervised warmup."""
    
    def __init__(self, input_dim, num_experts=3, hidden_dim=64, 
                 ranges=[(0, 5), (6, 50), (51, 500)],
                 warmup_epochs=5, decay_rate=0.8):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ranges = ranges
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.current_epoch = 0
        
        # Standard learned router
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Range mapping - ensure we have one range per expert
        assert len(ranges) == num_experts, f"Number of ranges ({len(ranges)}) must match number of experts ({num_experts})"
    
    def forward(self, x, citation_counts=None, training=True):
        """
        Forward pass with citation-based routing during warmup.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            citation_counts: Tensor of citation counts [batch_size]
            training: Whether in training mode
            
        Returns:
            Expert weights [batch_size, num_experts]
        """
        batch_size = x.shape[0]
        
        # Get learned routing logits
        learned_logits = self.router(x)
        
        if citation_counts is not None and training:
            # Calculate warmup coefficient (from 1.0 to 0.0 over warmup_epochs)
            warmup_coef = max(0.0, 1.0 - (self.current_epoch / self.warmup_epochs))
            warmup_coef = warmup_coef ** self.decay_rate  # Apply non-linear decay
            
            if warmup_coef > 0:
                # Create target routing based on citation ranges
                range_routing = torch.zeros(batch_size, self.num_experts, device=x.device)
                
                # Assign each sample to the appropriate expert based on citation count
                for i in range(batch_size):
                    count = citation_counts[i].item()
                    for expert_idx, (min_val, max_val) in enumerate(self.ranges):
                        if min_val <= count <= max_val:
                            range_routing[i, expert_idx] = 1.0
                            break
                    
                    # Edge case: if count is outside all ranges, assign to last expert
                    if range_routing[i].sum() == 0:
                        range_routing[i, -1] = 1.0
                
                # Apply temperature scaling to learned logits (higher temp = more uniform)
                temperature = 1.0 + 9.0 * warmup_coef  # Start high (10.0), end at 1.0
                scaled_logits = learned_logits / temperature
                
                # Interpolate between learned and range-based routing
                combined_logits = (1 - warmup_coef) * scaled_logits + warmup_coef * (range_routing * 10.0)  # High value to create sharp distribution
                
                # Apply softmax to get expert weights
                expert_weights = F.softmax(combined_logits, dim=1)
                
                return expert_weights
        
        # Standard routing for inference or after warmup
        expert_weights = F.softmax(learned_logits, dim=1)
        return expert_weights
    
    def update_epoch(self, epoch):
        """Track current epoch for warmup calculation."""
        self.current_epoch = epoch


# Mixture of Experts Citation Predictor
class MoECitationPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_horizons, num_experts=3, 
                 citation_ranges=[(0, 5), (6, 20), (21, 500)],
                 warmup_epochs=5, warmup_decay=0.8):
        super().__init__()
        self.num_experts = num_experts
        self.horizons = num_horizons
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.citation_ranges = citation_ranges
        self.warmup_epochs = warmup_epochs
        self.warmup_decay = warmup_decay
        self.current_epoch = 0
        
        # Make sure we have the right number of ranges for experts
        assert len(citation_ranges) == num_experts, f"Must provide {num_experts} citation ranges, got {len(citation_ranges)}"
        
        # Gating network
        self.gating_network = ExpertGatingNetwork(input_dim, hidden_dim, num_experts)
        
        # Expert networks - initialize with biases appropriate for each range
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            # Get the citation range for this expert
            min_val, max_val = citation_ranges[i]
            
            # Create expert with specialized initialization
            expert = ExpertCitationHead(
                input_dim, hidden_dim, num_horizons, 
                expert_id=i,
                init_mean_citation=(min_val + max_val) / 2
            )
            self.experts.append(expert)
        
    def forward(self, x, citation_count=None):
        """
        Forward pass with citation count-guided expert selection.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            citation_count: Optional tensor of citation counts [batch_size]
                           to guide expert selection
        """
        # x: [batch_size, input_dim]
        batch_size = x.shape[0]
        
        # Get expert weights from gating network
        learned_weights = self.gating_network(x)  # [batch_size, num_experts]
        
        # If citation count is provided, use range-based routing during warmup
        if citation_count is not None and self.training:
            # Calculate warmup coefficient (1.0 to 0.0 over warmup_epochs)
            warmup_coef = max(0.0, 1.0 - (self.current_epoch / self.warmup_epochs))
            warmup_coef = warmup_coef ** self.warmup_decay  # Apply non-linear decay
            
            if warmup_coef > 0:
                # Create target expert assignment based on citation ranges
                range_weights = torch.zeros_like(learned_weights)
                
                # Assign each sample to appropriate expert based on citation count
                for i in range(batch_size):
                    count = citation_count[i].item()
                    assigned = False
                    
                    # Check which range the citation count falls into
                    for expert_idx, (min_val, max_val) in enumerate(self.citation_ranges):
                        if min_val <= count <= max_val:
                            range_weights[i, expert_idx] = 1.0
                            assigned = True
                            break
                    
                    # If count is outside all ranges, assign to closest expert
                    if not assigned:
                        if count < self.citation_ranges[0][0]:
                            range_weights[i, 0] = 1.0  # Assign to first expert
                        else:
                            range_weights[i, -1] = 1.0  # Assign to last expert
                
                # Combine learned weights with range-based weights using warmup coefficient
                expert_weights = (1 - warmup_coef) * learned_weights + warmup_coef * range_weights
                
                # Ensure weights sum to 1
                expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)
            else:
                expert_weights = learned_weights
        else:
            expert_weights = learned_weights
        
        # Get predictions from each expert
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [batch_size, 2*horizons]
            expert_outputs.append(expert_out)
        
        # Stack expert outputs
        stacked_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, 2*horizons]
        
        # Weight the expert outputs
        weighted_outputs = stacked_outputs * expert_weights.unsqueeze(-1)  # [batch_size, num_experts, 2*horizons]
        
        # Sum over experts
        combined_output = weighted_outputs.sum(dim=1)  # [batch_size, 2*horizons]
        
        return combined_output, expert_weights
    
    def update_epoch(self, epoch):
        """Update current epoch for warmup calculation."""
        self.current_epoch = epoch
    
    def get_distribution_for_horizon(self, params, horizon_idx, debug=False):
        """
        Extract Negative Binomial distribution for a specific horizon.
        
        Args:
            params: Tuple of (combined_params, expert_weights) from forward pass
            horizon_idx: Index of horizon to get distribution for
            debug: Whether to print debug information
            
        Returns:
            Tuple of (NegativeBinomial distribution, mean)
        """
        if isinstance(params, tuple) and len(params) == 2:
            combined_params, expert_weights = params
        else:
            # Handle case where params is not a tuple (shouldn't happen)
            if debug:
                print("Warning: params is not a tuple in get_distribution_for_horizon")
            combined_params = params
            expert_weights = None
        
        # Extract parameters for this horizon from combined params
        alpha_idx = horizon_idx
        beta_idx = self.horizons + horizon_idx
        
        if len(combined_params.shape) == 2:
            alpha = combined_params[:, alpha_idx]
            beta = combined_params[:, beta_idx]
        else:
            alpha = combined_params[alpha_idx]
            beta = combined_params[beta_idx]
        
        # Process alpha and beta for Negative Binomial distribution
        total_count = F.softplus(alpha) + 1.0
        total_count = torch.clamp(total_count, min=0.1, max=500.0)
        probs = torch.sigmoid(beta)
        probs = torch.clamp(probs, min=0.01, max=0.99)
        
        try:
            # Create the distribution
            nb_dist = torch.distributions.NegativeBinomial(total_count=total_count, probs=probs)
            
            # Calculate mean for log-transform
            mean = total_count * (1 - probs) / probs
            
            # Return log of mean for log-transformed models
            log_mean = torch.log1p(mean)
            return nb_dist, log_mean
        except Exception as e:
            if debug:
                print(f"Error creating NB distribution: {str(e)}")
            # Return fallback values
            mean = torch.ones_like(alpha)
            return None, mean
        

# Policy, Value, and Citation Networks
class ARITPolicyNetwork(nn.Module):
    """
    Single trunk -> multiple heads for action components.
    field_positioning(25), novelty(1), collab(1), citation(10), combined_focus(10), timing(1).
    """
    def __init__(self, d_model=256, hidden_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )
        self.field_pos_head = nn.Linear(hidden_dim, 25)
        self.novelty_head = nn.Linear(hidden_dim, 1)
        self.collab_head = nn.Linear(hidden_dim, 1)
        self.citation_choices_head = nn.Linear(hidden_dim, 10)
        self.combined_focus_head = nn.Linear(hidden_dim, 10)
        self.timing_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        z = self.trunk(x)
        return (
            self.field_pos_head(z),
            self.novelty_head(z),
            self.collab_head(z),
            self.citation_choices_head(z),
            self.combined_focus_head(z),
            self.timing_head(z),
        )


class ARITValueNetwork(nn.Module):
    def __init__(self, d_model=256, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class ARITCitationHead(nn.Module):
    """
    Outputs negative binomial (r, alpha) for each horizon => 2*horizons total outputs.
    """
    def __init__(self, d_model=256, hidden_dim=256, horizons=4):
        super().__init__()
        self.horizons = horizons
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2*horizons)
        )

    def forward(self, x):
        return self.net(x)


# ARITModel - MoE support
class ARITModel(nn.Module):
    def __init__(self, config, input_dim=None, num_fields=172, use_multi_head_predictor=False, use_moe=False, num_experts=3):
        super().__init__()
        mc = config.model_config
        if input_dim is None:
            input_dim = mc.get("input_dim", mc["d_model"])
        
        self.device = torch.device("cpu")  # Default device, updated later with .to(device)
        self.d_model = mc["d_model"]
        self.horizons = mc["prediction_output_dim"]
        self.time_horizons = [1, 3, 6, 12]
        self.use_log_transform = mc.get("use_log_transform", True)
        self.use_moe = use_moe

        # Core transformer
        self.transformer = ARITTransformerEncoder(
            d_model=mc["d_model"],
            input_dim=input_dim,
            n_heads=mc["n_heads"],
            num_layers=mc["num_layers"],
            dim_feedforward=mc["dim_feedforward"],
            dropout=mc["dropout"],
            activation=mc["activation"],
            layer_norm_eps=mc["layer_norm_eps"],
            use_custom_attention=True,
            gradient_checkpointing=config.training_config.get("use_gradient_checkpointing", False)
        )
        
        # Add Graph Attention Network for citation networks
        self.citation_gat = CitationGAT(
            in_features=mc["d_model"],
            hidden_dim=mc["d_model"] // 4,  # Smaller hidden dimension
            out_features=mc["d_model"],
            dropout=mc["dropout"],
            alpha=0.2,
            n_heads=mc["n_heads"]
        )
        
        # Integration layer to combine transformer and GAT outputs
        self.feature_integration = nn.Sequential(
            nn.Linear(mc["d_model"] * 2, mc["d_model"]),
            nn.ReLU(),
            nn.Dropout(mc["dropout"]),
            nn.Linear(mc["d_model"], mc["d_model"])
        )
        
        self.policy_net = ARITPolicyNetwork(d_model=mc["d_model"], hidden_dim=mc["value_hidden_dim"])
        self.value_net = ARITValueNetwork(d_model=mc["d_model"], hidden_dim=mc["value_hidden_dim"])
        
        # Choose citation head based on configuration
        if use_moe:
            self.citation_head = MoECitationPredictor(
                input_dim=mc["d_model"],
                hidden_dim=mc["prediction_hidden_dim"],
                num_horizons=self.horizons,
                num_experts=num_experts
            )
        elif use_multi_head_predictor:
            self.citation_head = MultiHeadCitationPredictor(
                input_dim=mc["d_model"],
                hidden_dim=mc["d_model"],
                num_horizons=self.horizons
            )
        else:
            self.citation_head = ImprovedNBCitationHead(
                input_dim=mc["d_model"],
                hidden_dim=mc["prediction_hidden_dim"],
                num_horizons=mc["prediction_output_dim"]
            )
        
        # Field classification head
        self.field_head = nn.Sequential(
            nn.Linear(mc["d_model"], 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_fields)
        )
        
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if args and isinstance(args[0], torch.device):
            self.device = args[0]
        elif 'device' in kwargs:
            self.device = kwargs['device']
        return self
    
    def forward(self, x, citation_data=None, mask=None, batch_idx=None, citation_count=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            citation_data: Optional dict with citation network information
            mask: Optional attention mask
            batch_idx: Optional batch index for debugging
            citation_count: Optional tensor of citation counts for range-aware MoE
        
        Returns:
            policy_outputs: Policy head outputs
            value: Value head output
            citation_params: Citation prediction parameters
            field_logits: Field classification logits
        """
        # Debug: Check input tensor
        if batch_idx is None:
            batch_idx = 0  # Fallback for inference
        
        enc = self.transformer(x, mask=mask)
        final_rep = enc[:, -1, :]
        
        if citation_data is not None:
            node_features = citation_data['node_features']
            adj_matrices = citation_data['adj_matrices']
            node_masks = citation_data.get('masks')
            
            if node_features.shape[-1] != self.d_model:
                node_projection = nn.Linear(node_features.shape[-1], self.d_model).to(node_features.device)
                node_features = node_projection(node_features)
            
            graph_embedding, _ = self.citation_gat(node_features, adj_matrices, node_masks)
            combined_embedding = self.feature_integration(torch.cat([final_rep, graph_embedding], dim=1))
        else:
            combined_embedding = final_rep
        
        policy_outputs = self.policy_net(combined_embedding)
        value = self.value_net(combined_embedding)
        
        # Get citation predictions based on head type
        if isinstance(self.citation_head, MoECitationPredictor):
            # Pass citation_count for range-aware expert selection
            if hasattr(self.citation_head, 'forward') and 'citation_count' in str(self.citation_head.forward.__code__.co_varnames):
                citation_params, expert_weights = self.citation_head(combined_embedding, citation_count=citation_count)
            else:
                citation_params, expert_weights = self.citation_head(combined_embedding)
            citation_params = (citation_params, expert_weights)
            
            # # Debug MoE output
            # if batch_idx % 10 == 0:  # Limit frequency to avoid spam
            #     print(f"\n----- DEBUG: forward (MoECitationPredictor) Batch {batch_idx} -----")
            #     print(f"  Citation params shape: {citation_params[0].shape}")
            #     print(f"  Expert weights shape: {expert_weights.shape}")
            #     print(f"  Params min: {citation_params[0].min().item():.4f}, "
            #         f"max: {citation_params[0].max().item():.4f}, avg: {citation_params[0].mean().item():.4f}")
            #     if torch.any(torch.isinf(citation_params[0])) or torch.any(torch.isnan(citation_params[0])):
            #         print(f"  WARNING: Inf/NaN detected in citation_params: {citation_params[0]}")
            #     elif citation_params[0].max().item() > 100:  # Arbitrary threshold for raw params
            #         print(f"  WARNING: Large values in citation_params: max={citation_params[0].max().item():.4f}")
        
        elif isinstance(self.citation_head, MultiHeadCitationPredictor):
            head_preds, range_probs = self.citation_head(combined_embedding)
            citation_params = (head_preds, range_probs)
            
            # # Debug MultiHead output
            # if batch_idx % 10 == 0:
            #     print(f"\n----- DEBUG: forward (MultiHeadCitationPredictor) Batch {batch_idx} -----")
            #     print(f"  Head preds shape: {head_preds.shape}")
            #     print(f"  Range probs shape: {range_probs.shape}")
            #     print(f"  Head preds min: {head_preds.min().item():.4f}, "
            #         f"max: {head_preds.max().item():.4f}, avg: {head_preds.mean().item():.4f}")
            #     if torch.any(torch.isinf(head_preds)) or torch.any(torch.isnan(head_preds)):
            #         print(f"  WARNING: Inf/NaN detected in head_preds: {head_preds}")
        
        else:
            citation_params = self.citation_head(combined_embedding)
            
            # Debug standard head output
            # if batch_idx % 10 == 0:
            #     print(f"\n----- DEBUG: forward (Standard Head) Batch {batch_idx} -----")
            #     print(f"  Citation params shape: {citation_params.shape}")
            #     print(f"  Params min: {citation_params.min().item():.4f}, "
            #         f"max: {citation_params.max().item():.4f}, avg: {citation_params.mean().item():.4f}")
            #     if torch.any(torch.isinf(citation_params)) or torch.any(torch.isnan(citation_params)):
            #         print(f"  WARNING: Inf/NaN detected in citation_params: {citation_params}")
        
        # Apply calibration only if not using specialized predictors
        if (hasattr(self, 'citation_calibration') and 
            not isinstance(self.citation_head, MultiHeadCitationPredictor) and
            not isinstance(self.citation_head, MoECitationPredictor)):
            citation_counts = citation_data.get('citation_counts') if citation_data else citation_count
            if citation_counts is not None:
                citation_params = self.citation_calibration(citation_params, citation_counts)
        
        field_logits = self.field_head(combined_embedding)
        return policy_outputs, value, citation_params, field_logits
    
    # Distribution creation
    def get_policy_distributions(self, policy_outputs):
        """
        policy_outputs: tuple of 6 Tensors => (field_pos_logits, novelty, collab, citation, focus, timing)
        Each shape => [batch_size, dim]
        Returns a dict of distributions keyed by sub-action name.
        """
        (field_pos_logits, novelty_logits, collab_logits, citation_logits, focus_logits, timing_logits) = policy_outputs
        
        # Add safety checks for NaN or Inf values
        field_pos_logits = torch.nan_to_num(field_pos_logits, nan=0.0, posinf=1.0, neginf=-1.0)
        novelty_logits = torch.nan_to_num(novelty_logits, nan=0.0, posinf=1.0, neginf=-1.0)
        collab_logits = torch.nan_to_num(collab_logits, nan=0.0, posinf=1.0, neginf=-1.0)
        citation_logits = torch.nan_to_num(citation_logits, nan=0.0, posinf=1.0, neginf=-1.0)
        focus_logits = torch.nan_to_num(focus_logits, nan=0.0, posinf=1.0, neginf=-1.0)
        timing_logits = torch.nan_to_num(timing_logits, nan=0.0, posinf=1.0, neginf=-1.0)

        # 1) field_positioning: Categorical(25)
        field_pos_dist = dist.Categorical(logits=field_pos_logits)

        # 2) novelty_level: interpret as Beta. We'll convert novelty_logits => alpha/beta
        novelty_sig = torch.sigmoid(novelty_logits)  # shape [b,1]
        alpha_nov = novelty_sig * 5.0 + 1.0
        beta_nov = (1.0 - novelty_sig) * 5.0 + 1.0
        novelty_dist = dist.Beta(alpha_nov.squeeze(-1), beta_nov.squeeze(-1))

        # 3) collaboration_strategy: Bernoulli
        collab_prob = torch.sigmoid(collab_logits).squeeze(-1)
        collab_dist = dist.Bernoulli(probs=collab_prob)

        # 4) citation_choices: Categorical(10)
        citation_dist = dist.Categorical(logits=citation_logits)

        # 5) combined_focus: Categorical(10)
        focus_dist = dist.Categorical(logits=focus_logits)

        # 6) timing: Beta again
        timing_sig = torch.sigmoid(timing_logits).squeeze(-1)
        alpha_time = timing_sig * 5.0 + 1.0
        beta_time = (1.0 - timing_sig) * 5.0 + 1.0
        timing_dist = dist.Beta(alpha_time, beta_time)

        return {
            "field_positioning": field_pos_dist,
            "novelty": novelty_dist,
            "collab": collab_dist,
            "citation_choices": citation_dist,
            "combined_focus": focus_dist,
            "timing": timing_dist
        }

    # Batched Action Sampling
    def sample_action(self, policy_outputs) -> 'ARITAction':
        """
        Returns a batched ARITAction. 
        If the batch size is B, each distribution sample has shape [B].
        """
        dists = self.get_policy_distributions(policy_outputs)
        b = dists["field_positioning"].logits.shape[0]

        # field_positioning => categorical => sample => [b]
        field_pos_idx = dists["field_positioning"].sample()  # [b]
        # convert to one-hot => shape [b,25]
        field_pos_oh = F.one_hot(field_pos_idx, num_classes=25).float()

        # novelty => Beta => sample => shape [b]
        novelty_samp = dists["novelty"].sample()

        # collab => Bernoulli => [b]
        collab_samp = dists["collab"].sample()

        # citation_choices => categorical => [b]
        citation_idx = dists["citation_choices"].sample()
        citation_oh = F.one_hot(citation_idx, num_classes=10).float()

        # combined_focus => categorical => [b]
        focus_idx = dists["combined_focus"].sample()
        focus_oh = F.one_hot(focus_idx, num_classes=10).float()

        # timing => Beta => [b]
        timing_samp = dists["timing"].sample()

        return ARITAction(
            field_positioning=field_pos_oh.detach().cpu().numpy(),
            novelty_level=novelty_samp.detach().cpu().numpy(),
            collaboration_strategy=collab_samp.detach().cpu().numpy(),
            citation_choices=citation_oh.detach().cpu().numpy(),
            combined_focus=focus_oh.detach().cpu().numpy(),
            timing=timing_samp.detach().cpu().numpy(),
        )

    # Log Probability of an existing ARITAction
    def compute_action_log_prob(self, policy_outputs, action: 'ARITAction'):
        """
        Compute the combined log probability of each sub-action in the batch.

        policy_outputs => 6 Tensors each shape [b, ...]
        action => ARITAction with shape [b, ...]
        Returns => log_probs => [b], sum of log probs from each subcomponent
        """
        dists = self.get_policy_distributions(policy_outputs)
        b = dists["field_positioning"].logits.shape[0]

        # Helper function to convert to NumPy safely
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return x  # Already NumPy array

        # 1) field_positioning => shape [b,25] one-hot
        field_pos_oh = to_numpy(action.field_positioning)
        field_pos_idx = np.argmax(field_pos_oh, axis=1)
        field_pos_idx_t = torch.tensor(field_pos_idx, device=dists["field_positioning"].logits.device)
        lp_field = dists["field_positioning"].log_prob(field_pos_idx_t)

        # 2) novelty => shape [b], interpret as Beta
        novelty_val = torch.tensor(to_numpy(action.novelty_level), device=lp_field.device, dtype=torch.float32)
        lp_novelty = dists["novelty"].log_prob(novelty_val)

        # 3) collab => shape [b], Bernoulli
        collab_val = torch.tensor(to_numpy(action.collaboration_strategy), device=lp_field.device, dtype=torch.float32)
        lp_collab = dists["collab"].log_prob(collab_val)

        # 4) citation_choices => shape [b,10] one-hot => pick index
        citation_oh = to_numpy(action.citation_choices)
        citation_idx = np.argmax(citation_oh, axis=1)
        citation_idx_t = torch.tensor(citation_idx, device=lp_field.device)
        lp_citation = dists["citation_choices"].log_prob(citation_idx_t)

        # 5) combined_focus => shape [b,10] => pick index
        focus_oh = to_numpy(action.combined_focus)
        focus_idx = np.argmax(focus_oh, axis=1)
        focus_idx_t = torch.tensor(focus_idx, device=lp_field.device)
        lp_focus = dists["combined_focus"].log_prob(focus_idx_t)

        # 6) timing => shape [b], Beta
        timing_val = torch.tensor(to_numpy(action.timing), device=lp_field.device, dtype=torch.float32)
        lp_timing = dists["timing"].log_prob(timing_val)

        combined_lp = lp_field + lp_novelty + lp_collab + lp_citation + lp_focus + lp_timing
        return combined_lp

    # Negative Binomial for Citation Prediction
    def get_citation_distribution(self, citation_params, horizon_idx, debug=False):
        """
        Get the Negative Binomial distribution for a given horizon.
        """
        if debug:
            print("\n----- DEBUG: get_citation_distribution -----")
            print(f"Citation params type: {type(citation_params)}")
            if isinstance(citation_params, tuple):
                print(f"Citation params is a tuple of length {len(citation_params)}")
                if len(citation_params) == 2:
                    params, expert_weights = citation_params
                    print(f"  Params shape: {params.shape}")
                    print(f"  Expert weights shape: {expert_weights.shape}")
                    
                    # Fix: Only print first item or summarize when dealing with batches
                    if expert_weights.shape[0] == 1:
                        print(f"  Expert selection: {torch.argmax(expert_weights, dim=1).item()}")
                        print(f"  Expert probabilities: {expert_weights.detach().cpu().numpy()}")
                    else:
                        top_experts = torch.argmax(expert_weights, dim=1)
                        expert_counts = torch.bincount(top_experts, minlength=expert_weights.shape[1])
                        print(f"  Expert selection counts: {expert_counts.cpu().numpy()}")
                        print(f"  Batch size: {expert_weights.shape[0]}, showing first item:")
                        print(f"  First item expert probs: {expert_weights[0].detach().cpu().numpy()}")
            elif torch.is_tensor(citation_params):
                print(f"  Citation params shape: {citation_params.shape}")

        if isinstance(self.citation_head, MultiHeadCitationPredictor):
            if debug:
                print("  Using MultiHeadCitationPredictor")
            dist, mean = self.citation_head.get_distribution_for_horizon(citation_params, horizon_idx, debug=debug)
            
            # Apply capping to prevent inf after exponential transformation
            if self.use_log_transform:
                # Cap the log-space values to prevent inf after exp transform
                LOG_SPACE_MAX = 6.2  # Corresponds to exp(6.2)-1 ≈ 490 citations
                mean = torch.clamp(mean, max=LOG_SPACE_MAX)
                if debug:
                    print(f"  Applied log-space cap at {LOG_SPACE_MAX}")
                    transformed = np.exp(mean[0].item()) - 1 if mean.shape[0] > 1 else np.exp(mean.item()) - 1
                    print(f"  After capping, will transform to: {transformed:.4f}")
            
            if debug:
                # Check for inf/nan and large values
                if torch.any(torch.isinf(mean)) or torch.any(torch.isnan(mean)):
                    print(f"  WARNING: Inf/NaN detected in mean: {mean}")
                elif mean.max().item() > 50:  # Threshold for log space stability
                    print(f"  WARNING: Large mean detected: max={mean.max().item():.4f}")
                # Print mean distribution info
                if mean.shape[0] == 1:
                    print(f"  Distribution mean: {mean.item():.4f}")
                    transformed = np.exp(mean.item()) - 1 if self.use_log_transform else mean.item()
                    print(f"  Expected value (after transform): {transformed:.4f}")
                    if transformed == float('inf'):
                        print(f"  WARNING: Transformed value is inf")
                else:
                    print(f"  Distribution mean (batch): min={mean.min().item():.4f}, "
                        f"max={mean.max().item():.4f}, avg={mean.mean().item():.4f}")
                    first_mean = mean[0].item()
                    transformed = np.exp(first_mean) - 1 if self.use_log_transform else first_mean
                    print(f"  Expected value for first item: {transformed:.4f}")
                    if transformed == float('inf'):
                        print(f"  WARNING: First transformed value is inf")
            return dist, mean

        elif isinstance(self.citation_head, MoECitationPredictor):
            if debug:
                print("  Using MoECitationPredictor")
            dist, mean = self.citation_head.get_distribution_for_horizon(citation_params, horizon_idx, debug=debug)
            
            # Apply capping to prevent inf after exponential transformation
            if self.use_log_transform:
                # Cap the log-space values to prevent inf after exp transform
                LOG_SPACE_MAX = 6.2  # Corresponds to exp(6.2)-1 ≈ 490 citations
                mean = torch.clamp(mean, max=LOG_SPACE_MAX)
                if debug:
                    print(f"  Applied log-space cap at {LOG_SPACE_MAX}")
                    transformed = np.exp(mean[0].item()) - 1 if mean.shape[0] > 1 else np.exp(mean.item()) - 1
                    print(f"  After capping, will transform to: {transformed:.4f}")
            
            if debug:
                # Check for inf/nan and large values
                if torch.any(torch.isinf(mean)) or torch.any(torch.isnan(mean)):
                    print(f"  WARNING: Inf/NaN detected in mean: {mean}")
                elif mean.max().item() > 50:
                    print(f"  WARNING: Large mean detected: max={mean.max().item():.4f}")
                # Print mean distribution info
                if mean.shape[0] == 1:
                    print(f"  Distribution mean: {mean.item():.4f}")
                    transformed = np.exp(mean.item()) - 1 if self.use_log_transform else mean.item()
                    print(f"  Expected value (after transform): {transformed:.4f}")
                    if transformed == float('inf'):
                        print(f"  WARNING: Transformed value is inf")
                else:
                    print(f"  Distribution mean (batch): min={mean.min().item():.4f}, "
                        f"max={mean.max().item():.4f}, avg={mean.mean().item():.4f}")
                    first_mean = mean[0].item()
                    transformed = np.exp(first_mean) - 1 if self.use_log_transform else first_mean
                    print(f"  Expected value for first item: {transformed:.4f}")
                    if transformed == float('inf'):
                        print(f"  WARNING: First transformed value is inf")
            return dist, mean

        else:
            if debug:
                print("  Using standard citation head")
            
            alpha_idx = horizon_idx
            beta_idx = self.horizons + horizon_idx
            
            if len(citation_params.shape) == 2:
                alpha = citation_params[:, alpha_idx]
                beta = citation_params[:, beta_idx]
            else:
                alpha = citation_params[alpha_idx]
                beta = citation_params[beta_idx]
            
            total_count = F.softplus(alpha) * 1.5 + 1.0
            total_count = torch.clamp(total_count, min=0.1, max=500.0)
            p = torch.sigmoid(beta) * 0.9 + 0.05
            p = torch.clamp(p, min=0.01, max=0.99)
            
            if debug:
                # Check alpha and beta
                if alpha.shape[0] == 1:
                    print(f"  Alpha (raw): {alpha.item():.4f}")
                    print(f"  Beta (raw): {beta.item():.4f}")
                else:
                    print(f"  Alpha (batch): min={alpha.min().item():.4f}, max={alpha.max().item():.4f}")
                    print(f"  Beta (batch): min={beta.min().item():.4f}, max={beta.max().item():.4f}")
                print(f"  Total count: {total_count[0].item():.4f}" if total_count.shape[0] > 1 else f"  Total count: {total_count.item():.4f}")
                print(f"  Probability p: {p[0].item():.4f}" if p.shape[0] > 1 else f"  Probability p: {p.item():.4f}")
            
            nb_dist = torch.distributions.NegativeBinomial(total_count=total_count, probs=p)
            raw_mean = nb_dist.mean
            
            if debug:
                if raw_mean.shape[0] == 1:
                    print(f"  Raw mean: {raw_mean.item():.4f}")
                else:
                    print(f"  Raw mean (batch): min={raw_mean.min().item():.4f}, max={raw_mean.max().item():.4f}")
            
            if self.use_log_transform:
                log_mean = torch.log1p(raw_mean)
                log_mean = torch.clamp(log_mean, min=0.1, max=torch.log1p(torch.tensor(500.0, device=self.device)))
                if debug:
                    if log_mean.shape[0] == 1:
                        print(f"  Log mean (log(1+raw_mean)): {log_mean.item():.4f}")
                        transformed = torch.exp(log_mean).item() - 1
                        print(f"  Will transform back as: exp({log_mean.item():.4f})-1 = {transformed:.4f}")
                        if transformed == float('inf'):
                            print(f"  WARNING: Transformed value is inf")
                    else:
                        print(f"  Log mean (batch): min={log_mean.min().item():.4f}, max={log_mean.max().item():.4f}")
                        first_log_mean = log_mean[0].item()
                        transformed = torch.exp(torch.tensor(first_log_mean)) - 1
                        print(f"  Will transform back (first item): exp({first_log_mean:.4f})-1 = {transformed:.4f}")
                        if transformed == float('inf'):
                            print(f"  WARNING: First transformed value is inf")
                    if torch.any(torch.isinf(log_mean)):
                        print(f"  WARNING: Inf detected in log_mean: {log_mean}")
                return nb_dist, log_mean
            else:
                capped_mean = torch.clamp(raw_mean, min=0.1, max=500.0)
                if debug:
                    if capped_mean.shape[0] == 1:
                        print(f"  Capped mean: {capped_mean.item():.4f}")
                    else:
                        print(f"  Capped mean (batch): min={capped_mean.min().item():.4f}, max={capped_mean.max().item():.4f}")
                    if torch.any(torch.isinf(capped_mean)):
                        print(f"  WARNING: Inf detected in capped_mean: {capped_mean}")
                return nb_dist, capped_mean
                
    def predict_value(self, x):
        with torch.no_grad():
            _, v, _, _ = self.forward(x)
        return v

    def predict_citations(self, x):
        with torch.no_grad():
            _, _, c_params, _ = self.forward(x)
        return c_params
    
class CitationGraphProcessor:
    def __init__(self, citation_network, max_nodes=20, embedding_dim=768):
        """
        Process citation networks into graph structures for the model.
        
        Args:
            citation_network: Dictionary mapping paper IDs to citation data
            max_nodes: Maximum number of nodes to include in each subgraph
            embedding_dim: Dimension of node feature embeddings
        """
        self.citation_network = citation_network
        self.max_nodes = max_nodes
        self.embedding_dim = embedding_dim
        
    def create_subgraph(self, paper_id, embedding_dict):
        """
        Create a subgraph centered around a specific paper.
        
        Args:
            paper_id: ID of the central paper
            embedding_dict: Dictionary mapping paper IDs to embeddings
            
        Returns:
            nodes: Node features [max_nodes, embedding_dim]
            adj_matrix: Adjacency matrix [max_nodes, max_nodes]
            mask: Binary mask for valid nodes [max_nodes]
        """
        if paper_id not in self.citation_network:
            return (
                np.zeros((self.max_nodes, self.embedding_dim), dtype=np.float32),
                np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32),
                np.zeros(self.max_nodes, dtype=np.float32)
            )
        
        # Get references and citations
        references = self.citation_network[paper_id]['references'][:self.max_nodes//2]
        citations = self.citation_network[paper_id]['citations'][:self.max_nodes//2]
        
        # Create neighborhood (center paper + references + citations)
        neighborhood = [paper_id] + references + citations
        # Remove duplicates while preserving order
        neighborhood = list(dict.fromkeys(neighborhood))
        
        # Limit neighborhood size to max_nodes
        if len(neighborhood) > self.max_nodes:
            neighborhood = neighborhood[:self.max_nodes]
        
        # Create mapping from paper ID to local index
        node_mapping = {p_id: i for i, p_id in enumerate(neighborhood)}
        
        # Create node features
        nodes = np.zeros((self.max_nodes, self.embedding_dim), dtype=np.float32)
        for i, p_id in enumerate(neighborhood):
            if i >= self.max_nodes:
                break
                
            if p_id in embedding_dict:
                # Use stored embedding
                nodes[i] = embedding_dict[p_id]
        
        # Create adjacency matrix
        n = len(neighborhood)
        adj_matrix = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        
        # Add edges for references
        for ref_id in references:
            if ref_id in node_mapping:
                # Paper -> Reference
                adj_matrix[0, node_mapping[ref_id]] = 1.0
                # Also add reverse edge for bidirectional attention
                adj_matrix[node_mapping[ref_id], 0] = 1.0
        
        # Add edges for citations
        for cit_id in citations:
            if cit_id in node_mapping:
                # Citation -> Paper
                adj_matrix[node_mapping[cit_id], 0] = 1.0
                # Also add reverse edge for bidirectional attention
                adj_matrix[0, node_mapping[cit_id]] = 1.0
        
        # Create mask for valid nodes
        mask = np.zeros(self.max_nodes, dtype=np.float32)
        mask[:n] = 1.0
        
        return nodes, adj_matrix, mask
    
    def process_batch(self, state_ids, states):
        """
        Process a batch of papers to create graph inputs.
        
        Args:
            state_ids: List of state IDs in the batch
            states: Dictionary mapping state IDs to ARITState objects
            
        Returns:
            node_features: Tensor of shape [batch_size, max_nodes, embedding_dim]
            adj_matrices: Tensor of shape [batch_size, max_nodes, max_nodes]
            masks: Tensor of shape [batch_size, max_nodes]
        """
        batch_size = len(state_ids)
        
        # Create embedding dictionary
        embedding_dict = {}
        for sid in state_ids:
            state = states[sid]
            embedding_dict[sid] = state.content_embedding
        
        # Process each state
        node_features = np.zeros((batch_size, self.max_nodes, self.embedding_dim), dtype=np.float32)
        adj_matrices = np.zeros((batch_size, self.max_nodes, self.max_nodes), dtype=np.float32)
        masks = np.zeros((batch_size, self.max_nodes), dtype=np.float32)
        
        for i, sid in enumerate(state_ids):
            nodes, adj, mask = self.create_subgraph(sid, embedding_dict)
            node_features[i] = nodes
            adj_matrices[i] = adj
            masks[i] = mask
        
        return node_features, adj_matrices, masks
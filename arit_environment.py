import numpy as np
import random
from typing import Dict, List, Optional, Tuple

from arit_types import ARITAction
from arit_model import CitationGraphProcessor

# FieldDynamicsTracker: Tracks field centroids, momentum, and publication counts
# to compute emergent topic signals and saturation.
class FieldDynamicsTracker:
    def __init__(self, history_window: int = 10, embedding_dim: int = 25):
        self.history_window = history_window
        self.embedding_dim = embedding_dim
        # Store historical field centroid vectors from recent states
        self.history: List[np.ndarray] = []
        # Maintain a momentum vector based on recent changes
        self.momentum = np.zeros(embedding_dim, dtype=np.float32)
        # Simulated publication counts for saturation detection
        self.publication_counts: List[int] = []

    def update(self, state: 'ARITState'):
        """
        Update the tracker with the field centroid from a given state.
        """
        field_vector = state.field_centroid.astype(np.float32)
        self.history.append(field_vector)
        if len(self.history) > self.history_window:
            self.history.pop(0)
        # Update momentum as the difference between the two most recent centroids
        if len(self.history) >= 2:
            self.momentum = self.history[-1] - self.history[-2]
        else:
            self.momentum = np.zeros(self.embedding_dim, dtype=np.float32)
        # Update publication counts (simulate a new publication per update)
        self.publication_counts.append(1)
        if len(self.publication_counts) > self.history_window:
            self.publication_counts.pop(0)

    def compute_growth_rate(self) -> float:
        """
        Calculate growth rate as the norm of the momentum vector.
        """
        return float(np.linalg.norm(self.momentum))

    def compute_novelty_signal(self, state: 'ARITState') -> float:
        """
        Calculate novelty as the distance between the state's field centroid
        and the average centroid from recent history.
        """
        if not self.history:
            return 0.0
        avg_centroid = np.mean(np.stack(self.history), axis=0)
        return float(np.linalg.norm(state.field_centroid - avg_centroid))

    def compute_saturation(self) -> float:
        """
        Compute saturation as the ratio of publication counts over the history
        window, capped between 0 and 1.
        """
        total = sum(self.publication_counts)
        return min(total / self.history_window, 1.0)

    def get_emergent_topics(self) -> np.ndarray:
        """
        Return a vector (of length equal to embedding_dim, e.g. 25) that
        signals emergent topics. Here we combine the average novelty and
        growth rate as a simple proxy.
        """
        growth_rate = self.compute_growth_rate()
        if self.history:
            # Calculate average novelty over the history
            avg_novelty = np.mean([
                np.linalg.norm(vec - np.mean(np.stack(self.history), axis=0))
                for vec in self.history
            ])
        else:
            avg_novelty = 0.0
        emergent_value = avg_novelty + growth_rate
        emergent_vector = np.full((self.embedding_dim,), emergent_value, dtype=np.float32)
        return emergent_vector

# StrategicTransitionModel: Predicts next state based on the current state and action.
class StrategicTransitionModel:
    def __init__(self, exploration_rate: float = 0.3, decay_factor: float = 0.99, min_exploration: float = 0.05, action_dim=48, embedding_dim=768):
        self.exploration_rate = exploration_rate
        self.decay_factor = decay_factor
        self.min_exploration = min_exploration
        self.transition_history = []
        # Linear projection from action space to embedding space
        self.projection = np.random.randn(action_dim, embedding_dim).astype(np.float32) * 0.01  # Random init, to be refined later

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def predict_next_state(self, current_state: 'ARITState', action: 'ARITAction', candidate_state_ids: List[int], states: Dict[int, 'ARITState']) -> int:
        action_vector = action.flatten() if not hasattr(action.field_positioning, 'shape') or len(action.field_positioning.shape) == 1 else action.flatten(0)
        
        if random.random() < self.exploration_rate or len(candidate_state_ids) == 0:
            chosen = random.choice(candidate_state_ids) if candidate_state_ids else current_state.state_id
        else:
            action_emb = np.dot(action_vector, self.projection)  # Shape: (768,)
            scores = []
            for cand_id in candidate_state_ids:
                candidate_state = states[cand_id]
                score = self._cosine_similarity(action_emb, candidate_state.content_embedding)
                scores.append(score)
            chosen = candidate_state_ids[int(np.argmax(scores))]
        
        self.exploration_rate = max(self.exploration_rate * self.decay_factor, self.min_exploration)
        self.transition_history.append({
            "current_state": current_state.state_id,
            "action": action_vector.tolist(),
            "chosen_next": chosen,
            "exploration_rate": self.exploration_rate
        })
        return chosen

# ARITTransitions
class ARITTransitions:
    """
    Optionally incorporates strategic transitions using a learned model.
    """
    def __init__(self, states: Dict[int, 'ARITState'], transitions_dict: Dict[int, List[int]]):
        self.states = states
        self.transitions_dict = transitions_dict
        # Initialize the strategic transition model.
        self.transition_model = StrategicTransitionModel(exploration_rate=0.3, decay_factor=0.99, min_exploration=0.05)

    def get_next_state_ids(self, current_state_ids: List[int], actions: List['ARITAction']) -> List[int]:
        next_ids = []
        for s_id, act in zip(current_state_ids, actions):
            possible_next = self.transitions_dict.get(s_id, [])
            if len(possible_next) == 0:
                next_ids.append(s_id)  # Fallback: remain in the current state if no candidate exists.
            else:
                # Use the strategic model to predict the next state.
                current_state = self.states[s_id]
                next_id = self.transition_model.predict_next_state(current_state, act, possible_next, self.states)
                next_ids.append(next_id)
        return next_ids

# ARITState
class ARITState:
    def __init__(
            self,
            content_embedding: np.ndarray,
            field_centroid: np.ndarray,
            reference_diversity: float,
            citation_count: float,
            field_impact_factor: float,
            collaboration_info: float,
            time_index: int,
            state_id: Optional[int] = None,
            future_citations: Optional[List[float]] = None,
            emerging_topics: Optional[np.ndarray] = None,
            field_saturation: float = 0.5,
            strategy_memory: Optional[np.ndarray] = None,
            # New network features
            network_data: Optional[dict] = None,
            primary_category: Optional[str] = None,
            field_target: Optional[int] = None,
    ):
        self.content_embedding = content_embedding
        self.field_centroid = field_centroid
        self.reference_diversity = reference_diversity
        self.citation_count = citation_count
        self.field_impact_factor = field_impact_factor
        self.collaboration_info = collaboration_info
        self.time_index = time_index
        self.state_id = state_id
        self.future_citations = future_citations if future_citations is not None else [0, 0, 0, 0]

        if emerging_topics is None:
            emerging_topics = np.zeros(25, dtype=np.float32)
        if strategy_memory is None:
            strategy_memory = np.zeros(10, dtype=np.float32)

        self.emerging_topics = emerging_topics.astype(np.float32)
        self.field_saturation = np.float32(field_saturation)
        self.strategy_memory = strategy_memory.astype(np.float32)
        
        # Add new fields
        self.network_data = network_data if network_data is not None else {}
        self.primary_category = primary_category
        self.field_target = field_target

    def to_numpy(self) -> np.ndarray:
        # Extract network features
        network_features = np.array([
            self.network_data.get('reference_count_actual', 0) / 100.0,  # Normalize
            self.network_data.get('citation_count_actual', 0) / 100.0,   # Normalize
            self.network_data.get('internal_reference_count', 0) / 50.0, # Normalize
            self.network_data.get('internal_citation_count', 0) / 50.0,  # Normalize
        ], dtype=np.float32)
        
        return np.concatenate([
            self.content_embedding,
            self.field_centroid,
            np.array([
                self.reference_diversity,
                self.citation_count,
                self.field_impact_factor,
                self.collaboration_info,
                float(self.time_index)
            ]),
            self.emerging_topics,
            np.array([self.field_saturation]),
            self.strategy_memory,
            np.array(self.future_citations),
            network_features
        ]).astype(np.float32)

    def update_field_dynamics(self, emerging_topics: np.ndarray, field_saturation: float):
        self.emerging_topics = emerging_topics.astype(np.float32)
        self.field_saturation = np.float32(field_saturation)

# ARITAction
class ARITAction:
    def __init__(
        self,
        field_positioning,
        novelty_level,
        collaboration_strategy,
        citation_choices,
        combined_focus,
        timing
    ):
        self.field_positioning = field_positioning
        self.novelty_level = novelty_level
        self.collaboration_strategy = collaboration_strategy
        self.citation_choices = citation_choices
        self.combined_focus = combined_focus
        self.timing = timing

    def flatten(self, index=0):
        """
        Return a single flattened action if batched. 
        If single instance, flatten everything directly.
        """
        # Check if we have a batch dimension
        if hasattr(self.field_positioning, 'shape') and len(self.field_positioning.shape) == 2:
            fpos = self.field_positioning[index]
            novelty = self.novelty_level[index]
            collab = self.collaboration_strategy[index]
            c_c = self.citation_choices[index]
            comb = self.combined_focus[index]
            time_ = self.timing[index]
        else:
            fpos = self.field_positioning
            novelty = self.novelty_level
            collab = self.collaboration_strategy
            c_c = self.citation_choices
            comb = self.combined_focus
            time_ = self.timing

        return np.concatenate([
            fpos,
            np.array([novelty], dtype=np.float32),
            np.array([collab], dtype=np.float32),
            c_c,
            comb,
            np.array([time_], dtype=np.float32)
        ]).astype(np.float32)

# AdaptiveRewardSystem: Dynamically adjusts reward weights based on
# observed reward patterns and contextual state features.
class AdaptiveRewardSystem:
    def __init__(self, initial_weights: Dict[str, float] = None, adaptation_rate: float = 0.005):
        if initial_weights is None:
            initial_weights = {
                "citation": 1.0,
                "novelty": 1.0,
                "field_impact": 1.0,
                "temporal_consistency": 1.0
            }
        self.weights = initial_weights
        self.adaptation_rate = adaptation_rate
        self.reward_history = []
        self.min_weight = 0.2
        self.max_weight = 2.0

    def update_weights(self, observed_reward: float, context: Dict[str, float]):
        fs = context.get("field_saturation", 0.5)
        es = context.get("emerging_strength", 0.0)
        
        # More stable update with dampening
        new_citation_weight = self.weights["citation"] * (1 - self.adaptation_rate * (fs - es))
        new_novelty_weight = self.weights["novelty"] * (1 + self.adaptation_rate * es)
        new_field_impact_weight = self.weights["field_impact"] * (1 + self.adaptation_rate * (1 - fs))
        new_temporal_consistency_weight = self.weights["temporal_consistency"]
        
        # Bound all weights
        self.weights["citation"] = max(min(new_citation_weight, self.max_weight), self.min_weight)
        self.weights["novelty"] = max(min(new_novelty_weight, self.max_weight), self.min_weight)
        self.weights["field_impact"] = max(min(new_field_impact_weight, self.max_weight), self.min_weight)
        self.weights["temporal_consistency"] = max(min(new_temporal_consistency_weight, self.max_weight), self.min_weight)
        
        # Record history
        self.reward_history.append((observed_reward, context))
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

    def get_adaptive_weights(self, state: 'ARITState') -> Dict[str, float]:
        fs = float(state.field_saturation)
        emerging_strength = float(np.mean(state.emerging_topics))
        
        # More conservative adaptive weight adjustments
        adaptive_weights = {
            "citation": self.weights["citation"] * (1 - 0.2 * fs + 0.2 * emerging_strength),
            "novelty": self.weights["novelty"] * (1 + 0.1 * emerging_strength),
            "field_impact": self.weights["field_impact"] * (1 + 0.1 * (1 - fs)),
            "temporal_consistency": self.weights["temporal_consistency"]
        }
        
        # Bound all weights
        for k in adaptive_weights:
            adaptive_weights[k] = max(min(adaptive_weights[k], self.max_weight), self.min_weight)
            
        return adaptive_weights

# RewardCalculator
class RewardCalculator:
    def __init__(self, config):
        self.config = config
        rw = config.environment_config["reward_weights"]
        self.static_weights = rw
        self.adaptive_system = AdaptiveRewardSystem(initial_weights=rw, adaptation_rate=0.01)
        self.collab_multiplier = config.environment_config["collab_multiplier"]
        self.long_term_decay = config.environment_config["long_term_decay"]
        self.reward_clip_min = config.environment_config["reward_clip_min"]
        self.reward_clip_max = config.environment_config["reward_clip_max"]
        # Horizons for future_citations alignment
        self.horizons = [1, 3, 6, 12]

    def calculate_rewards(self, prev_states, actions, next_states, step_indices):
        batch_size = len(prev_states)
        rewards = np.zeros(batch_size, dtype=np.float32)

        def is_batched(x):
            return hasattr(x, 'shape') and len(x.shape) > 0

        for i in range(batch_size):
            s_next = next_states[i]
            step_idx = step_indices[i]
            
            # Select horizon based on step index
            horizon_idx = min(step_idx // 2, len(self.horizons) - 1)
            
            # Get citation count from network data if available
            network_data = getattr(s_next, 'network_data', {})
            actual_citation_count = network_data.get('citation_count_actual', s_next.citation_count)
            
            # Use the most accurate citation count available
            citation_score = self._compute_citation_reward(s_next, horizon_idx, actual_citation_count)
            
            # Use network-based reference diversity if available
            network_diversity = network_data.get('reference_diversity', s_next.reference_diversity)
            novelty_score = self._compute_novelty_score(
                s_next.content_embedding, 
                s_next.field_centroid, 
                network_diversity
            )
            
            field_impact_score = self._compute_field_impact(s_next.field_impact_factor)
            
            timing_val = float(actions.timing[i] if is_batched(actions.timing) else actions.timing)
            temporal_consistency_score = self._compute_temporal_consistency(timing_val)
            
            # Get adaptive weights
            adaptive_weights = self.adaptive_system.get_adaptive_weights(s_next)
            
            # Calculate network position reward
            network_position_reward = self._compute_network_position_reward(s_next)
            
            # Combine components
            base_reward = (
                adaptive_weights["citation"] * citation_score +
                adaptive_weights["novelty"] * novelty_score +
                adaptive_weights["field_impact"] * field_impact_score +
                adaptive_weights["temporal_consistency"] * temporal_consistency_score +
                network_position_reward  # Add network position reward
            )
            
            # Apply collaboration factor
            collab_factor = (1.0 + self.collab_multiplier * s_next.collaboration_info)
            base_reward *= collab_factor
            
            # Apply long-term decay
            decayed_reward = base_reward * (self.long_term_decay ** step_idx)
            
            # Clip reward
            rewards[i] = np.clip(decayed_reward, self.reward_clip_min, self.reward_clip_max)
            
            # Update adaptive reward system
            context = {
                "field_saturation": float(s_next.field_saturation),
                "emerging_strength": float(np.mean(s_next.emerging_topics))
            }
            self.adaptive_system.update_weights(rewards[i], context)
            
        return rewards

    def _compute_citation_reward(self, state, horizon_idx, actual_citation_count=None):
        """
        Compute citation reward using future_citations or actual citation count.
        
        Args:
            state: Current state
            horizon_idx: Index of horizon to use
            actual_citation_count: Actual citation count from network data
        """
        # Base reward from future citations prediction
        future_citations = state.future_citations[horizon_idx]
        
        # If actual citation count is provided, combine with future predictions
        if actual_citation_count is not None:
            # Use a weighted combination
            combined_count = 0.7 * future_citations + 0.3 * actual_citation_count
        else:
            combined_count = future_citations
            
        # Apply log transformation to handle skewed distribution
        return float(np.log1p(combined_count))
    
    def _compute_network_position_reward(self, state):
        """
        Compute reward based on paper's position in the citation network.
        
        This rewards papers that are well-connected in the citation network.
        """
        network_data = getattr(state, 'network_data', {})
        
        # Get network metrics
        citation_count = network_data.get('citation_count_actual', 0)
        reference_count = network_data.get('reference_count_actual', 0)
        internal_citation_count = network_data.get('internal_citation_count', 0)
        internal_reference_count = network_data.get('internal_reference_count', 0)
        
        # Calculate measures of network centrality
        citation_factor = min(np.log1p(citation_count) / 5.0, 1.0)
        reference_factor = min(np.log1p(reference_count) / 4.0, 0.8)
        
        # Reward for being cited by papers in the dataset (higher quality signal)
        internal_citation_bonus = min(internal_citation_count / max(citation_count, 1), 1.0) * 0.5
        
        # Small reward for citing papers in the dataset (coherence)
        internal_reference_bonus = min(internal_reference_count / max(reference_count, 1), 1.0) * 0.2
        
        # Combine factors
        network_reward = (
            0.5 * citation_factor +
            0.2 * reference_factor + 
            internal_citation_bonus +
            internal_reference_bonus
        )
        
        return network_reward * 0.5  # Scale the overall network reward

    def _compute_novelty_score(self, content_embedding, field_centroid, reference_diversity):
        eps = 1e-8
        dot = np.dot(content_embedding, field_centroid)
        norm_a = np.linalg.norm(content_embedding) + eps
        norm_b = np.linalg.norm(field_centroid) + eps
        cos_sim = dot / (norm_a * norm_b)
        return 0.7 * cos_sim + 0.3 * reference_diversity

    def _compute_field_impact(self, field_impact_factor):
        return field_impact_factor

    def _compute_temporal_consistency(self, timing_val):
        desired_timing = 0.5
        dist = abs(timing_val - desired_timing)
        return float(np.exp(-(dist**2 * 5)))
    
# ARITEnvironment
class ARITEnvironment:
    def __init__(self, config, states, transitions, reward_calculator, citation_network=None):
        self.config = config
        self.states = states
        self.transitions = transitions
        self.reward_calculator = reward_calculator
        self.citation_network = citation_network

        self.max_steps = config.environment_config["max_steps"]
        self.gamma = config.environment_config["gamma"]

        self.current_state_ids = []
        self.current_steps = []

        # Initialize the FieldDynamicsTracker for emergent field detection
        self.field_dynamics_tracker = FieldDynamicsTracker(history_window=10, embedding_dim=25)
        
        # Initialize the citation graph processor if citation network is available
        if self.citation_network is not None:
            self.graph_processor = CitationGraphProcessor(
                citation_network=self.citation_network,
                max_nodes=32,
                embedding_dim=config.model_config["d_model"]
            )

    def reset_batch(self, batch_size):
        self.current_state_ids = []
        self.current_steps = []
        all_ids = list(self.states.keys())
        for _ in range(batch_size):
            sid = random.choice(all_ids)
            self.current_state_ids.append(sid)
            self.current_steps.append(0)
        return [self.states[sid] for sid in self.current_state_ids]

    def step_batch(self, actions):
        batch_size = len(self.current_state_ids)
        prev_states = [self.states[sid] for sid in self.current_state_ids]

        # Create action objects for each state in batch
        from_actions = []
        if batch_size > 1:
            for i in range(batch_size):
                single_action = ARITAction(
                    field_positioning=actions.field_positioning[i],
                    novelty_level=actions.novelty_level[i],
                    collaboration_strategy=actions.collaboration_strategy[i],
                    citation_choices=actions.citation_choices[i],
                    combined_focus=actions.combined_focus[i],
                    timing=actions.timing[i]
                )
                from_actions.append(single_action)
        else:
            from_actions = [actions]

        # Get next states
        next_state_ids = self.transitions.get_next_state_ids(self.current_state_ids, from_actions)
        next_states = [self.states[sid] for sid in next_state_ids]

        # Process citation graph data
        citation_data = None
        if hasattr(self, 'graph_processor'):
            # Create embedding dictionary
            embedding_dict = {}
            for sid in next_state_ids:
                embedding_dict[sid] = self.states[sid].content_embedding
                
            # Process citation graphs
            node_features, adj_matrices, masks = self.graph_processor.process_batch(
                next_state_ids, self.states
            )
            
            citation_data = {
                'node_features': node_features,
                'adj_matrices': adj_matrices,
                'masks': masks
            }

        # Calculate rewards using actual citation data
        rewards = self.reward_calculator.calculate_rewards(prev_states, actions, next_states, self.current_steps)

        # Check if episodes are done
        dones = []
        for i in range(batch_size):
            done = (self.current_steps[i] >= self.max_steps)
            dones.append(done)
            if not done:
                self.current_steps[i] += 1
            else:
                self.current_steps[i] = 0

        # Update current state IDs
        self.current_state_ids = next_state_ids

        # Update field dynamics
        for i in range(batch_size):
            self.field_dynamics_tracker.update(next_states[i])
            emergent_topics = self.field_dynamics_tracker.get_emergent_topics()
            saturation = self.field_dynamics_tracker.compute_saturation()
            next_states[i].update_field_dynamics(emergent_topics, saturation)

        # Update strategy memory
        for i in range(batch_size):
            if not dones[i]:
                new_mem = self._update_strategy_memory(
                    old_mem=next_states[i].strategy_memory,
                    action_vector=actions.flatten(i)[:10] if batch_size > 1 else actions.flatten()[:10]
                )
                next_states[i].strategy_memory = new_mem

        return next_states, rewards, dones, {'citation_data': citation_data}

    def _update_strategy_memory(self, old_mem, action_vector):
        alpha = 0.9
        return (alpha * old_mem + (1 - alpha) * action_vector).astype(np.float32)
    
    def seed(self, seed_value: int):
        random.seed(seed_value)
        np.random.seed(seed_value)

    def get_action_space(self) -> int:
        # For a single action flatten: 25 (field_pos) + 1 (novelty) + 1 (collab) +
        # 10 (citation) + 10 (focus) + 1 (timing) = 48
        return 48

    def get_state_space(self) -> int:
        example_state = next(iter(self.states.values()))
        return len(example_state.to_numpy())

class ImprovedRewardCalculator:
    def __init__(self, config):
        self.config = config
        rw = config.environment_config["reward_weights"]
        
        # Set bounds for reward weights
        self.min_weight = 0.2
        self.max_weight = 2.0
        
        # Initialize bounded weights
        self.static_weights = {
            k: max(min(v, self.max_weight), self.min_weight) 
            for k, v in rw.items()
        }
        
        # More conservative adaptation rate
        self.adaptive_system = AdaptiveRewardSystem(
            initial_weights=self.static_weights, 
            adaptation_rate=0.005  # Reduced from 0.01
        )
        
        # Other parameters
        self.collab_multiplier = config.environment_config["collab_multiplier"]
        self.long_term_decay = max(config.environment_config["long_term_decay"], 0.95)  # Minimum 0.95
        self.reward_clip_min = config.environment_config["reward_clip_min"]
        self.reward_clip_max = config.environment_config["reward_clip_max"]
        self.horizons = [1, 3, 6, 12]
        
        # Reward statistics tracking
        self.reward_stats = {
            "total": [], "citation": [], "novelty": [],
            "field_impact": [], "temporal": [], "network": []
        }

    def calculate_rewards(self, prev_states, actions, next_states, step_indices):
        batch_size = len(prev_states)
        rewards = np.zeros(batch_size, dtype=np.float32)

        def is_batched(x):
            return hasattr(x, 'shape') and len(x.shape) > 0

        citation_rewards = []
        novelty_rewards = []
        field_rewards = []
        temporal_rewards = []
        network_rewards = []

        for i in range(batch_size):
            s_prev = prev_states[i]
            s_next = next_states[i]
            step_idx = step_indices[i]
            
            # Select horizon based on step index
            horizon_idx = min(step_idx // 2, len(self.horizons) - 1)
            
            # ==== CITATION REWARD ====
            # Only use future citations for consistency with training target
            future_citations = s_next.future_citations[horizon_idx]
            citation_score = float(np.log1p(future_citations))
            # Normalize to [0, 1] range assuming log1p(citations) typically < 5
            citation_score = min(citation_score / 5.0, 1.0)
            
            # ==== NOVELTY SCORE ====
            content_embedding = s_next.content_embedding
            field_centroid = s_next.field_centroid
            reference_diversity = s_next.reference_diversity
            
            # Calculate cosine similarity
            eps = 1e-8
            dot = np.dot(content_embedding, field_centroid)
            norm_a = np.linalg.norm(content_embedding) + eps
            norm_b = np.linalg.norm(field_centroid) + eps
            cos_sim = dot / (norm_a * norm_b)
            
            # Add differential component - reward improvement
            if hasattr(s_prev, 'content_embedding'):
                prev_cos_sim = np.dot(s_prev.content_embedding, s_prev.field_centroid) / (
                    (np.linalg.norm(s_prev.content_embedding) + eps) * 
                    (np.linalg.norm(s_prev.field_centroid) + eps)
                )
                cos_improvement = max(0, cos_sim - prev_cos_sim)
                novelty_score = 0.5 * cos_sim + 0.3 * reference_diversity + 0.2 * cos_improvement
            else:
                novelty_score = 0.7 * cos_sim + 0.3 * reference_diversity
                
            # Normalize to [0, 1]
            novelty_score = min(max(novelty_score, 0.0), 1.0)
            
            # ==== FIELD IMPACT ====
            field_impact_score = min(s_next.field_impact_factor, 1.0)
            
            # ==== TEMPORAL CONSISTENCY ====
            timing_val = float(actions.timing[i] if is_batched(actions.timing) else actions.timing)
            # Prefer publications at optimal timing (0.5)
            temporal_score = float(np.exp(-(abs(timing_val - 0.5)**2 * 5)))
            
            # ==== SIMPLIFIED NETWORK REWARD ====
            network_data = getattr(s_next, 'network_data', {})
            citation_count = network_data.get('citation_count_actual', 0)
            internal_citation_count = network_data.get('internal_citation_count', 0)
            
            # Focus on citations and internal citations only
            network_score = min(np.log1p(citation_count) / 5.0, 1.0) * 0.7
            if citation_count > 0:
                internal_ratio = min(internal_citation_count / citation_count, 1.0)
                network_score += internal_ratio * 0.3
            
            # Get adaptive weights with bounds
            weights = self.adaptive_system.get_adaptive_weights(s_next)
            weights = {k: max(min(v, self.max_weight), self.min_weight) 
                      for k, v in weights.items()}
            
            # Store for statistics
            citation_rewards.append(citation_score)
            novelty_rewards.append(novelty_score)
            field_rewards.append(field_impact_score)
            temporal_rewards.append(temporal_score)
            network_rewards.append(network_score)
            
            # Combine with normalized weights
            total_weight = sum(weights.values()) + 0.5  # +0.5 for network
            base_reward = (
                weights["citation"] * citation_score +
                weights["novelty"] * novelty_score +
                weights["field_impact"] * field_impact_score +
                weights["temporal_consistency"] * temporal_score +
                0.5 * network_score
            ) / total_weight  # Normalize by total weight
            
            # Apply collaboration factor
            collab_factor = (1.0 + self.collab_multiplier * s_next.collaboration_info)
            base_reward *= collab_factor
            
            # Use more moderate decay
            decay_factor = max(self.long_term_decay ** step_idx, 0.5)  # Floor at 0.5
            decayed_reward = base_reward * decay_factor
            
            # Add small baseline reward to prevent zero rewards
            decayed_reward = max(decayed_reward, 0.05)
            
            # Clip reward
            rewards[i] = np.clip(decayed_reward, self.reward_clip_min, self.reward_clip_max)
            
            # Update adaptive system
            context = {
                "field_saturation": float(s_next.field_saturation),
                "emerging_strength": float(np.mean(s_next.emerging_topics))
            }
            self.adaptive_system.update_weights(rewards[i], context)
        
        # Update statistics
        self.reward_stats["total"].append(float(np.mean(rewards)))
        self.reward_stats["citation"].append(float(np.mean(citation_rewards)))
        self.reward_stats["novelty"].append(float(np.mean(novelty_rewards)))
        self.reward_stats["field_impact"].append(float(np.mean(field_rewards)))
        self.reward_stats["temporal"].append(float(np.mean(temporal_rewards)))
        self.reward_stats["network"].append(float(np.mean(network_rewards)))
        
        # Keep stats list bounded
        for key in self.reward_stats:
            if len(self.reward_stats[key]) > 1000:
                self.reward_stats[key] = self.reward_stats[key][-1000:]
                
        return rewards
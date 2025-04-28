import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import List
from collections import namedtuple, deque
import torch.nn as nn

from arit_environment import ARITAction
from arit_evaluation import CitationMetrics

Transition = namedtuple("Transition", [
    "state",            # ARITState object
    "state_id",         # int
    "action",           # ARITAction
    "log_prob",         # float or shape [batch]
    "value",            # float
    "reward",           # float
    "done",             # bool
    "citation_target",
    "field_target"      # field classification target
])

class RolloutBuffer:
    def __init__(self):
        self.storage = []

    def add(self, *args):
        self.storage.append(Transition(*args))

    def clear(self):
        self.storage = []

    def __len__(self):
        return len(self.storage)


class PPOTrainer:
    def __init__(self, model, config, optimizer, device='cpu', scaler=None, pretrained=True):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.pretrained = pretrained

        tc = config.training_config
        self.gamma = tc["gamma"]
        self.lmbda = tc["ppo_lambda"]
        self.clip_param = tc["ppo_clip_param"]
        self.entropy_coeff = tc["entropy_weight"]
        self.value_coeff = tc["value_weight"]
        
        # Use config values directly, adjust only if needed
        self.citation_coeff = tc.get("citation_weight", 0.05)  # Default 0.05 if missing
        self.policy_weight = tc.get("policy_weight", 1.0)      # Default 1.0 if missing
        if pretrained:
            self.citation_coeff = min(self.citation_coeff, 0.5)  # Cap at 0.1 post-pretraining, not hard 0.01
        
        print(f"Using citation_coeff={self.citation_coeff}, policy_weight={self.policy_weight}")
        
        self.horizon_weights = tc.get("citation_horizon_weights", [1.0] * 4)
        self.normalize_advantages = tc.get("normalize_advantages", True)
        self.num_horizons = config.model_config["prediction_output_dim"]
        
        # Add gradient clipping value
        self.grad_clip = tc.get("clip_grad_norm", 0.5)
        
        # Optional learning rate warmup
        self.warmup_steps = 0
        self.current_step = 0
        
        # Add step counter for diagnostics
        self.update_count = 0

    def compute_gae(self, rollouts: List[Transition]):
        """
        Generalized Advantage Estimation.
        """
        advantages = []
        returns_ = []
        gae = 0.0
        next_value = 0.0
        
        for i in reversed(range(len(rollouts))):
            if rollouts[i].done:
                next_value = 0.0
                gae = 0.0
            
            delta = rollouts[i].reward + self.gamma * next_value - rollouts[i].value
            gae = delta + self.gamma * self.lmbda * gae
            advantages.insert(0, gae)
            next_value = rollouts[i].value

        for i in range(len(rollouts)):
            returns_.append(rollouts[i].value + advantages[i])

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns_ = torch.tensor(returns_, dtype=torch.float32)

        # Normalize returns and advantages for better training stability
        returns_ = (returns_ - returns_.mean()) / (returns_.std() + 1e-8)
        
        if self.normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        return advantages, returns_

    def ppo_update(self, rollouts: List[Transition], advantages: torch.Tensor, returns_: torch.Tensor):
        self.update_count += 1
        
        # Prepare data
        states_np = [tr.state.to_numpy() for tr in rollouts]
        actions_list = [tr.action for tr in rollouts]
        old_log_probs = torch.tensor([tr.log_prob for tr in rollouts], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([tr.value for tr in rollouts], dtype=torch.float32, device=self.device)
        citation_targets = [tr.citation_target for tr in rollouts if tr.citation_target is not None]
        field_targets = [tr.field_target for tr in rollouts if hasattr(tr, 'field_target') and tr.field_target is not None]
        
        advantages = advantages.to(self.device)
        returns_ = returns_.to(self.device)
        citation_targets_tensor = torch.tensor(citation_targets, dtype=torch.float32, device=self.device) if citation_targets else None
        citation_targets_tensor = torch.log1p(citation_targets_tensor) if citation_targets_tensor is not None else None
        field_targets_tensor = torch.tensor(field_targets, dtype=torch.long, device=self.device) if field_targets else None
        
        state_batch = torch.tensor(np.array(states_np), dtype=torch.float32, device=self.device).unsqueeze(1)
        
        self.model.train()
        log_gradients = (self.update_count % 5 == 0)
        criterion_field = nn.CrossEntropyLoss()
        
        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                policy_outputs, values, c_params, field_logits = self.model(state_batch)
                batched_action = self._combine_actions(actions_list, self.device)
                new_log_probs = self.model.compute_action_log_prob(policy_outputs, batched_action)
                
                # PPO losses
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                dists = self.model.get_policy_distributions(policy_outputs)
                total_entropy = sum(dist_.entropy().mean() for dist_ in dists.values())
                policy_loss = policy_loss - self.entropy_coeff * total_entropy
                
                values = values.squeeze(-1)
                value_pred_clipped = old_values + torch.clamp(values - old_values, -self.clip_param, self.clip_param)
                value_loss = torch.max(F.mse_loss(values, returns_), F.mse_loss(value_pred_clipped, returns_))
                
                # Citation loss
                citation_loss = torch.tensor(0.0, device=self.device)
                if citation_targets_tensor is not None and self.citation_coeff > 0:
                    citation_loss = self._compute_citation_loss(c_params, citation_targets_tensor)
                
                # Field loss
                field_loss = torch.tensor(0.0, device=self.device)
                if field_targets_tensor is not None and len(field_targets_tensor) > 0:
                    field_loss = criterion_field(field_logits[:len(field_targets_tensor)], field_targets_tensor)
                
                # Combined loss with higher citation weight
                total_loss = (
                    self.policy_weight * policy_loss +
                    self.value_coeff * value_loss +
                    10.0 * citation_loss + 
                    self.config.training_config.get("field_weight", 0.3) * field_loss
                )
            
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            if log_gradients:
                self._log_gradient_stats()
            self.scaler.unscale_(self.optimizer)
            self._clip_grad_value(self.model.parameters(), clip_value=100)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        else:
            policy_outputs, values, c_params, field_logits = self.model(state_batch)
            batched_action = self._combine_actions(actions_list, self.device)
            new_log_probs = self.model.compute_action_log_prob(policy_outputs, batched_action)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            dists = self.model.get_policy_distributions(policy_outputs)
            total_entropy = sum(dist_.entropy().mean() for dist_ in dists.values())
            policy_loss = policy_loss - self.entropy_coeff * total_entropy
            
            values = values.squeeze(-1)
            value_pred_clipped = old_values + torch.clamp(values - old_values, -self.clip_param, self.clip_param)
            value_loss = torch.max(F.mse_loss(values, returns_), F.mse_loss(value_pred_clipped, returns_))
            
            citation_loss = torch.tensor(0.0, device=self.device)
            if citation_targets_tensor is not None and self.citation_coeff > 0:
                citation_loss = self._compute_citation_loss(c_params, citation_targets_tensor)
            
            field_loss = torch.tensor(0.0, device=self.device)
            if field_targets_tensor is not None and len(field_targets_tensor) > 0:
                field_loss = criterion_field(field_logits[:len(field_targets_tensor)], field_targets_tensor)
            
            total_loss = (
                self.policy_weight * policy_loss +
                self.value_coeff * value_loss +
                10.0 * citation_loss +
                self.config.training_config.get("field_weight", 0.3) * field_loss
            )
            
            self.optimizer.zero_grad()
            total_loss.backward()
            if log_gradients:
                self._log_gradient_stats()
            self._clip_grad_value(self.model.parameters(), clip_value=100)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        
        self.current_step += 1
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "citation_loss": citation_loss.item(),
            "field_loss": field_loss.item(),
            "entropy": total_entropy.item(),
            "total_loss": total_loss.item()
        }
                
    # In PPOTrainer._compute_citation_loss
    def _compute_citation_loss(self, c_params, citation_targets):
        citation_loss = torch.tensor(0.0, device=self.device)
        
        if citation_targets is None or len(citation_targets) == 0:
            return citation_loss
        
        for h in range(self.num_horizons):
            nb_dist, mean = self.model.get_citation_distribution(c_params, h)
            target_h = citation_targets[:, h]
            
            # Huber loss
            diff = mean - target_h
            huber_loss = torch.where(
                torch.abs(diff) < 1.0,
                0.5 * diff * diff,
                torch.abs(diff) - 0.5
            )
            
            horizon_loss = huber_loss.mean()
            citation_loss = citation_loss + horizon_loss * self.horizon_weights[h]
        
        # Remove extra scaling to give citation loss more weight
        citation_loss = citation_loss / len(self.horizon_weights)
        return citation_loss
        
    def _log_gradient_stats(self):
        """Log gradient statistics for debugging"""
        # Only log for certain parameters to avoid spamming
        component_groups = {
            'transformer': [],
            'policy': [],
            'value': [],
            'citation': []
        }
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                
                # Categorize by component
                component = None
                if 'transformer' in name:
                    component = 'transformer'
                elif 'policy' in name:
                    component = 'policy'
                elif 'value' in name:
                    component = 'value'
                elif 'citation' in name:
                    component = 'citation'
                
                if component:
                    component_groups[component].append((grad_norm, param_norm))
        
        # Log summary statistics
        print("\n--- Gradient Statistics ---")
        for component, values in component_groups.items():
            if values:
                grad_norms = [v[0] for v in values]
                param_norms = [v[1] for v in values]
                
                avg_grad = sum(grad_norms) / len(grad_norms)
                max_grad = max(grad_norms) if grad_norms else 0
                
                avg_param = sum(param_norms) / len(param_norms) if param_norms else 1
                grad_param_ratio = avg_grad / avg_param if avg_param > 0 else 0
                
                print(f"{component.upper()}: Avg Grad: {avg_grad:.6f}, Max: {max_grad:.6f}, Ratio: {grad_param_ratio:.6f}")
                
                # Warnings
                if max_grad > 10.0:
                    print(f"  WARNING: Large gradients in {component}")
                if max_grad < 1e-6 and avg_grad < 1e-6:
                    print(f"  WARNING: Vanishing gradients in {component}")
        print("-------------------------\n")
            
    def _combine_actions(self, actions_list, device):
        """
        Combine a list of ARITAction (each single) into a single batched ARITAction.
        We'll assume len(actions_list) = B.
        """
        B = len(actions_list)

        # field_positioning => [B,25]
        field_pos = []
        novelty = []
        collab = []
        citation_c = []
        focus_c = []
        timing = []

        for act in actions_list:
            field_pos.append(act.field_positioning if act.field_positioning.ndim == 1 else act.field_positioning[0])
            novelty.append(act.novelty_level if np.ndim(act.novelty_level) == 0 else act.novelty_level[0])
            collab.append(act.collaboration_strategy if np.ndim(act.collaboration_strategy) == 0 else act.collaboration_strategy[0])
            citation_c.append(act.citation_choices if act.citation_choices.ndim == 1 else act.citation_choices[0])
            focus_c.append(act.combined_focus if act.combined_focus.ndim == 1 else act.combined_focus[0])
            timing.append(act.timing if np.ndim(act.timing) == 0 else act.timing[0])

        field_pos = np.array(field_pos, dtype=np.float32)       # [B,25]
        novelty = np.array(novelty, dtype=np.float32)           # [B]
        collab = np.array(collab, dtype=np.float32)             # [B]
        citation_c = np.array(citation_c, dtype=np.float32)     # [B,10]
        focus_c = np.array(focus_c, dtype=np.float32)           # [B,10]
        timing = np.array(timing, dtype=np.float32)             # [B]

        return ARITAction(
            field_positioning=torch.tensor(field_pos, device=device),
            novelty_level=torch.tensor(novelty, device=device),
            collaboration_strategy=torch.tensor(collab, device=device),
            citation_choices=torch.tensor(citation_c, device=device),
            combined_focus=torch.tensor(focus_c, device=device),
            timing=torch.tensor(timing, device=device)
        )
    
    def _clip_grad_value(self, parameters, clip_value=10):
        """Clips gradient values in addition to norm clipping"""
        for p in filter(lambda p: p.grad is not None, parameters):
            p.grad.data.clamp_(-clip_value, clip_value)

# ARITTrainer for environment loop, validation, early stopping, logging
class ARITTrainer:
    def __init__(self, model, config, env, optimizer, scheduler=None, device='cpu', writer=None, scaler=None, 
                 citation_targets=None, field_targets=None, val_states=None, val_citation_targets=None, 
                 val_field_targets=None, use_log_transform=True):
        self.model = model.to(device)
        self.config = config
        self.env = env
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = writer
        self.scaler = scaler
        self.citation_targets = citation_targets
        self.field_targets = field_targets  # Add field targets
        self.use_log_transform = use_log_transform
        
        self.ppotrainer = PPOTrainer(model, config, optimizer, device=device, scaler=scaler)
        self.num_epochs = config.training_config["num_epochs"]
        self.batch_size = config.training_config["batch_size"]
        self.evaluation_interval = config.training_config["evaluation_interval"]
        self.patience = config.training_config["patience"]
        
        self.grad_accum_steps = config.training_config.get("gradient_accumulation_steps", 1)
        self.use_mixed_precision = config.training_config.get("use_mixed_precision", False)

        self.metrics_history = {
            "policy_loss": [],
            "value_loss": [],
            "citation_loss": [],
            "field_loss": [],  # Add field loss tracking
            "entropy": [],
            "total_loss": [],
            "val_citation_mae": [],
            "val_citation_rmse": [],
            "val_field_acc": []  # Add field accuracy tracking
        }

        self.best_val_metric = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None
        
        self.val_states = val_states
        self.val_citation_targets = val_citation_targets
        self.val_field_targets = val_field_targets  # Add validation field targets

    def collect_trajectories(self, steps=2000):
        rollout = RolloutBuffer()
        states = self.env.reset_batch(self.batch_size)
        step_count = 0

        while step_count < steps:
            # Convert states to tensors
            state_np = [s.to_numpy() for s in states]
            state_tensor = torch.tensor(np.array(state_np), dtype=torch.float32, device=self.device).unsqueeze(1)

            with torch.no_grad():
                # Forward pass through model
                policy_out, values, citation_params, field_logits = self.model(state_tensor)
                action_batched = self.model.sample_action(policy_out)
                lp_batched = self.model.compute_action_log_prob(policy_out, action_batched)
                lp_batched_np = lp_batched.cpu().numpy()
                val_batched_np = values.squeeze(-1).cpu().numpy()

            # Step the environment
            next_states, rewards, dones, info = self.env.step_batch(action_batched)
            
            # Extract citation data from info
            citation_data = info.get('citation_data', None)
            
            # If citation data is available, convert to tensors
            if citation_data is not None:
                citation_data = {
                    'node_features': torch.tensor(citation_data['node_features'], 
                                               dtype=torch.float32, device=self.device),
                    'adj_matrices': torch.tensor(citation_data['adj_matrices'], 
                                              dtype=torch.float32, device=self.device),
                    'masks': torch.tensor(citation_data['masks'], 
                                       dtype=torch.float32, device=self.device)
                }

            # Add transitions to rollout buffer
            for i in range(self.batch_size):
                state_id = states[i].state_id
                citation_target = self.citation_targets.get(state_id, None)
                field_target = self.field_targets.get(state_id, None)
                
                rollout.add(
                    states[i],
                    state_id,
                    self._single_action(action_batched, i),
                    lp_batched_np[i],
                    val_batched_np[i],
                    rewards[i],
                    dones[i],
                    citation_target,
                    field_target
                )

            # Reset if any episodes are done
            if any(dones):
                states = self.env.reset_batch(self.batch_size)
            else:
                states = next_states

            step_count += self.batch_size

        return rollout
    
    def _single_action(self, batched_action, idx):
        if isinstance(batched_action.field_positioning, torch.Tensor):
            field_pos = batched_action.field_positioning[idx].cpu().numpy()
        else:
            field_pos = batched_action.field_positioning[idx]
            
        if isinstance(batched_action.novelty_level, torch.Tensor):
            novelty = batched_action.novelty_level[idx].cpu().item()
        else:
            novelty = float(batched_action.novelty_level[idx])
            
        if isinstance(batched_action.collaboration_strategy, torch.Tensor):
            collab = batched_action.collaboration_strategy[idx].cpu().item()
        else:
            collab = float(batched_action.collaboration_strategy[idx])
        
        if isinstance(batched_action.citation_choices, torch.Tensor):
            citation_ch = batched_action.citation_choices[idx].cpu().numpy()
        else:
            citation_ch = batched_action.citation_choices[idx]
        
        if isinstance(batched_action.combined_focus, torch.Tensor):
            focus_ch = batched_action.combined_focus[idx].cpu().numpy()
        else:
            focus_ch = batched_action.combined_focus[idx]
            
        if isinstance(batched_action.timing, torch.Tensor):
            tim = batched_action.timing[idx].cpu().item()
        else:
            tim = float(batched_action.timing[idx])
        
        return ARITAction(field_pos, novelty, collab, citation_ch, focus_ch, tim)

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.num_epochs):
            # Collect trajectories
            rollouts = self.collect_trajectories(steps=1000)
            advantages, returns_ = self.ppotrainer.compute_gae(rollouts.storage)
            
            # Perform PPO update with all losses included
            metrics = self.ppotrainer.ppo_update(rollouts.storage, advantages, returns_)
            
            # Step the scheduler if provided
            if self.scheduler is not None:
                prev_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}: Learning Rate = {current_lr:.8f}")
            
            # Record metrics in history
            for k, v in metrics.items():
                self.metrics_history[k].append(v)
            
            # Evaluation and early stopping
            if (epoch + 1) % self.evaluation_interval == 0:
                val_mae, val_rmse, val_field_acc = self.validate_citation()
                self.metrics_history["val_citation_mae"].append(val_mae)
                self.metrics_history["val_citation_rmse"].append(val_rmse)
                self.metrics_history["val_field_acc"].append(val_field_acc)
                
                # Log to tensorboard if available
                if self.writer is not None:
                    self.writer.add_scalar("val/mae", val_mae, epoch)
                    self.writer.add_scalar("val/rmse", val_rmse, epoch)
                    self.writer.add_scalar("val/field_acc", val_field_acc, epoch)
                
                # Early stopping check
                if val_rmse < self.best_val_metric:
                    self.best_val_metric = val_rmse
                    self.epochs_no_improve = 0
                    self.best_model_state = self.model.state_dict()
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Log metrics
            if self.writer is not None:
                for k, v in metrics.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch)
            
            print(f"Epoch {epoch+1}/{self.num_epochs} => {metrics}")

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.metrics_history
    
    def validate_citation(self):
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_field_preds = []
        all_field_targets = []
        
        # Debugging variables
        pred_magnitudes = []
        
        # Batch processing
        batch_size = 32
        state_ids = list(self.val_citation_targets.keys())
        
        with torch.no_grad():
            for i in range(0, len(state_ids), batch_size):
                batch_ids = state_ids[i:i + batch_size]
                states = [self.val_states[sid].to_numpy() for sid in batch_ids]
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
                for j, sid in enumerate(batch_ids):
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
        print(f"Number of predictions: {len(all_predictions)}")
        print(f"Prediction stats: min={min(pred_magnitudes):.4f}, max={max(pred_magnitudes):.4f}, "
            f"mean={sum(pred_magnitudes)/len(pred_magnitudes):.4f}")
        
        # Sample predictions vs targets
        print("\nSample predictions vs targets:")
        for i in range(min(5, len(all_predictions))):
            print(f"  Sample {i+1}:")
            print(f"    Predictions: {[round(p, 4) for p in all_predictions[i]]}")
            print(f"    Targets: {all_targets[i]}")
        
        # Compute metrics
        predictions_tensor = torch.tensor(all_predictions, device=self.device)
        targets_tensor = torch.tensor(all_targets, device=self.device)
        
        diff = predictions_tensor - targets_tensor
        mae = torch.mean(torch.abs(diff)).item()
        rmse = torch.sqrt(torch.mean(diff**2)).item()
        field_acc = sum(p == t for p, t in zip(all_field_preds, all_field_targets)) / len(all_field_targets)
        
        print(f"Final metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, Field Acc: {field_acc:.4f}")
        return mae, rmse, field_acc

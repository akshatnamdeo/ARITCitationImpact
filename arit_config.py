from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class ARITConfig:
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "d_model": 256,
        "n_heads": 4,
        "num_layers": 4,
        "dim_feedforward": 2048,
        "dropout": 0.2,
        "activation": "gelu",
        "layer_norm_eps": 1e-5,
        "max_sequence_length": 512,
        "temporal_window": 24,
        "value_hidden_dim": 256,
        "prediction_hidden_dim": 256,
        "prediction_output_dim": 4
    })

    training_config: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 16,
        "num_epochs": 50,
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_steps": 1000,
        "clip_grad_norm": 1.0,
        "policy_weight": 1.0,
        "value_weight": 0.5,
        "citation_weight": 0.5,
        "field_weight": 0.3,
        "entropy_weight": 0.05,
        "checkpoint_interval": 1000,
        "evaluation_interval": 1,
        "patience": 30,
        "seed": 42,
        "ppo_clip_param": 0.1,
        "ppo_lambda": 0.95,
        "gamma": 0.99,
        "normalize_advantages": True,
        "citation_horizon_weights": [1.0, 1.0, 1.0, 1.0],
    })

    environment_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_steps": 8,
        "gamma": 0.99,
        "reward_weights": {
            "citation": 0.60,
            "novelty": 0.15,
            "field_impact": 0.20,
            "temporal_consistency": 0.05
        },
        "collab_multiplier": 0.10,
        "long_term_decay": 0.95,
        "reward_clip_min": -10.0,
        "reward_clip_max": 10.0,
        "vectorized": True,
    })

    evaluation_config: Dict[str, Any] = field(default_factory=lambda: {
        "metrics": ["mae", "rmse", "r2", "spearman"],
        "num_seeds": 1
    })

    def validate(self):
        assert self.model_config["d_model"] > 0, "d_model must be positive."
        assert 0 < self.environment_config["gamma"] <= 1, "Gamma must be within (0,1]."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_config": self.model_config,
            "training_config": self.training_config,
            "environment_config": self.environment_config,
            "evaluation_config": self.evaluation_config
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            model_config=data.get("model_config", {}),
            training_config=data.get("training_config", {}),
            environment_config=data.get("environment_config", {}),
            evaluation_config=data.get("evaluation_config", {})
        )

    def merge(self, other: 'ARITConfig'):
        for section in ["model_config", "training_config", "environment_config", "evaluation_config"]:
            self_section = getattr(self, section)
            other_section = getattr(other, section)
            for k, v in other_section.items():
                self_section[k] = v
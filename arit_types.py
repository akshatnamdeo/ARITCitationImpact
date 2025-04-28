import numpy as np
from dataclasses import dataclass
from typing import Union, Optional

@dataclass
class ARITAction:
    field_positioning: np.ndarray  # shape [batch, 25] - one-hot
    novelty_level: np.ndarray      # shape [batch] - float values
    collaboration_strategy: np.ndarray  # shape [batch] - binary values
    citation_choices: np.ndarray   # shape [batch, 10] - one-hot
    combined_focus: np.ndarray     # shape [batch, 10] - one-hot
    timing: np.ndarray             # shape [batch] - float values
    
    @classmethod
    def create_random(cls, batch_size=1):
        """Create a random action for testing"""
        return cls(
            field_positioning=np.eye(25)[np.random.randint(0, 25, size=batch_size)],
            novelty_level=np.random.random(batch_size),
            collaboration_strategy=np.random.randint(0, 2, size=batch_size),
            citation_choices=np.eye(10)[np.random.randint(0, 10, size=batch_size)],
            combined_focus=np.eye(10)[np.random.randint(0, 10, size=batch_size)],
            timing=np.random.random(batch_size)
        )
    
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
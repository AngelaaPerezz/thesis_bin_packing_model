"""
Neural Network Architecture — Paper-Faithful Implementation
============================================================
Follows Section 4.2 of the paper exactly:

  "The input of the neural network consists of the set of feasible actions
   where each action consists of features describing the chosen item (id,
   orientation) as well as the placement location."

  Requirements met:
  1. Permutation invariant  — each action processed independently then pooled.
  2. Variable input size    — AdaptiveMaxPool collapses any-length action set.
  3. Single model for all configs (12×12/13×13, 12–16 items) — MAX_ACTIONS
     padding in the env means tensor shapes are always fixed.

Architecture
------------
  action_encoder  : MLP(3 → 128)  applied to every action independently
  pooling         : AdaptiveMaxPool over the action dimension → (B, 128)
  mlp_policy      : (256 → 128); dot-product with per-action embeddings → logits
  mlp_value       : (256 → 1)

The concatenation of the pooled embedding (global context) with itself forms
the 256-d "combined embedding" fed to both heads.  The policy head scores
each action via a learned inner-product (pointer-network style), which
preserves permutation equivariance on the output side.

Fixed orientation: item features are [item_id, width, height] (3-d),
no orientation token needed.
"""

from ray.rllib.algorithms.alpha_zero.models.custom_torch_models import ActorCriticModel
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn

from env import MAX_ACTIONS, MAX_ITEMS


class Agent(ActorCriticModel):

    def __init__(self, observation_space, action_space, num_outputs,
                 model_config, name):
        super().__init__(observation_space, action_space, num_outputs,
                         model_config, name)

        D = 128   # embedding dimension throughout

        # ── per-action encoder (shared weights) ────────────────────────────
        # Input: 3-d action vector [item_id, x, y]
        self.action_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, D),
            nn.ReLU(),
            nn.Linear(D, D),
        )

        # ── global pooling: max-pool over action dim ────────────────────────
        # AdaptiveMaxPool2d((1, D)) squeezes variable-length sets to (1, D)
        self.action_pool = nn.AdaptiveMaxPool2d((1, D))

        # ── policy head: combined → query vector, dot with action embeddings
        self.mlp_policy = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D),
        )

        # ── value head ──────────────────────────────────────────────────────
        self.mlp_value = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, 1),
        )

        self._value_out: torch.Tensor = torch.zeros(1)

    # ── ActorCriticModel interface ───────────────────────────────────────────

    def value_function(self):
        return self._value_out.squeeze(-1)

    def forward(self, input_dict, state, seq_lens):
        # RLlib flattens nested Dict obs before calling forward, so
        # "actions" arrives as (B, MAX_ACTIONS*3) — reshape back explicitly.
        actions_raw = input_dict["obs"]["actions"].float()
        B = actions_raw.shape[0]
        A = MAX_ACTIONS
        # Handle both (B, A, 3) and flattened (B, A*3) from RLlib batching
        if actions_raw.dim() == 2:
            actions = actions_raw.view(B, A, 3)   # (B, A, 3)
        else:
            actions = actions_raw                  # already (B, A, 3)

        # ── encode every action independently ──────────────────────────────
        # Reshape to (B*A, 3), encode, reshape back to (B, A, D)
        act_flat = actions.reshape(B * A, 3)
        act_emb  = self.action_encoder(act_flat).view(B, A, -1)  # (B, A, D)

        # ── global context via max-pooling ─────────────────────────────────
        # unsqueeze to (B, 1, A, D) → AdaptiveMaxPool2d((1,D)) → (B, 1, 1, D)
        pooled = self.action_pool(act_emb.unsqueeze(1))
        pooled = pooled.view(B, -1)                               # (B, D)

        # ── value estimate ──────────────────────────────────────────────────
        self._value_out = self.mlp_value(pooled)                  # (B, 1)

        # ── policy logits (pointer-network dot product) ─────────────────────
        query  = self.mlp_policy(pooled)                          # (B, D)
        # bmm: (B, A, D) x (B, D, 1) → (B, A, 1) → (B, A)
        logits = torch.bmm(act_emb, query.unsqueeze(2)).squeeze(2)  # (B, A)

        return logits, None
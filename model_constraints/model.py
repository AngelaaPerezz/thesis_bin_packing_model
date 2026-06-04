from ray.rllib.algorithms.alpha_zero.models.custom_torch_models import ActorCriticModel

from gym import spaces
import numpy as np
import torch

from typing import Dict, List, Optional, Tuple, Type, Union


class Agent(ActorCriticModel):

    # Must match env.py constants
    MAX_ITEMS   = 16
    MAX_BIN_DIM = 13
    DIM         = 2
    N_ACTIONS   = MAX_ITEMS * MAX_BIN_DIM * MAX_BIN_DIM  # 2704
    STATES_SIZE = MAX_ITEMS * (2 * DIM + 1)              # 80
    ACTIONS_SIZE = N_ACTIONS * 3                          # 8112
    MASK_SIZE   = N_ACTIONS                               # 2704
    # Total obs size: 80 + 8112 + 2704 = 10896

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        self.state_encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.STATES_SIZE, 128),  # 80 -> 128
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
        )

        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 64),   # (item, x, y) — 3 inputs
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
        )

        self.action_pool = torch.nn.AdaptiveMaxPool2d((1, 128))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
        )

        self.mlp_value = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

        # Xavier initialization for stability
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def value_function(self):
        return self._value_out.squeeze(-1)

    def custom_loss(self, policy_loss, loss_inputs):
        action_logits = loss_inputs['action_dist_inputs']
        mcts_policies = loss_inputs['mcts_policies']
        value_label   = loss_inputs['value_label']

        values = self.value_function()
        value_loss = torch.mean((values - value_label.float()) ** 2)

        log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
        log_probs = torch.clamp(log_probs, min=-10.0)
        policy_loss_stable = -torch.mean(
            torch.sum(mcts_policies.float() * log_probs, dim=-1)
        )

        total_loss = policy_loss_stable + value_loss
        return [total_loss]

    def forward(self, input_dict, state, seq_lens):
        # input_dict can be a tensor (from MCTS) or a dict (from rollout)
        if isinstance(input_dict, torch.Tensor):
            obs = input_dict.float()
        elif 'obs_flat' in input_dict:
            obs = input_dict['obs_flat'].float()
        else:
            obs = input_dict['obs'].float()

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        s = self.STATES_SIZE
        a = s + self.ACTIONS_SIZE

        states      = obs[:, :s].reshape(-1, self.MAX_ITEMS, 2 * self.DIM + 1)
        actions     = obs[:, s:a].reshape(-1, self.N_ACTIONS, 3)
        action_mask = obs[:, a:]

        state_embedding  = self.state_encoder(states)
        action_embedding = self.action_encoder(actions)

        action_pool = self.action_pool(action_embedding).reshape(-1, action_embedding.shape[-1])
        embedding   = torch.hstack((state_embedding, action_pool))

        final_embedding = self.mlp(embedding)
        action_out = torch.matmul(
            action_embedding,
            final_embedding.reshape(-1, action_embedding.shape[-1], 1)
        )[:, :, 0]

        # Mask invalid actions
        action_out = torch.where(
            action_mask.bool(),
            action_out,
            torch.full_like(action_out, -1e2)
        )

        self._value_out = self.mlp_value(embedding)

        return action_out, None
from ray.rllib.algorithms.alpha_zero.models.custom_torch_models import ActorCriticModel

from gym import spaces
import numpy as np
import torch

from typing import Dict, List, Optional, Tuple, Type, Union


class Agent(ActorCriticModel):
        
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs,
        model_config,
        name,
    ):
        super().__init__(
            observation_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )

        self.state_encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(80, 128),  # 16 items * 5 features
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
            )

        self.action_encoder = torch.nn.Sequential(
                torch.nn.Linear(2, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 128),
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
        
    def value_function(self):
        # Devuelve el valor estimado calculado en forward
        return self._value_out.squeeze(-1)

    # Constantes que deben coincidir con env.py
    MAX_ITEMS   = 16
    MAX_BIN_DIM = 13
    DIM         = 2
    STATES_SIZE      = MAX_ITEMS * (2 * DIM + 1)        # 80
    ACTIONS_SIZE     = MAX_ITEMS * MAX_BIN_DIM * 2      # 416
    ACTION_MASK_SIZE = MAX_ITEMS * MAX_BIN_DIM           # 208

    def forward(self, input_dict, state, seq_lens):
        # Durante MCTS (compute_priors_and_value) input_dict ES el tensor de obs.
        # Durante rollout normal es un dict con clave 'obs_flat' o 'obs'.
        if isinstance(input_dict, torch.Tensor):
            obs = input_dict.float()
        elif 'obs_flat' in input_dict:
            obs = input_dict['obs_flat'].float()
        else:
            obs = input_dict['obs'].float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # añadir dimensión batch si falta

        # Reconstruir los tres componentes via slicing
        s = self.STATES_SIZE
        a = s + self.ACTIONS_SIZE

        states  = obs[:, :s].reshape(-1, self.MAX_ITEMS, 2 * self.DIM + 1)  # (B, 16, 5)
        actions = obs[:, s:a].reshape(-1, self.MAX_ITEMS * self.MAX_BIN_DIM, 2)  # (B, 208, 2)

        state_emdedding = self.state_encoder(states)
        action_embedding = self.action_encoder(actions)

        action_pool = self.action_pool(action_embedding).reshape(-1, action_embedding.shape[-1])

        embedding = torch.hstack((state_emdedding, action_pool))
        final_embedding = self.mlp(embedding)

        action_out = torch.matmul(action_embedding, final_embedding.reshape(-1, action_embedding.shape[-1], 1))[:, :, 0]

        self._value_out = self.mlp_value(embedding)

        return action_out, None
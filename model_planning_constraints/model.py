import torch
from ray.rllib.algorithms.alpha_zero.models.custom_torch_models import ActorCriticModel


class Agent(ActorCriticModel):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        cfg = model_config.get("custom_model_config", {})

        self.state_dim = cfg.get("state_dim", 100)
        self.action_dim = cfg.get("action_dim", 3)

        # ---------------- STATE ----------------
        self.state_encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
        )

        # ---------------- ACTION ----------------
        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.action_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
        )

        self.action_proj = torch.nn.Linear(128, 128)

        # ---------------- COMBINED ----------------
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, input_dict, state, seq_lens):

        states = input_dict["obs"]["states"].float()
        actions = input_dict["obs"]["actions"].float()
        mask = input_dict["action_mask"].float()

        # state encoding
        state_emb = self.state_encoder(states)

        # action encoding
        action_emb = self.action_encoder(actions)
        action_emb = action_emb * mask.unsqueeze(-1)

        # pool over actions
        pooled = action_emb.max(dim=1)[0]
        pooled = self.action_proj(pooled)

        # combine
        emb = torch.cat([state_emb, pooled], dim=-1)

        hidden = self.mlp(emb)

        logits = torch.sum(action_emb * hidden.unsqueeze(1), dim=-1)

        self._value_out = self.value_head(emb)

        return logits, []
    
    
    def value_function(self):
        return self._value_out.squeeze(1)
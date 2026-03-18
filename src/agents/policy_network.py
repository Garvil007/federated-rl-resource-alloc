import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Actor-Critic policy network for resource allocation.

    Outputs both action logits (actor) and a scalar state value (critic)
    for advantage-based variance reduction during training.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()

        # Shared feature extractor
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)

        # Actor head  (policy logits)
        self.actor = nn.Linear(hidden_dim // 2, act_dim)

        # Critic head (state value)
        self.critic = nn.Linear(hidden_dim // 2, 1)

        # Orthogonal initialization for better RL convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Smaller init for output heads
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

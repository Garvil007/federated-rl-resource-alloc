import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm for stable federated training."""

    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return x + F.relu(self.ln(self.fc(x)))


class TaskAttention(nn.Module):
    """Lightweight attention: lets the network weigh which node features
    matter most for the current set of pending tasks."""

    def __init__(self, node_feat_dim, task_feat_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(task_feat_dim, hidden_dim)
        self.key = nn.Linear(node_feat_dim, hidden_dim)
        self.value = nn.Linear(node_feat_dim, hidden_dim)
        self.scale = hidden_dim**0.5

    def forward(self, node_feats, task_feats):
        """
        node_feats: [batch, num_nodes, node_dim]
        task_feats: [batch, num_tasks, task_dim]
        returns:    [batch, context_dim]
        """
        q = self.query(task_feats)  # [B, T, H]
        k = self.key(node_feats)  # [B, N, H]
        v = self.value(node_feats)  # [B, N, H]

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.scale  # [B, T, N]
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, v)  # [B, T, H]

        # Pool across tasks
        return context.mean(dim=1)  # [B, H]


class PolicyNetwork(nn.Module):
    """Actor-Critic policy network with residual connections,
    task-aware attention, and a decoupled critic backbone.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=256,
        max_nodes=10,
        node_feat_dim=4,
        task_feat_dim=3,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feat_dim = node_feat_dim
        self.task_feat_dim = task_feat_dim
        self.num_node_feats = max_nodes * node_feat_dim
        self.num_task_feats = obs_dim - self.num_node_feats

        # Shared feature extractor with residual connections
        self.fc_in = nn.Linear(obs_dim, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)

        # Task-aware attention
        attn_dim = hidden_dim // 4
        self.attention = TaskAttention(node_feat_dim, task_feat_dim, attn_dim)

        # Actor head (combines residual features + attention context)
        actor_in_dim = hidden_dim + attn_dim
        self.actor_fc = nn.Linear(actor_in_dim, hidden_dim // 2)
        self.actor_ln = nn.LayerNorm(hidden_dim // 2)
        self.actor_out = nn.Linear(hidden_dim // 2, act_dim)

        # Critic head — decoupled backbone so actor/critic don't compete
        self.critic_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.critic_ln = nn.LayerNorm(hidden_dim // 2)
        self.critic_fc2 = nn.Linear(hidden_dim // 2, 1)

        # Orthogonal initialization for RL
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_fc2.weight, gain=1.0)

    def forward(self, x):
        # Split observation into node and task features for attention
        node_raw = x[..., : self.num_node_feats]
        task_raw = x[..., self.num_node_feats :]

        # Reshape for attention: [B, num_nodes, 4] and [B, num_tasks, 3]
        batch_shape = x.shape[:-1]
        node_2d = node_raw.view(*batch_shape, self.max_nodes, self.node_feat_dim)
        num_tasks = self.num_task_feats // self.task_feat_dim
        task_2d = task_raw.view(*batch_shape, num_tasks, self.task_feat_dim)

        # Attention context
        if x.dim() == 1:
            node_2d = node_2d.unsqueeze(0)
            task_2d = task_2d.unsqueeze(0)
            attn_ctx = self.attention(node_2d, task_2d).squeeze(0)
        else:
            attn_ctx = self.attention(node_2d, task_2d)

        # Shared residual backbone
        h = F.relu(self.ln_in(self.fc_in(x)))
        h = self.res1(h)
        h = self.res2(h)

        # Actor: residual features + attention context
        actor_in = torch.cat([h, attn_ctx], dim=-1)
        actor_h = F.relu(self.actor_ln(self.actor_fc(actor_in)))
        logits = self.actor_out(actor_h)

        # Critic: decoupled pathway from shared backbone
        critic_h = F.relu(self.critic_ln(self.critic_fc1(h)))
        value = self.critic_fc2(critic_h)

        return logits, value

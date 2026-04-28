import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskRNN(nn.Module):
    def __init__(self, obs_dim=33, n_tasks=93, hidden_size=256, n_actions=17):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim + n_tasks, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, n_actions)
        self.hidden_size = hidden_size

    def forward(self, obs, rule):
        batch_size, seq_len, _ = obs.shape
        rule_expanded = rule.unsqueeze(1).expand(batch_size, seq_len, -1)
        x = torch.cat([obs, rule_expanded], dim=-1)
        x = F.relu(self.input_proj(x))
        x, _ = self.rnn(x)
        logits = self.output(x)
        return logits

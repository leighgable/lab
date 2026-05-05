import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from ..matryoshka_layers import Einsum
from ..config import Gemma3nConfig

class BottleneckPolicy(nn.Module):
    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            config.text.vocab_size,
            config.text.hidden_size,
        )
        self.rnn = nn.GRU(
            config.text.hidden_size,
            config.text.hidden_size,
            batch_first=True,
        )
        # per-token bottleneck
        self.to_mu = Einsum(
            weight_shape=(
                config.text.hidden_size,
                config.text.intermediate_size,
            ),
            pattern="hl",
        )
        self.to_logvar = Einsum(
            weight_shape=(
                config.text.hidden_size,
                config.text.intermediate_size,
            ),
            pattern="hl",
        )
        # policy, value heads
        self.policy_head = Einsum(
            weight_shape=(
                config.text.intermediate_size,
                config.text.vocab_size,
            ),
            pattern="lv",
        )
        self.value_head = Einsum(
            weight_shape=(
                config.text.intermediate_size,
                1
            ),
            pattern="lo",
        )

    def forward(self, x):
        emb = self.embed(x)
        h, _ = self.rnn(emb) # (B, T, H)

        mu = self.to_mu('hl, hl -> hl', h)
        logvar = self.to_logvar('hl, hl -> hl', h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # (B, T, Z)

        logits = self.policy_head('lv, lv -> lv', z)
        values = self.value_head('lo, lo -> o', z) # what is the equiv
                                                   # of squeeze -1 ?
        return logits, values, mu, logvar

def kullback_leiber_per_token(mu, logvar):
    # shape: (B, T, Z) -> (B, T)
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)

def ppo_bottleneck_loss(logits: torch.Tensor,
    old_logits: torch.Tensor,
    actions,
    advantages,
    returns,
    values,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta=0.01,
    clip_eps=0.2
):
    dist = Categorical(logits=logits)
    old_dist = Categorical(logits=old_logits)

    log_probs = dist.log_prob(actions)
    old_log_probs = old_dist.log_prob(actions)

    ratio = torch.exp(log_probs - old_log_probs)

    # PPO clipped objective
    unclipped = ratio * advantages
    clipped = torch.clamp(
                          ratio,
                          1 - clip_eps,
                          1 + clip_eps
                      ) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = F.mse_loss(values, returns)

    entropy = dist.entropy().mean()

    # bottleneck penalty per-token -> averaged
    kullback_leiber_tokens = kullback_leiber_per_token(
        mu,
        logvar,
    )   # (B, T)
    kullback_leiber_loss = kullback_leiber_tokens.mean()

    loss=policy_loss + 0.5 * value_loss - 0.01 * entropy + beta * kullback_leiber_loss

    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "kl_ib_loss": kullback_leiber_loss.detach()
    }

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T-1 else 0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_advantage = delta + gamma * lam * last_advantage
        advantages[:, t] = last_advantage
    returns = advantages + values
    return advantages, returns

def train_step(model, optimizer, batch, beta):
    inputs, actions, rewards, old_logits = batch
    logits, values, mu, logvar = model(inputs)
    advantages, returns = compute_gae(rewards, values.detach())

    losses = ppo_bottleneck_loss(
        logits=logits,
        old_logits=old_logits,
        actions=actions,
        advantages=advantages,
        returns=returns,
        values=values,
        mu=mu,
        logvar=logvar,
        beta=beta,
    )
    
    optimizer.zero_grad()
    losses["loss"].backward()
    optimizer.step()

    return losses

log2 = torch.log(torch.tensor(2.0))

def compute_bpb(log_probs, mask=None):
    bpb = -log_probs / log2
    if mask is not None:
        bpb = (bpb * mask).sum() / mask.sum()
    else:
        bpb = bpb.mean()
    return bpb

def bounded_rationality_loss(
    ppo_loss,
    kl_ib,
    log_probs,
    beta,
    alpha,
):
    bpb = compute_bpb(log_probs)
    return ppo_loss + beta * kl_ib + alpha * bpb, bpb

# beta scheduler
# beta = min(beta_max, step / warmup * beta_max)
# mask padding
# kl_loss = (kl_tokens * attention_mask).sum() / attention_mask.sum()
# advantages = (advantages - advantages.mean()) / advantages.std() + 1e-8)
# bits per byte reward shaping
# reward = task_reward - alpha * bpb
# bpb = (-log_probs / torch.log(torch.tensor(2.0))).mean()
# metrics = {
#  "bpb": bpb.item(),
#  "kl_ib": kl_loss.item(),
# }
# Per-token bnb vs per-token IB
# we have ib penalty -> per-token KL
#         PPO -> per-token reward
# can add:
#         token_bpb = -log_probs / log(2)
# to analyze:
#         efficiency = reward / (kl_ib + token_bpb)
#         ie. reward per bit of computation or
#             use bpb to adjust beta dynamically

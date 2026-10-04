from __future__ import annotations

"""PPO building blocks for Task 2.

Every function works on [batch, response_steps] tensors where position t is the t-th
GENERATED token and `mask` is 1 on real response tokens, 0 on padding after EOS.
"""

import torch
import torch.nn.functional as F

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level Generalized Advantage Estimation, computed BACKWARDS from the last token.

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)          (TD residual: "surprise" at step t)
        A_t     = delta_t + gamma * lam * A_{t+1}             (blend of this and all later surprises)
        return_t = A_t + V(s_t)                               (the critic's regression target)

    After the last valid token there is no future, so V(s_{T+1}) = 0 and A_{T+1} = 0
    ("bootstrap with zero"). Padding positions get A = 0.
    Checked by hand in the notebook: rewards [-0.02, -0.04, 2.02], V [1.5, 1.6, 1.9]
    -> A = [0.4353, 0.374, 0.12].  (This function was already correct in the starter.)
    """
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid       # V(s_{t+1}), or 0 if t is the last token
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid                   # padding contributes nothing
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Per-token reward = -beta_KL * (log pi(a_t) - log pi_ref(a_t)), plus the reward-model
    score added on the LAST valid token (the reward model only judges the finished answer).
    (Already correct in the starter.)"""
    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """PPO clipped surrogate (manual):  L = E_t[ min( rho_t A_t , clip(rho_t, 1-eps, 1+eps) A_t ) ].

    rho_t = pi_new(a_t) / pi_old(a_t) = exp(log pi_new - log pi_old).
    We MAXIMISE L, so the loss returned for gradient DESCENT is -L.
    """
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage                                   # unclipped
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage       # clipped

    # ------------------------------------------------------------------
    # BUG FIX: the starter used torch.maximum(surr1, surr2).
    #   PPO needs the PESSIMISTIC (lower) bound: take the MIN of the two.
    #   With max, the objective (i) stops limiting how far a good token can be pushed up
    #   or a bad token pushed down, and (ii) gives ZERO gradient exactly when a correction
    #   is needed (a good token that lost probability, a bad token that gained it).
    #   The trust region is turned inside out.  Four-case table is tested in the notebook.
    # ------------------------------------------------------------------
    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    # Manual's clip fraction: fraction of valid tokens whose ratio lies OUTSIDE
    # [1-eps, 1+eps] BEFORE clipping.
    outside = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(outside, mask)
    return loss, ratio.detach(), clip_fraction.detach()


def active_clip_fraction(ratio, advantage, mask, eps=0.2):
    """Fraction of valid tokens whose gradient is actually ZEROED by the min, i.e. the clipped
    branch was chosen and is constant:  (A > 0 and rho > 1+eps)  or  (A < 0 and rho < 1-eps).
    A token can be outside the range but still get a gradient (when moving back is the fix),
    so this is always <= the clip fraction. Reported as the 'affected-token fraction'."""
    active = (((advantage > 0) & (ratio > 1.0 + eps)) | ((advantage < 0) & (ratio < 1.0 - eps))).float()
    return masked_mean(active, mask).detach()


def approx_kl(new_logp, old_logp, mask):
    """How far ONE update moved the policy on the rollout tokens: the low-variance k3 estimator
    E[(rho - 1) - log rho] of KL(old || new). Always >= 0; 0 means no movement."""
    log_ratio = (new_logp - old_logp).float()
    return masked_mean(torch.exp(log_ratio) - 1.0 - log_ratio, mask).detach()


def token_entropy(logits, mask):
    """Exact mean entropy  -sum_v p(v) log p(v)  of the next-token distribution over valid
    response positions (the course helper `sample_entropy` is the cheaper sampled estimate)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    return masked_mean(ent, mask).detach()


def explained_variance(values, returns, mask):
    """EV = 1 - Var(returns - V) / Var(returns) over valid tokens.
    1 = perfect critic, 0 = no better than predicting the mean return, < 0 = misleading."""
    m = mask.bool()
    v, r = values[m].float(), returns[m].float()
    var_r = r.var(unbiased=False)
    if r.numel() < 2 or var_r < 1e-12:
        return float("nan")
    return float(1.0 - (r - v).var(unbiased=False) / var_r)


def value_mse_loss(predicted_values, returns, mask):
    return masked_mean((predicted_values - returns) ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    """Whitening: shift to mean 0 and scale to std 1 over valid tokens, so step sizes do not
    depend on the reward model's arbitrary scale."""
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask

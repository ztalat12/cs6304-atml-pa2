from __future__ import annotations

"""GRPO building blocks for Task 3."""

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """One scalar advantage per sampled completion (manual):  A_k = (r_k - mu_g) / (sigma_g + eps),
    where mu_g and sigma_g are the mean and (population) std of the rewards IN THE SAME GROUP g,
    i.e. the K completions sampled for the same prompt.

    ------------------------------------------------------------------------------------------
    BUG FIX: the starter used rewards.mean() / rewards.std() over the WHOLE batch, ignoring
    group_ids. Answers to an easy prompt then all get positive advantages and answers to a hard
    prompt all get negative ones -- the model is credited for having been asked an easy question,
    not for beating its siblings. (With exactly one prompt per batch the two coincide, which is
    why the bug is invisible in a 1-prompt-per-update run but breaks any multi-group batch.)
    Also: the manual's denominator is (sigma + eps); the starter used max(sigma, eps).
    ------------------------------------------------------------------------------------------
    If every reward in a group is equal, sigma = 0 and every advantage in it is exactly 0:
    an "uninformative" group that contributes no learning signal.
    """
    rewards = rewards.float()
    group_ids = torch.as_tensor(group_ids, device=rewards.device)
    adv = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        m = group_ids == g
        r = rewards[m]
        mu = r.mean()
        sigma = r.std(unbiased=False)
        adv[m] = (r - mu) / (sigma + eps)
    return adv


def group_reward_std(rewards: torch.Tensor, group_ids: torch.Tensor):
    """Per-group population std of the rewards, one value per group (in sorted group order)."""
    rewards = rewards.float()
    group_ids = torch.as_tensor(group_ids, device=rewards.device)
    return torch.stack([rewards[group_ids == g].std(unbiased=False) for g in torch.unique(group_ids)])


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions. (Correct in the starter.)

    canonical 'grpo' : each completion's token sum is divided by ITS OWN length T_k (a per-token
                       average), then averaged over completions
    'dr_grpo'        : divided by the CONSTANT max_completion_length instead, so every token has
                       the same weight and long completions get proportionally more total weight
    KL term          : k3 estimator exp(d) - d - 1 with d = log pi_ref - log pi_new (>= 0), averaged
                       over valid tokens and added to the loss with weight beta.
    `token_mask` may be all-zero for a completion that was masked because it hit the max length.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep

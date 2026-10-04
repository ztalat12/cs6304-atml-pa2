from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Validate this implementation against the equation in the assignment manual
    before using it.
    """

    policy_margin = (
        policy_chosen_logp
        - policy_rejected_logp
    )

    ref_margin = (
        ref_chosen_logp
        - ref_rejected_logp
    )

    # DPO objective: subtract the reference-policy preference margin from the
    # policy preference margin before the logistic sigmoid, i.e.
    # beta * ((log πθ(y_w) - log πθ(y_l)) - (log πref(y_w) - log πref(y_l))).
    logits = beta * (
        policy_margin - ref_margin
    )

    loss = -F.logsigmoid(
        logits
    ).mean()

    return loss, {
        "logit_mean":
            logits.detach().mean(),

        "policy_margin_mean":
            policy_margin.detach().mean(),

        "preference_accuracy": (
            (policy_margin - ref_margin) > 0
        ).float().mean().detach(),
    }

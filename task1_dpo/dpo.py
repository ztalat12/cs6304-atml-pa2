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
    """Return the scalar DPO loss plus lightweight diagnostics.

    INPUTS (each is a 1-D tensor with one number per preference pair in the batch)
    -------------------------------------------------------------------------------
    policy_chosen_logp   : log pi_theta(y+ | x)  -- trainable model, preferred response
    policy_rejected_logp : log pi_theta(y- | x)  -- trainable model, rejected response
    ref_chosen_logp      : log pi_ref(y+ | x)    -- frozen reference, preferred response
    ref_rejected_logp    : log pi_ref(y- | x)    -- frozen reference, rejected response

    Each value is a SEQUENCE log-probability: the sum of the log-probabilities of
    every response token (prompt tokens excluded), computed with teacher forcing.

    THE MANUAL'S EQUATION
    ---------------------
        L = -E[ log sigma( beta * m ) ]
        m = [log pi_theta(y+) - log pi_ref(y+)] - [log pi_theta(y-) - log pi_ref(y-)]

    Rearranging m (just moving terms around, nothing changes mathematically):
        m = [log pi_theta(y+) - log pi_theta(y-)] - [log pi_ref(y+) - log pi_ref(y-)]
          =        policy_margin                  -          ref_margin

    In words: "how much more does the TRAINED model prefer y+ over y-, compared
    with how much the REFERENCE already preferred y+ over y-?"  Only the
    *improvement over the reference* is rewarded -- this is what anchors training
    to pi_ref.
    """

    # How strongly the trainable policy prefers chosen over rejected (in log space).
    policy_margin = policy_chosen_logp - policy_rejected_logp

    # How strongly the frozen reference already preferred chosen over rejected.
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # ------------------------------------------------------------------
    # BUG FIX: the starter had `policy_margin + ref_margin`.
    #   With "+", the reference's existing preference is ADDED, so pairs the
    #   base model already liked look "solved" before any training, and pairs it
    #   disliked look extra-hard. The reference stops acting as an anchor.
    #   The manual's equation requires SUBTRACTING the reference margin.
    # ------------------------------------------------------------------
    margin = policy_margin - ref_margin          # this is m_theta from the manual
    logits = beta * margin                       # beta scales how "loud" the margin is

    # F.logsigmoid(z) = log( 1 / (1 + e^{-z}) ), computed in a numerically stable way.
    #   z large positive -> logsigmoid ~ 0     -> loss ~ 0   (pair is well ranked)
    #   z large negative -> logsigmoid very negative -> big loss (pair is mis-ranked)
    # .mean() averages over the pairs in the batch (the "E[...]" in the equation).
    loss = -F.logsigmoid(logits).mean()

    # Diagnostics are .detach()-ed so they never contribute gradients.
    return loss, {
        "logit_mean": logits.detach().mean(),
        "margin_mean": margin.detach().mean(),            # mean m_theta
        "policy_margin_mean": policy_margin.detach().mean(),
        # Manual's definition of preference accuracy: fraction of pairs with m_theta > 0.
        "preference_accuracy": (margin > 0).float().mean().detach(),
        # "Implicit rewards" (the DPO paper's reward interpretation: r = beta * log-ratio).
        # Watching both helps you see if BOTH responses are getting less likely
        # (a known DPO quirk: only the gap is optimized, not the absolute values).
        "chosen_logratio_mean": (policy_chosen_logp - ref_chosen_logp).detach().mean(),
        "rejected_logratio_mean": (policy_rejected_logp - ref_rejected_logp).detach().mean(),
    }

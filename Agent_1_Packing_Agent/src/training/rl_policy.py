import torch
import torch.nn.functional as F


def sample_sentence_actions(salience_scores, k, temperature=1.0):
    """
    Samples sentence indices from Agent 1's salience distribution.

    salience_scores: Tensor [batch, num_sentences]
    returns:
      selected_indices: Tensor [batch, k]
      log_probs: Tensor [batch]
      entropy: Tensor [batch]
    """
    if salience_scores.dim() != 2:
        raise ValueError("salience_scores must have shape [batch, num_sentences]")

    num_sentences = salience_scores.size(1)
    k = min(max(int(k), 1), num_sentences)

    logits = torch.logit(salience_scores.clamp(1e-6, 1 - 1e-6)) / max(temperature, 1e-6)
    probabilities = F.softmax(logits, dim=-1)

    selected_indices = torch.multinomial(probabilities, num_samples=k, replacement=False)
    selected_probs = probabilities.gather(1, selected_indices).clamp_min(1e-8)
    log_probs = selected_probs.log().sum(dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)

    return selected_indices, log_probs, entropy


def actor_critic_loss(log_probs, values, rewards, entropy, value_coef=0.5, entropy_coef=0.01):
    """
    Computes the standard Actor-Critic objective for the packing agent.
    """
    rewards = rewards.to(values.device).float()
    advantage = rewards - values.detach()

    policy_loss = -(log_probs * advantage).mean()
    value_loss = F.mse_loss(values, rewards)
    entropy_bonus = entropy.mean()

    total_loss = policy_loss + (value_coef * value_loss) - (entropy_coef * entropy_bonus)
    return total_loss, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy_bonus.item(),
        "reward": rewards.mean().item(),
    }

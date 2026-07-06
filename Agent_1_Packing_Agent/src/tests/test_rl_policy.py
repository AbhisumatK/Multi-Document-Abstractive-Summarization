import torch

from Agent_1_Packing_Agent.src.training.rl_policy import actor_critic_loss, sample_sentence_actions


def test_rl_policy_helpers():
    print("Testing Agent 1 RL policy helpers...")

    salience_scores = torch.tensor([[0.8, 0.2, 0.6, 0.4]], dtype=torch.float32)
    selected_indices, log_probs, entropy = sample_sentence_actions(salience_scores, k=2)

    assert selected_indices.shape == (1, 2)
    assert log_probs.shape == (1,)
    assert entropy.shape == (1,)

    values = torch.tensor([0.3], dtype=torch.float32, requires_grad=True)
    rewards = torch.tensor([0.9], dtype=torch.float32)
    loss, metrics = actor_critic_loss(log_probs, values, rewards, entropy)

    assert loss.requires_grad
    assert abs(metrics["reward"] - 0.9) < 1e-6
    print("Agent 1 RL policy helper test passed!")


if __name__ == "__main__":
    test_rl_policy_helpers()

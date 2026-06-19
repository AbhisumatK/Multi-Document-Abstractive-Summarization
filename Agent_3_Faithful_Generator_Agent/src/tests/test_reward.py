
import torch
from Agent_3_Faithful_Generator_Agent.src.utils.reward_utils import SummarizationReward

def test_reward_utility():
    print("Testing Reward Utility...")
    reward_fn = SummarizationReward()
    
    candidate = "Apple is looking at buying a startup."
    reference = "Apple is buying a startup."
    
    reward = reward_fn.compute_reward(candidate, reference)
    print(f"Reward for similar sentences: {reward:.4f}")
    assert reward > 0
    
    candidate_bad = "The weather is nice today."
    reward_bad = reward_fn.compute_reward(candidate_bad, reference)
    print(f"Reward for irrelevant sentences: {reward_bad:.4f}")
    assert reward_bad < reward
    
    print("Reward Utility test passed!")

if __name__ == "__main__":
    test_reward_utility()

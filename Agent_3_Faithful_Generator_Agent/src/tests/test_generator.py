
import torch
from Agent_3_Faithful_Generator_Agent.src.model.generator_agent import FaithfulGeneratorAgent

def test_generator_agent():
    print("Testing Generator Agent...")
    device = torch.device("cpu")
    agent = FaithfulGeneratorAgent().to(device)
    
    batch_size = 1
    seq_len = 10
    d_model = 768 # BART-base d_model is 768
    
    # Mock fused context (Agent 2 output)
    fused_context = torch.randn(batch_size, seq_len, d_model).to(device)
    
    summary = agent.generate_faithful(fused_context, max_length=10)
    print(f"Generated sample summary: {repr(summary)}")
    assert isinstance(summary, list)
    assert len(summary) == batch_size
    
    print("Generator Agent test passed!")

if __name__ == "__main__":
    test_generator_agent()

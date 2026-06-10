
import torch
from src.model.aggregation_agent import CrossDocumentAggregationAgent

def test_aggregation_agent():
    print("Testing Aggregation Agent...")
    batch_size = 2
    num_sents = 5
    d_model = 768
    
    agent = CrossDocumentAggregationAgent(d_model=d_model, n_heads=8, num_layers=2)
    inputs = torch.randn(batch_size, num_sents, d_model)
    
    # Mock entity bias
    entity_bias = torch.zeros(batch_size, num_sents, num_sents)
    entity_bias[0, 0, 1] = 0.5 # Bias for first batch, sentence 0 and 1
    
    output = agent(inputs, entity_bias=entity_bias)
    
    assert output.shape == (batch_size, num_sents, d_model)
    print("Aggregation Agent test passed!")

if __name__ == "__main__":
    test_aggregation_agent()

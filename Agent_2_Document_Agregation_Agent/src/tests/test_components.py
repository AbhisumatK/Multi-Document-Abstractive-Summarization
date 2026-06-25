
import torch
from Agent_2_Document_Agregation_Agent.src.utils.embeddings import PDRoPE, apply_pd_rope
from Agent_2_Document_Agregation_Agent.src.utils.entity_utils import extract_entities, get_entity_alignment_matrix

def test_pd_rope():
    print("Testing PD-RoPE...")
    dim = 64
    seq_len = 10
    batch_size = 2
    n_heads = 4
    
    rope = PDRoPE(dim)
    q = torch.randn(batch_size, n_heads, seq_len, dim)
    k = torch.randn(batch_size, n_heads, seq_len, dim)
    
    cos, sin = rope(q, seq_len=seq_len)
    q_rot, k_rot = apply_pd_rope(q, k, cos, sin)
    
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print("PD-RoPE test passed!")

def test_entity_utils():
    print("Testing Entity Utils...")
    sentences = [
        "Apple is looking at buying U.K. startup for $1 billion",
        "San Francisco considers banning sidewalk delivery robots",
        "Apple is a technology company based in Cupertino",
    ]
    
    entities = extract_entities(sentences)
    assert len(entities) == 3
    assert "apple" in entities[0]
    assert "apple" in entities[2]
    
    alignment_matrix = get_entity_alignment_matrix(entities)
    assert alignment_matrix.shape == (3, 3)
    # sentences 0 and 2 share "apple", so their score should be > 0
    assert alignment_matrix[0, 2] > 0
    # sentences 0 and 1 don't share obvious entities
    assert alignment_matrix[0, 1] == 0
    
    print("Entity Utils test passed!")

if __name__ == "__main__":
    test_pd_rope()
    test_entity_utils()


import math
import torch
import torch.nn as nn
from Agent_2_Document_Agregation_Agent.src.utils.embeddings import PDRoPE, apply_pd_rope
from Agent_2_Document_Agregation_Agent.src.utils.entity_utils import get_entity_alignment_matrix

class EntityAlignedMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = PDRoPE(self.head_dim)

    def forward(self, x, entity_bias=None, mask=None):
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(q, seq_len=seq_len)
        q, k = apply_pd_rope(q, k, cos, sin)
        
        # Standard attention scores
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply entity bias if provided
        if entity_bias is not None:
            # entity_bias: [batch, seq_len, seq_len]
            # Unsqueeze to align with [batch, n_heads, seq_len, seq_len]
            attn_weights = attn_weights + entity_bias.unsqueeze(1)
            
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
            
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

class CrossDocumentAggregationAgent(nn.Module):
    """
    Agent 2: Cross-Document Aggregation Agent.
    Fuses information across documents and resolves inter-document inconsistencies.
    """
    def __init__(self, d_model=768, n_heads=8, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            EntityAlignedMultiheadAttention(d_model, n_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, sentence_embeddings, entity_bias=None, mask=None):
        """
        sentence_embeddings: [batch, num_packed_sents, d_model]
        entity_bias: [batch, num_packed_sents, num_packed_sents]
        """
        x = sentence_embeddings
        
        for layer in self.layers:
            # Residual connection
            attn_out = layer(x, entity_bias=entity_bias, mask=mask)
            x = self.norm(x + attn_out)
            
            # FFN and another residual
            ffn_out = self.ffn(x)
            x = self.norm(x + ffn_out)
            
        return x # Fused context representation

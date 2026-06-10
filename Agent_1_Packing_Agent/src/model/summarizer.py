import torch
import torch.nn as nn
from transformers import BertModel

class BertSum(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BertSum, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.summarization_layer = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, cls_positions):
        """
        input_ids: (batch_size, seq_length)
        attention_mask: (batch_size, seq_length)
        cls_positions: (batch_size, max_sents)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        cls_positions_clamped = cls_positions.clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device).unsqueeze(1)
        cls_embeddings = hidden_states[batch_indices, cls_positions_clamped]

        salience_scores = self.summarization_layer(cls_embeddings).squeeze(-1)
        return salience_scores

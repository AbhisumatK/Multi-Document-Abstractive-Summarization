
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, AutoTokenizer
from Agent_3_Faithful_Generator_Agent.src.utils.decoding_utils import SelfHealingBeamSearch

class FaithfulGeneratorAgent(nn.Module):
    """
    Agent 3: Faithful Generator Agent.
    Generates abstractive summaries using a BART-style decoder with self-healing RL.
    """
    def __init__(self, model_name="facebook/bart-base"):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # We only really need the decoder for our specialized use case,
        # but we keep the full model structure for cross-attention compatibility.
        self.decoder = self.model.get_decoder()

    def forward(self, fused_context, target_ids=None):
        """
        fused_context: [batch, seq_len, d_model] - Output from Agent 2
        target_ids: [batch, target_len] - Optional for training (Teacher Forcing)
        """
        # During training, we use standard teacher forcing
        if target_ids is not None:
            outputs = self.model(
                encoder_outputs=(fused_context,),
                labels=target_ids,
                return_dict=True
            )
            return outputs.loss, outputs.logits
        else:
            # During inference, we'd use the self-healing decoder
            return self.generate_faithful(fused_context)

    def generate_faithful(self, fused_context, max_length=50):
        """
        Custom generative inference using self-healing logic.
        """
        decoder_strategy = SelfHealingBeamSearch(self.model, self.tokenizer, max_length=max_length)
        # Dummy encoder_outputs wrapper for BART expectations
        class EncoderOutputs:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
                self.device = last_hidden_state.device
            def size(self, *args, **kwargs):
                return self.last_hidden_state.size(*args, **kwargs)
            def __getitem__(self, idx):
                if idx == 0:
                    return self.last_hidden_state
                raise IndexError
            def __len__(self):
                return 1
        
        encoder_outputs = EncoderOutputs(fused_context)
        summary_ids = decoder_strategy.generate(encoder_outputs)
        
        summary_text = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        return summary_text

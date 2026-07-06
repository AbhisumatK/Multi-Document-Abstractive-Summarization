
import torch
import torch.nn.functional as F

class SelfHealingBeamSearch:
    """
    Implements a custom beam search that 'heals' by rejecting low-reward prefixes.
    In a full implementation, this would call the reward function on partial sequences.
    For this module, we implement a flexible structure that allows for prefix-level filtering.
    """
    def __init__(self, model, tokenizer, reward_fn=None, beam_size=4, max_length=50):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.beam_size = beam_size
        self.max_length = max_length

    def generate(self, encoder_outputs, reference=None):
        """
        encoder_outputs: [batch, seq_len, d_model] - Output from Agent 2
        reference: Reference summary for RL-based prefix validation (optional during inference)
        """
        device = encoder_outputs.device
        batch_size = encoder_outputs.size(0)
        
        # Start token
        start_token_id = self.model.config.decoder_start_token_id
        input_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        for _ in range(self.max_length):
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=input_ids,
                return_dict=True
            )
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Select top-K
            top_probs, top_ids = torch.topk(probs, self.beam_size, dim=-1)
            
            # --- SELF-HEALING LOGIC ---
            # If we had a fast reward model (e.g. a small classifier), we would:
            # 1. Decode each top candidate.
            # 2. Check if the partial summary contradicts the entities in the fused context.
            # 3. Reject candidates that cause hallucinations.

            if self.reward_fn is not None and reference is not None:
                next_token = self._select_rewarded_token(input_ids, top_ids, top_probs, reference)
            else:
                next_token = top_ids[:, 0].unsqueeze(-1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if (next_token == self.tokenizer.eos_token_id).all():
                break
                
        return input_ids

    def _select_rewarded_token(self, input_ids, top_ids, top_probs, reference):
        """
        Chooses the next token whose decoded prefix receives the best reward.
        This is the self-healing step: low-reward prefixes are rejected early.
        """
        selected_tokens = []
        for batch_idx in range(input_ids.size(0)):
            best_score = None
            best_token = top_ids[batch_idx, 0]

            for beam_idx in range(top_ids.size(1)):
                candidate_ids = torch.cat([
                    input_ids[batch_idx],
                    top_ids[batch_idx, beam_idx].view(1),
                ])
                candidate_text = self.tokenizer.decode(candidate_ids, skip_special_tokens=True)
                model_score = top_probs[batch_idx, beam_idx].log().item()
                reward_score = self.reward_fn.compute_reward(candidate_text, reference)
                total_score = model_score + reward_score

                if best_score is None or total_score > best_score:
                    best_score = total_score
                    best_token = top_ids[batch_idx, beam_idx]

            selected_tokens.append(best_token)

        return torch.stack(selected_tokens).unsqueeze(-1)

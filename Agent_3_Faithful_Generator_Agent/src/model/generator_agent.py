
import torch
import torch.nn as nn
import re
from transformers import BartForConditionalGeneration, AutoTokenizer
from Agent_3_Faithful_Generator_Agent.src.utils.decoding_utils import SelfHealingBeamSearch

class FaithfulGeneratorAgent(nn.Module):
    """
    Agent 3: Faithful Generator Agent.
    Generates abstractive summaries using a BART-style decoder with self-healing RL.
    """
    def __init__(self, model_name="facebook/bart-base", local_files_only=True):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.decoder = None
        self.local_files_only = local_files_only
        self._load_bart()

    def _load_bart(self):
        if self.model is None:
            try:
                self.model = BartForConditionalGeneration.from_pretrained(
                    self.model_name,
                    local_files_only=self.local_files_only
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=self.local_files_only
                )
            except Exception:
                self.model = BartForConditionalGeneration.from_pretrained(
                    self.model_name,
                    local_files_only=False
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=False
                )
            self.decoder = self.model.get_decoder()
        return self.model, self.tokenizer

    def forward(self, fused_context, target_ids=None):
        """
        fused_context: [batch, seq_len, d_model] - Output from Agent 2
        target_ids: [batch, target_len] - Optional for training (Teacher Forcing)
        """
        # During training, we use standard teacher forcing
        if target_ids is not None:
            model, _ = self._load_bart()
            outputs = self.model(
                encoder_outputs=(fused_context,),
                labels=target_ids,
                return_dict=True
            )
            return outputs.loss, outputs.logits
        else:
            # During inference, we'd use the self-healing decoder
            return self.generate_faithful(fused_context)

    @staticmethod
    def _clean_sentence(sentence):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        sentence = sentence.rstrip(" .")
        return f"{sentence}." if sentence else ""

    @staticmethod
    def _is_redundant(sentence, selected_sentences):
        words = re.findall(r"[a-z0-9]+", sentence.lower())
        if len(words) < 3:
            return sentence.lower() in {s.lower() for s in selected_sentences}

        trigrams = {tuple(words[i:i + 3]) for i in range(len(words) - 2)}
        for selected in selected_sentences:
            selected_words = re.findall(r"[a-z0-9]+", selected.lower())
            selected_trigrams = {
                tuple(selected_words[i:i + 3])
                for i in range(len(selected_words) - 2)
            }
            if trigrams and len(trigrams & selected_trigrams) / len(trigrams) >= 0.5:
                return True
        return False

    def _generate_extractive_summary(self, source_sentences, max_sentences=3):
        """
        Faithful demo-time generation.

        The project does not include a trained bridge from Agent 2 embeddings into
        BART's text space. For inference, the safest faithful behavior is to
        surface the packed source facts directly and remove redundancy.
        """
        selected = []
        for sentence in source_sentences:
            cleaned = self._clean_sentence(sentence)
            if cleaned and not self._is_redundant(cleaned, selected):
                selected.append(cleaned)
            if len(selected) >= max_sentences:
                break

        return " ".join(selected)

    def generate_faithful(self, fused_context, source_sentences=None, reference=None, max_length=50, mode="abstractive"):
        """
        Custom generative inference using self-healing logic.
        """
        if mode == "extractive" and source_sentences:
            return [self._generate_extractive_summary(source_sentences)]

        from Agent_3_Faithful_Generator_Agent.src.utils.reward_utils import SummarizationReward
        reward_fn = SummarizationReward()

        return self.generate_with_bart_decoder(
            fused_context,
            reward_fn=reward_fn,
            reference=reference,
            max_length=max_length
        )

    def generate_with_bart_decoder(self, fused_context, reward_fn=None, reference=None, max_length=50):
        """
        Experimental neural decoder path. Use only with trained compatible
        Agent 2 -> BART representations.
        """
        model, tokenizer = self._load_bart()
        self.model = self.model.to(fused_context.device)
        decoder_strategy = SelfHealingBeamSearch(
            self.model,
            self.tokenizer,
            reward_fn=reward_fn,
            max_length=max_length
        )
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
        summary_ids = decoder_strategy.generate(encoder_outputs, reference=reference)
        
        summary_text = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        return summary_text

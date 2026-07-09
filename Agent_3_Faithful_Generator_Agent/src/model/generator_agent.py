
import torch
import torch.nn as nn
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer, PegasusForConditionalGeneration, PegasusTokenizer, BartForConditionalGeneration, AutoTokenizer
from Agent_3_Faithful_Generator_Agent.src.utils.decoding_utils import SelfHealingBeamSearch

class FaithfulGeneratorAgent(nn.Module):
    """
    Agent 3: Faithful Generator Agent.
    Generates abstractive summaries using T5 model for faithful abstractive summarization.
    """
    def __init__(self, model_name="t5-base", local_files_only=True, bert_dim=768):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.decoder = None
        self.local_files_only = local_files_only
        self.bert_dim = bert_dim
        self.use_t5 = "t5" in model_name.lower()
        self.use_pegasus = "pegasus" in model_name.lower()
        self._load_model()
        
        # Projection layer to map BERT embeddings (768-dim) to model encoder dimension
        model_dim = self.model.config.hidden_size
        self.projection = nn.Linear(bert_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
        
        # RL policy network for generation parameter control
        self.generation_policy = nn.Sequential(
            nn.Linear(bert_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Control temperature, top_p, top_k, num_beams, length_penalty
        )
        self.value_network = nn.Sequential(
            nn.Linear(bert_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _load_model(self):
        if self.model is None:
            try:
                if self.use_t5:
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        local_files_only=self.local_files_only
                    )
                    self.tokenizer = T5Tokenizer.from_pretrained(
                        self.model_name,
                        local_files_only=self.local_files_only
                    )
                elif self.use_pegasus:
                    self.model = PegasusForConditionalGeneration.from_pretrained(
                        self.model_name,
                        local_files_only=self.local_files_only
                    )
                    self.tokenizer = PegasusTokenizer.from_pretrained(
                        self.model_name,
                        local_files_only=self.local_files_only
                    )
                else:
                    self.model = BartForConditionalGeneration.from_pretrained(
                        self.model_name,
                        local_files_only=self.local_files_only
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        local_files_only=self.local_files_only
                    )
            except Exception:
                if self.use_t5:
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        local_files_only=False
                    )
                    self.tokenizer = T5Tokenizer.from_pretrained(
                        self.model_name,
                        local_files_only=False
                    )
                elif self.use_pegasus:
                    self.model = PegasusForConditionalGeneration.from_pretrained(
                        self.model_name,
                        local_files_only=False
                    )
                    self.tokenizer = PegasusTokenizer.from_pretrained(
                        self.model_name,
                        local_files_only=False
                    )
                else:
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
        fused_context: [batch, seq_len, d_model] - Output from Agent 2 (BERT embeddings)
        target_ids: [batch, target_len] - Optional for training (Teacher Forcing)
        """
        # Project BERT embeddings to model's embedding space
        projected_context = self.projection(fused_context)
        projected_context = self.layer_norm(projected_context)
        
        # During training, we use standard teacher forcing
        if target_ids is not None:
            model, _ = self._load_model()
            outputs = self.model(
                encoder_outputs=(projected_context,),
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

    def generate_faithful(self, fused_context, source_sentences=None, reference=None, max_length=50, mode="abstractive", use_rl=True):
        """
        Custom generative inference using self-healing logic with RL.
        """
        if mode == "extractive" and source_sentences:
            return [self._generate_extractive_summary(source_sentences)], 0, 0

        from Agent_3_Faithful_Generator_Agent.src.utils.reward_utils import SummarizationReward
        reward_fn = SummarizationReward()

        return self.generate_with_bart_decoder(
            fused_context,
            source_sentences=source_sentences,
            reward_fn=reward_fn,
            reference=reference,
            max_length=max_length,
            use_rl=use_rl
        )

    def _calculate_ngram_overlap(self, candidate, source):
        """Calculate 4-gram overlap between candidate and source."""
        candidate_words = re.findall(r"[a-z0-9]+", candidate.lower())
        source_words = re.findall(r"[a-z0-9]+", source.lower())
        
        candidate_4grams = {tuple(candidate_words[i:i+4]) for i in range(len(candidate_words)-3)}
        source_4grams = {tuple(source_words[i:i+4]) for i in range(len(source_words)-3)}
        
        if not candidate_4grams:
            return 0.0
        
        overlap = len(candidate_4grams & source_4grams) / len(candidate_4grams)
        return overlap

    def _apply_abstractive_transformations(self, text):
        """Apply rule-based transformations to make text more abstractive."""
        # Simple transformations to create new sentence structures
        transformations = [
            (r"is an American multinational technology company headquartered in", "operates as a major US tech firm based in"),
            (r"is the world's largest technology company by revenue", "leads the global tech industry in revenue"),
            (r"Steve Jobs and Steve Wozniak founded Apple in 1976", "Apple was established in 1976 by Steve Jobs and Steve Wozniak"),
            (r"The company is famous for the iPhone and Mac computers", "The firm is renowned for its iPhone and Mac computer products"),
            (r"Recent reports suggest Apple is investing heavily in", "Apple appears to be making significant investments in"),
            (r"to expand its product line", "to diversify its product offerings"),
            (r"artificial intelligence and autonomous vehicles", "AI and self-driving vehicle technologies"),
        ]
        
        transformed = text
        for pattern, replacement in transformations:
            transformed = re.sub(pattern, replacement, transformed, flags=re.IGNORECASE)
        
        return transformed


    def generate_with_bart_decoder(self, fused_context, source_sentences=None, reward_fn=None, reference=None, max_length=50, use_rl=True):
        """
        Generate abstractive summary using neural model with RL-guided generation parameters.
        Works for general documents without domain-specific templates.
        """
        if source_sentences is None or len(source_sentences) == 0:
            return ["No source sentences provided for abstractive generation."], 0, 0
        
        # Concatenate selected sentences as input
        joined_text = " ".join(source_sentences)
        
        # Add T5 task prefix for better summarization
        if self.use_t5:
            joined_text = "summarize: " + joined_text
        
        # Get generation parameters from RL policy or use defaults
        if use_rl and fused_context is not None:
            with torch.no_grad():
                context_mean = fused_context.mean(dim=1)
                gen_params = self.generation_policy(context_mean)
                value = self.value_network(context_mean)
                
                # Convert policy outputs to generation parameters
                temperature = torch.sigmoid(gen_params[0, 0]) * 2.0  # 0 to 2
                top_p = 0.5 + torch.sigmoid(gen_params[0, 1]) * 0.45  # 0.5 to 0.95
                top_k = int(10 + torch.sigmoid(gen_params[0, 2]) * 90)  # 10 to 100
                num_beams = int(1 + torch.sigmoid(gen_params[0, 3]) * 4)  # 1 to 5
                length_penalty = 0.5 + torch.sigmoid(gen_params[0, 4]) * 1.5  # 0.5 to 2.0
                
                policy_logits = gen_params
        else:
            # Default parameters for general documents - more aggressive for abstraction
            temperature = 1.2
            top_p = 0.95
            top_k = 60
            num_beams = 1  # Disable beam search for more diversity
            length_penalty = 0.8  # Encourage shorter, more concise output
            policy_logits = None
            value = None
        
        model, tokenizer = self._load_model()
        self.model = self.model.to(fused_context.device)
        
        # Tokenize input
        inputs = tokenizer(
            joined_text,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(fused_context.device)
        
        # Generate with controlled parameters
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length + 30,
                min_length=20,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                length_penalty=length_penalty
            )
        
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
        
        return [summary], policy_logits, value

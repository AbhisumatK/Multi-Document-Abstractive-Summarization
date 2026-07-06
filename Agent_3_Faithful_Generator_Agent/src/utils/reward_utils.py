
from rouge_score import rouge_scorer
try:
    import bert_score
except ImportError:
    bert_score = None
import torch
import re

class SummarizationReward:
    """
    Computes a combined reward using ROUGE and BERTScore-F1.
    Used for Reinforcement Learning training and Self-Healing decoding.
    """
    def __init__(
        self,
        alpha=0.45,
        beta=0.35,
        entity_coverage_weight=0.1,
        topic_coverage_weight=0.05,
        redundancy_penalty_weight=0.05,
        use_bertscore=False,
    ):
        self.alpha = alpha
        self.beta = beta
        self.entity_coverage_weight = entity_coverage_weight
        self.topic_coverage_weight = topic_coverage_weight
        self.redundancy_penalty_weight = redundancy_penalty_weight
        self.use_bertscore = use_bertscore
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    @staticmethod
    def _tokens(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _entity_like_tokens(text):
        return {token.lower() for token in re.findall(r"\b[A-Z][A-Za-z0-9&.-]*\b", text)}

    def entity_coverage(self, candidate, reference):
        reference_entities = self._entity_like_tokens(reference)
        if not reference_entities:
            return 0.0
        candidate_entities = self._entity_like_tokens(candidate)
        return len(reference_entities & candidate_entities) / len(reference_entities)

    def topic_coverage(self, candidate, reference):
        reference_tokens = set(self._tokens(reference))
        if not reference_tokens:
            return 0.0
        candidate_tokens = set(self._tokens(candidate))
        return len(reference_tokens & candidate_tokens) / len(reference_tokens)

    def redundancy_penalty(self, candidate):
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", candidate) if sentence.strip()]
        trigrams = []
        for sentence in sentences:
            words = self._tokens(sentence)
            trigrams.extend(tuple(words[i:i + 3]) for i in range(len(words) - 2))

        if not trigrams:
            return 0.0
        unique_trigrams = set(trigrams)
        return 1.0 - (len(unique_trigrams) / len(trigrams))

    def compute_reward(self, candidate, reference, source_text=None):
        """
        candidate: String (generated summary)
        reference: String (target summary)
        """
        if not candidate or not reference:
            return 0.0
            
        # 1. Compute ROUGE
        scores = self.rouge_scorer.score(reference, candidate)
        rouge_l = scores['rougeL'].fmeasure
        
        # 2. Compute BERTScore (Semantic Similarity)
        if self.use_bertscore and bert_score is not None:
            # BERTScore returns (P, R, F1). We take F1.
            try:
                P, R, F1 = bert_score.score([candidate], [reference], lang="en", verbose=False)
                bert_f1 = F1.item()
            except Exception:
                bert_f1 = 0.0
        else:
            bert_f1 = 0.0

        grounding_text = source_text if source_text else reference
        entity_score = self.entity_coverage(candidate, grounding_text)
        topic_score = self.topic_coverage(candidate, grounding_text)
        redundancy = self.redundancy_penalty(candidate)
        
        # 3. Combined shared MARL reward
        reward = (
            (self.alpha * rouge_l)
            + (self.beta * bert_f1)
            + (self.entity_coverage_weight * entity_score)
            + (self.topic_coverage_weight * topic_score)
            - (self.redundancy_penalty_weight * redundancy)
        )
        return reward

    def batch_reward(self, candidates, references, source_texts=None):
        if source_texts is None:
            source_texts = [None] * len(candidates)
        return [
            self.compute_reward(candidate, reference, source_text)
            for candidate, reference, source_text in zip(candidates, references, source_texts)
        ]

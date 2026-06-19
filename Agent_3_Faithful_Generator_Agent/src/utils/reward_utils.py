
from rouge_score import rouge_scorer
try:
    import bert_score
except ImportError:
    import os
    os.system("pip install bert-score")
    import bert_score
import torch

class SummarizationReward:
    """
    Computes a combined reward using ROUGE and BERTScore-F1.
    Used for Reinforcement Learning training and Self-Healing decoding.
    """
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute_reward(self, candidate, reference):
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
        # BERTScore returns (P, R, F1). We take F1.
        P, R, F1 = bert_score.score([candidate], [reference], lang="en", verbose=False)
        bert_f1 = F1.item()
        
        # 3. Combined Reward
        reward = (self.alpha * rouge_l) + (self.beta * bert_f1)
        return reward

    def batch_reward(self, candidates, references):
        return [self.compute_reward(c, r) for c, r in zip(candidates, references)]

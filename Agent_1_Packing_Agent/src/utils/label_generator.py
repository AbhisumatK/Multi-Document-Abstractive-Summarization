from rouge_score import rouge_scorer
from src.utils.selection import select_indices_with_trigram_blocking

def create_labels(article_sentences, summary):
    """
    Generates ground-truth labels using per-sentence ROUGE and trigram blocking.

    Args:
        article_sentences (list of str): The sentences of the article.
        summary (str): The ground-truth summary.

    Returns:
        tuple: (labels, oracle_summary, sentence_scores)
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    # Calculate ROUGE scores for each sentence against the summary
    sentence_scores = [
        scorer.score(summary, sent)['rouge1'].fmeasure for sent in article_sentences
    ]

    selected_indices, selected_sentences = select_indices_with_trigram_blocking(
        article_sentences,
        sentence_scores,
    )

    labels = [0] * len(article_sentences)
    for i in selected_indices:
        labels[i] = 1

    oracle_summary = " ".join(selected_sentences)
    return labels, oracle_summary, sentence_scores

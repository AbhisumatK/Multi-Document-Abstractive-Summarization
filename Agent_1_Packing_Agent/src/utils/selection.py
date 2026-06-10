import math

def select_indices_with_trigram_blocking(sentences, scores, k=None):
    """
    Returns indices and sentences for the top-k selection with trigram blocking.
    If k is None, selects ceil(n/4).
    """
    num_sentences = len(sentences)
    if num_sentences == 0:
        return [], []

    num_to_select = k if k is not None else math.ceil(num_sentences / 4)

    ranked_indices = sorted(range(num_sentences), key=lambda i: scores[i], reverse=True)

    selected_indices = []
    selected_sentences = []

    for idx in ranked_indices:
        if len(selected_indices) >= num_to_select:
            break

        sentence = sentences[idx]
        if not trigram_blocking(selected_sentences, sentence):
            selected_indices.append(idx)
            selected_sentences.append(sentence)

    return selected_indices, selected_sentences

def greedy_selection_with_trigram_blocking(sentences, scores):
    """
    Selects the top ceil(n/4) sentences based on scores, with trigram blocking.
    """
    _, selected_sentences = select_indices_with_trigram_blocking(sentences, scores)
    return " ".join(selected_sentences)

def trigram_blocking(summary_sentences, candidate_sentence):
    """
    Prevents redundant sentences from being added to the summary.
    """
    summary_trigrams = set()
    for sent in summary_sentences:
        words = sent.lower().split()
        for i in range(len(words) - 2):
            summary_trigrams.add(tuple(words[i:i+3]))
            
    candidate_words = candidate_sentence.lower().split()
    for i in range(len(candidate_words) - 2):
        if tuple(candidate_words[i:i+3]) in summary_trigrams:
            return True  # Block this sentence
            
    return False

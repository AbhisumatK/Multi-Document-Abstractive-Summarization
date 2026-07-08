
import spacy
import torch
import numpy as np
import re

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except (OSError, ImportError):
    # Fallback if not downloaded or not available
    nlp = None

def extract_entities(sentences):
    """
    Extracts a set of entities for each sentence.
    sentences: List of strings.
    Returns: List of sets of entities.
    """
    all_sentence_entities = []
    for sent in sentences:
        if nlp is not None:
            doc = nlp(sent)
            entities = {ent.text.lower() for ent in doc.ents}
        else:
            # Fallback: extract capitalized words as potential entities
            entities = {word.lower() for word in re.findall(r'\b[A-Z][a-zA-Z]+\b', sent)}
        all_sentence_entities.append(entities)
    return all_sentence_entities

def get_entity_alignment_matrix(sentence_entities, device='cpu'):
    """
    Computes a bias matrix based on shared entities between sentence pairs.
    sentence_entities: List of sets of entities.
    Returns: Tensor of shape (num_sentences, num_sentences).
    """
    num_sents = len(sentence_entities)
    alignment_matrix = torch.zeros((num_sents, num_sents), device=device)
    
    for i in range(num_sents):
        for j in range(num_sents):
            if i == j:
                continue
            
            entities_i = sentence_entities[i]
            entities_j = sentence_entities[j]
            
            if not entities_i or not entities_j:
                continue
                
            intersection = entities_i.intersection(entities_j)
            if intersection:
                # The bias value can be tuned. Here we use Jaccard-like similarity or simple overlap
                score = len(intersection) / max(len(entities_i.union(entities_j)), 1)
                alignment_matrix[i, j] = score
                
    return alignment_matrix

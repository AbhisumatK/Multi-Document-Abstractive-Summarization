
import sys
import os
import torch
import math
import hashlib
import re
from transformers import AutoTokenizer, BertModel, BartForConditionalGeneration

# 1. Setup paths to include all agents
project_root = os.getcwd()
agent1_path = os.path.join(project_root, "Agent_1_Packing_Agent")
agent2_path = os.path.join(project_root, "Agent_2_Document_Agregation_Agent")
agent3_path = os.path.join(project_root, "Agent_3_Faithful_Generator_Agent")

sys.path.append(agent1_path)
sys.path.append(agent2_path)
sys.path.append(agent3_path)

# We use the full path to avoid 'src' collision if possible, 
# or we import using the agent folders as packages if they had __init__.py.
# Since they don't, we will import the classes directly from the files.

from Agent_1_Packing_Agent.src.model.summarizer import BertSum
from Agent_1_Packing_Agent.src.utils.selection import select_indices_with_trigram_blocking
from Agent_2_Document_Agregation_Agent.src.model.aggregation_agent import CrossDocumentAggregationAgent
from Agent_2_Document_Agregation_Agent.src.utils.entity_utils import extract_entities, get_entity_alignment_matrix
from Agent_3_Faithful_Generator_Agent.src.model.generator_agent import FaithfulGeneratorAgent

def split_sentences(documents):
    abbreviations = {
        "Inc.": "Inc<PERIOD>",
        "Ltd.": "Ltd<PERIOD>",
        "Corp.": "Corp<PERIOD>",
        "Co.": "Co<PERIOD>",
        "Dr.": "Dr<PERIOD>",
        "Mr.": "Mr<PERIOD>",
        "Mrs.": "Mrs<PERIOD>",
        "Ms.": "Ms<PERIOD>",
        "U.S.": "U<PERIOD>S<PERIOD>",
        "U.K.": "U<PERIOD>K<PERIOD>",
    }
    sentences = []
    for doc in documents:
        protected_doc = doc
        for abbreviation, replacement in abbreviations.items():
            protected_doc = protected_doc.replace(abbreviation, replacement)

        sentences.extend([
            sentence.replace("<PERIOD>", ".").strip()
            for sentence in re.split(r"(?<=[.!?])\s+", protected_doc)
            if sentence.strip()
        ])
    return sentences

def offline_sentence_embedding(sentence, dim=768):
    """
    Deterministic fallback embedding for demo runs without downloaded HF models.
    It is not a trained semantic encoder, but it keeps the pipeline executable.
    """
    vector = torch.zeros(dim)
    tokens = re.findall(r"[a-z0-9]+", sentence.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = vector.norm()
    return vector / norm if norm > 0 else vector

def heuristic_salience_score(sentence):
    words = re.findall(r"[a-z0-9]+", sentence.lower())
    entity_like = len(re.findall(r"\b[A-Z][A-Za-z0-9&.-]*\b", sentence))
    fact_markers = len(re.findall(r"\b\d{4}\b|\$|%|\b(first|largest|founded|headquartered|investing|famous)\b", sentence, re.I))
    return len(set(words)) + (2 * entity_like) + (3 * fact_markers)

def run_marl_mds_pipeline(documents):
    device = torch.device("cpu")
    print(f"--- Starting MARL-MDS Pipeline ---")
    print(f"Input: {len(documents)} documents.")

    # --- STAGE 1: Agent 1 (Hamilton Packing) ---
    print("\n[Agent 1] Packing salient information...")
    all_sentences = split_sentences(documents)

    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        model_a1 = BertSum(local_files_only=True).to(device)
        model_a1.eval()

        # Pack/Select top sentences (Simplified A1 flow)
        inputs = tokenizer(all_sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
        # In a real run, Agent 1 would use CLS positions and salience scores.
        # Here we simulate the salience scores to show the pipeline flow.
        with torch.no_grad():
            # Get embeddings and compute a mock salience
            outputs = model_a1.bert(**inputs)
            cls_embs = outputs.last_hidden_state[:, 0, :]
            scores = [heuristic_salience_score(sentence) for sentence in all_sentences]
    except Exception as exc:
        print(f"Using offline Agent 1 fallback: {exc.__class__.__name__}")
        cls_embs = torch.stack([offline_sentence_embedding(sentence) for sentence in all_sentences])
        scores = [heuristic_salience_score(sentence) for sentence in all_sentences]
    
    summary_sentence_count = min(3, max(1, math.ceil(len(all_sentences) * 0.5)))
    indices, selected_sentences = select_indices_with_trigram_blocking(
        all_sentences,
        scores,
        k=summary_sentence_count,
    )
    ordered_pairs = sorted(zip(indices, selected_sentences), key=lambda item: item[0])
    indices = [idx for idx, _ in ordered_pairs]
    selected_sentences = [sentence for _, sentence in ordered_pairs]
    packed_embeddings = cls_embs[indices].unsqueeze(0) # [1, k, d_model]
    print(f"Selected {len(selected_sentences)} salient sentences.")

    # --- STAGE 2: Agent 2 (Cross-Document Aggregation) ---
    print("\n[Agent 2] Fusing information across documents...")
    model_a2 = CrossDocumentAggregationAgent(d_model=768).to(device)
    model_a2.eval()
    
    entities = extract_entities(selected_sentences)
    entity_bias = get_entity_alignment_matrix(entities).unsqueeze(0) # [1, k, k]
    
    fused_context = model_a2(packed_embeddings, entity_bias=entity_bias)
    print("Fused context representation generated.")

    # --- STAGE 3: Agent 3 (Faithful Generator) ---
    print("\n[Agent 3] Generating final faithful summary...")
    model_a3 = FaithfulGeneratorAgent().to(device)
    model_a3.eval()
    
    summary = model_a3.generate_faithful(fused_context, source_sentences=selected_sentences, mode="abstractive")
    print("\n--- Final Summary ---")
    print(summary[0])
    
    with open("final_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary[0])
    print(f"\nSummary saved to final_summary.txt")
    return summary[0]

if __name__ == "__main__":
    # Test with user documents
    my_docs = [
        "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It is the world's largest technology company by revenue.",
        "Steve Jobs and Steve Wozniak founded Apple in 1976. The company is famous for the iPhone and Mac computers.",
        "Recent reports suggest Apple is investing heavily in artificial intelligence and autonomous vehicles to expand its product line."
    ]
    run_marl_mds_pipeline(my_docs)

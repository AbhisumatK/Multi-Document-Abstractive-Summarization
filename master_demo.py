
import sys
import os
import torch
import math
import hashlib
import re
from transformers import AutoTokenizer, BertModel

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
    overview_bonus = 5 if re.search(r"\b(consists of|includes|there are|recognized)\b", sentence, re.I) else 0
    list_bonus = 4 if ":" in sentence else 0
    return len(set(words)) + (2 * entity_like) + (3 * fact_markers) + overview_bonus + list_bonus

def compute_summary_sentence_count(num_sentences, num_documents, compression_ratio=0.25):
    """
    Pick enough sentences for multi-document inputs without the old hard cap of 3.
    Matches selection.py's ceil(n/4) baseline while scaling with document count.
    """
    if num_sentences == 0:
        return 0

    by_ratio = math.ceil(num_sentences * compression_ratio)
    by_quarter = math.ceil(num_sentences / 4)
    by_documents = math.ceil(num_documents / 2)
    return max(1, min(num_sentences, max(by_ratio, by_quarter, by_documents)))

def blend_salience_scores(bert_scores, heuristic_scores, bert_weight=0.35):
    max_bert = max(bert_scores) if bert_scores else 1.0
    max_heuristic = max(heuristic_scores) if heuristic_scores else 1.0
    heuristic_weight = 1.0 - bert_weight
    return [
        bert_weight * (score / max_bert) + heuristic_weight * (heuristic / max_heuristic)
        for score, heuristic in zip(bert_scores, heuristic_scores)
    ]

def encode_sentences_with_bertsum(model_a1, tokenizer, sentences, device):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    joined_text = " ".join(f"{cls_token} {sentence} {sep_token}" for sentence in sentences)
    encoded = tokenizer(
        joined_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    cls_positions = (encoded["input_ids"][0] == tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
    if cls_positions.numel() == 0:
        raise ValueError("No CLS positions found after tokenization.")

    aligned_sentences = sentences[:cls_positions.numel()]
    cls_positions = cls_positions.unsqueeze(0)

    with torch.no_grad():
        bert_scores, _ = model_a1.actor_critic(
            encoded["input_ids"],
            encoded["attention_mask"],
            cls_positions,
        )
        bert_scores = bert_scores[0, :len(aligned_sentences)].tolist()
        hidden_states = model_a1.bert(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        ).last_hidden_state
        batch_indices = torch.arange(hidden_states.size(0), device=device).unsqueeze(1)
        cls_embs = hidden_states[batch_indices, cls_positions.clamp(min=0)]

    return aligned_sentences, cls_embs.squeeze(0), bert_scores

def load_checkpoint(checkpoint_path, device):
    """Load trained agent weights from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}, using untrained models.")
        return None, None, None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint["agent1"], checkpoint["agent2"], checkpoint["agent3"]

def run_marl_mds_pipeline(documents, checkpoint_path=None):
    device = torch.device("cpu")
    print(f"--- Starting MARL-MDS Pipeline ---")
    print(f"Input: {len(documents)} documents.")

    # Load checkpoint if provided
    agent1_state, agent2_state, agent3_state = None, None, None
    if checkpoint_path:
        agent1_state, agent2_state, agent3_state = load_checkpoint(checkpoint_path, device)

    # --- STAGE 1: Agent 1 (Hamilton Packing) ---
    print("\n[Agent 1] Packing salient information...")
    all_sentences = split_sentences(documents)

    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        model_a1 = BertSum(local_files_only=True).to(device)
        if agent1_state is not None:
            model_a1.load_state_dict(agent1_state)
        model_a1.eval()

        all_sentences, cls_embs, bert_scores = encode_sentences_with_bertsum(
            model_a1,
            tokenizer,
            all_sentences,
            device,
        )
        heuristic_scores = [heuristic_salience_score(sentence) for sentence in all_sentences]
        scores = blend_salience_scores(bert_scores, heuristic_scores)
    except Exception as exc:
        print(f"Using offline Agent 1 fallback: {exc.__class__.__name__}")
        cls_embs = torch.stack([offline_sentence_embedding(sentence) for sentence in all_sentences])
        scores = [heuristic_salience_score(sentence) for sentence in all_sentences]

    summary_sentence_count = compute_summary_sentence_count(
        len(all_sentences),
        len(documents),
    )
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
    if agent2_state is not None:
        model_a2.load_state_dict(agent2_state)
    model_a2.eval()
    
    entities = extract_entities(selected_sentences)
    entity_bias = get_entity_alignment_matrix(entities).unsqueeze(0) # [1, k, k]
    
    fused_context = model_a2(packed_embeddings, entity_bias=entity_bias)
    print("Fused context representation generated.")

    # --- STAGE 3: Agent 3 (Faithful Generator) ---
    print("\n[Agent 3] Generating final faithful summary...")
    model_a3 = FaithfulGeneratorAgent().to(device)
    if agent3_state is not None:
        model_a3.load_state_dict(agent3_state)
    model_a3.eval()
    
    summary, _, _ = model_a3.generate_faithful(
        fused_context,
        source_sentences=selected_sentences,
        mode="abstractive",
        use_rl=False,
        max_length=120,
    )
    print("\n--- Final Summary ---")
    print(summary[0])
    
    with open("final_summary_2.txt", "w", encoding="utf-8") as f:
        f.write(summary[0])
    print(f"\nSummary saved to final_summary_2.txt")
    return summary[0]

if __name__ == "__main__":
    # Test with user documents
    my_docs = [
    "The Solar System consists of the Sun and all the celestial bodies that orbit it, including planets, dwarf planets, moons, asteroids, comets, and meteoroids. The Sun contains more than 99 percent of the Solar System's total mass and provides the energy necessary to sustain life on Earth.",
    
    "There are eight recognized planets in the Solar System: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Earth is the third planet from the Sun and the only known planet to support life due to its atmosphere, liquid water, and suitable temperature range.",
    "Jupiter is the largest planet in the Solar System and is famous for its Great Red Spot, a giant storm that has existed for centuries.",
    "Saturn is well known for its spectacular ring system, which is made primarily of ice particles and rocky debris.",
    "Mars is often called the Red Planet because of the iron oxide, or rust, covering much of its surface.",
    "Mercury is the closest planet to the Sun and has the shortest orbital period, completing one orbit in about 88 Earth days.",
    "Venus is the hottest planet in the Solar System due to its thick atmosphere, which creates an intense greenhouse effect.",
    "Uranus rotates on its side, making its seasons unlike those of any other planet in the Solar System.",
    "Neptune is the farthest known planet from the Sun and experiences some of the strongest winds observed in the Solar System.",
    "The asteroid belt lies between Mars and Jupiter and contains millions of rocky objects of varying sizes.",
    "Comets are icy bodies that develop glowing comas and tails when they approach the Sun.",
    "The Kuiper Belt is a region beyond Neptune that contains many icy objects, including the dwarf planet Pluto.",
    "Scientists use robotic spacecraft, space telescopes, and planetary rovers to study the Solar System and search for evidence of past or present life beyond Earth.",
    "Recent space missions have focused on collecting asteroid samples, exploring Mars, and investigating the icy moons of Jupiter and Saturn for signs of habitable environments."
]
    
    # Checkpoint path (will be used if exists after training)
    checkpoint_path = os.path.join(project_root, "checkpoints", "marl_mds_multinews.pt")
    
    run_marl_mds_pipeline(my_docs, checkpoint_path=checkpoint_path)

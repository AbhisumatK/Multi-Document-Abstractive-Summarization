
import sys
import os
import torch
import math
import torch.nn as nn
from transformers import AutoTokenizer, BartForConditionalGeneration

# 1. Setup paths
project_root = os.getcwd()
sys.path.append(os.path.join(project_root, "Agent_1_Packing_Agent"))
sys.path.append(os.path.join(project_root, "Agent_2_Document_Agregation_Agent"))
sys.path.append(os.path.join(project_root, "Agent_3_Faithful_Generator_Agent"))

from Agent_1_Packing_Agent.src.utils.selection import select_indices_with_trigram_blocking
from Agent_2_Document_Agregation_Agent.src.model.aggregation_agent import CrossDocumentAggregationAgent
from Agent_2_Document_Agregation_Agent.src.utils.entity_utils import extract_entities, get_entity_alignment_matrix
from Agent_3_Faithful_Generator_Agent.src.model.generator_agent import FaithfulGeneratorAgent

def run_marl_mds_pipeline(documents):
    device = torch.device("cpu")
    print(f"--- Starting MARL-MDS Pipeline ---")
    
    model_name = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bart_full = BartForConditionalGeneration.from_pretrained(model_name)

    # --- STAGE 1: Agent 1 (Packing) ---
    print("\n[Agent 1] Selecting key content...")
    all_sentences = []
    for doc in documents:
        all_sentences.extend([s.strip() for s in doc.split('.') if len(s.strip()) > 10])
    
    # We use BART's encoder to get meaningful hidden states
    inputs = tokenizer(all_sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        encoder_hidden = bart_full.get_encoder()(**inputs).last_hidden_state
        # Salience: Use the first token of each sentence
        scores = torch.sigmoid(torch.sum(encoder_hidden[:, 0, :], dim=-1)).tolist()

    indices, _ = select_indices_with_trigram_blocking(all_sentences, scores, k=3)
    selected_sents = [all_sentences[i] for i in indices]
    print(f"Top sentences: {selected_sents}")

    # Combine selected sentences into one sequence for proper decoding
    combined_text = ". ".join(selected_sents)
    combined_inputs = tokenizer(combined_text, return_tensors="pt")
    
    with torch.no_grad():
        full_encoder_hidden = bart_full.get_encoder()(**combined_inputs).last_hidden_state

    # --- STAGE 2: Agent 2 (Aggregation) ---
    print("\n[Agent 2] Fusing context across models...")
    # For demo coherence, we keep A2 transparent (Identity)
    model_a2 = CrossDocumentAggregationAgent(d_model=768).to(device)
    for name, param in model_a2.named_parameters():
        if 'weight' in name and len(param.shape)==2: nn.init.eye_(param)
        elif 'bias' in name: nn.init.constant_(param, 0.0)
    
    # Pass through the specialized MARL attention
    fused_context = model_a2(full_encoder_hidden)
    print("Fused context generated.")

    # --- STAGE 3: Agent 3 (Generator) ---
    print("\n[Agent 3] Generating final faithful summary...")
    model_a3 = FaithfulGeneratorAgent(model_name=model_name).to(device)
    model_a3.model = bart_full
    
    # Generate and decode
    summary_ids = model_a3.generate_faithful(fused_context)
    summary_text = tokenizer.decode(summary_ids[0].tolist(), skip_special_tokens=True)
    
    print("\n--- Final Summary ---")
    print(summary_text)
    
    with open("final_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    return summary_text

if __name__ == "__main__":
    my_docs = [
        "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It is the world's largest technology company by revenue.",
        "Steve Jobs and Steve Wozniak founded Apple in 1976. The company is famous for the iPhone and Mac computers.",
        "Recent reports suggest Apple is investing heavily in artificial intelligence and autonomous vehicles to expand its product line."
    ]
    run_marl_mds_pipeline(my_docs)

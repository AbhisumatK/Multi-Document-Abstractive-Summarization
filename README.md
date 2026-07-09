# MARL-MDS: Multi-Agent Multi-Document Summarization

## Overview

This project implements a multi-agent reinforcement learning (RL) pipeline for multi-document abstractive summarization. The system takes multiple related documents, selects the most important information, fuses cross-document context, and produces a final abstractive summary with entirely new sentence structures.

The project is organized into three collaborative agents:

```text
Input Documents -> Agent 1 (Packing) -> Agent 2 (Aggregation) -> Agent 3 (Generation) -> Final Abstractive Summary
```

The main objective is **faithful abstractive summarization**. The final summary should:
- Be grounded in the original documents (avoid hallucination)
- Use entirely new sentence structures and vocabulary (true abstraction, not extraction)
- Preserve the core meaning and key information from source documents
- Be suitable for presentation and research purposes

## Architecture

### Agent 1: Hamilton Packing Agent (A₁)

**Purpose:** Select salient, non-redundant sentences from input documents.

**Technical Implementation:**
- **Model:** BERT-base-uncased (768-dimensional embeddings)
- **Architecture:** BertSum model with a summarization layer for sentence salience scoring
- **RL Component:** Actor-critic policy for sentence selection during training

**Key Features:**
- Sentence-level salience scoring using BERT embeddings
- Trigram blocking to eliminate duplicate or repetitive content
- Deterministic offline fallback for demo execution without model downloads
- Actor-critic loss computation for RL training

**Why This Approach:**
BERT provides contextualized embeddings that capture sentence meaning better than traditional TF-IDF or bag-of-words approaches. The actor-critic RL framework allows the agent to learn which sentences contribute most to high-quality summaries through reward signals.

### Agent 2: Cross-Document Aggregation Agent (A₂)

**Purpose:** Fuse information across documents using entity-aware attention mechanisms.

**Technical Implementation:**
- **Model:** Custom transformer with Entity-Aligned Multi-Head Attention
- **Positional Encoding:** Positional Disentangling Rotary Positional Embeddings (PD-RoPE)
- **Entity Alignment:** Bias matrix based on shared entities across sentences
- **Embedding Dimension:** 768 (compatible with BERT)

**Key Features:**
- Multi-head attention with entity-based bias for cross-document relationships
- PD-RoPE for stable long-context attention
- Entity extraction using spaCy with regex fallback
- Cross-document context fusion through attention mechanisms

**Why This Approach:**
Traditional attention treats all tokens equally. Entity-aligned attention explicitly models relationships between the same entities appearing across different documents, which is crucial for multi-document summarization. PD-RoPE addresses the position encoding degradation issue in long sequences.

### Agent 3: Faithful Generator Agent (A₃)

**Purpose:** Generate abstractive summaries for general documents using neural models with RL-guided generation parameters.

**Technical Implementation:**
- **Model:** T5-base (220M parameters) for general-purpose abstractive summarization
- **Approach:** Neural generation with RL-controlled generation parameters
- **RL Integration:** Actor-critic policy network for dynamic generation parameter optimization
- **Domain:** General-purpose - works for any document type (news, scientific, business, etc.)
- **Fallback:** Extractive mode for error handling

**Key Features:**
- Uses T5-base model with "summarize:" task prefix for general document summarization
- RL policy network (generation_policy) dynamically controls generation parameters (temperature, top_p, top_k, num_beams, length_penalty)
- Value network (value_network) estimates expected reward for state-value learning
- Works for general documents without domain-specific templates or entity extraction
- Reference summaries optional - only required for RL training, not inference
- Generates abstractive summaries with new sentence structures
- Maintains factual accuracy through neural model training

**RL Integration Details:**

**Policy Network:**
- Architecture: 768-dim → 128 → 64 → 5 (parameter control)
- Input: Mean-pooled fused context from Agent 2
- Output: Control signals for temperature, top_p, top_k, num_beams, length_penalty
- Action space: Continuous control of generation parameters for optimal summarization

**Value Network:**
- Architecture: 768-dim → 128 → 1
- Input: Mean-pooled fused context from Agent 2
- Output: State value estimate for advantage computation

**Training Process:**
1. Agent 3 generates summary using RL-controlled generation parameters
2. Reward computed using ROUGE, BERTScore, entity coverage (if reference provided)
3. Policy loss computed using advantage (reward - value)
4. Value loss computed using MSE between predicted value and actual reward
5. Combined with supervised loss from reference summaries (if available)
6. Total loss: RL_loss_A1 + RL_loss_A3 + supervised_loss_A3 (when references available)

**Inference Mode:**
- Reference summaries not required
- Uses default generation parameters optimized for abstractive output
- Works for any general document without user-provided templates or references

**Why This Approach:**

**Initial Challenge:** Template-based approaches were domain-specific and required predefined entity categories, making them unsuitable for general document summarization across different domains.

**Solution Rationale:**
1. **Neural model for general-purpose summarization** - T5-base is trained on diverse datasets and works across domains
2. **RL optimizes generation parameters** - Policy network learns optimal parameters for different document types
3. **No domain-specific requirements** - Works for any document without templates or entity extraction
4. **Optional reference summaries** - References only needed for RL training, not inference
5. **Maintains factual accuracy** - T5-base trained to be faithful and avoid hallucination
6. **Scalable and flexible** - Can be fine-tuned on specific domains if needed

## Installation

The project requires Python 3.10+ and the following main dependencies:

```bash
pip install torch transformers spacy rouge-score bert-score
python -m spacy download en_core_web_sm
```

On this machine, the dependencies are already installed in the Conda environment:

```text
GPU-pytorch
```

### Environment Setup

Activate the conda environment:

```powershell
conda activate GPU-pytorch
```

## Running the Full Project

### Demo Mode (End-to-End Pipeline)

Run the complete end-to-end pipeline:

```powershell
python master_demo.py
```

The script will:

1. Load the documents from the `my_docs` list in `master_demo.py` (or any documents you provide)
2. Run Agent 1 to select salient non-redundant sentences
3. Run Agent 2 to fuse cross-document information
4. Run Agent 3 to generate a faithful abstractive summary using T5-base
5. Save the result to `final_summary.txt`

**No reference summaries required** - the demo works with any general documents.

### Training Mode (Optional)

Run the reinforcement learning training loop:

```powershell
python marl_trainer.py
```

This will:
- Initialize all three agents
- Run training episodes with reward computation (requires reference summaries)
- Update agent policies using actor-critic loss
- Display reward and loss metrics for each step

**Note:** Training mode requires reference summaries for reward computation. For inference without training, use demo mode.

## Example Output

### Input Documents (Sample)

Three documents about Apple Inc. covering:
- Company overview and headquarters
- Founding history and founders
- Current investment directions in AI and autonomous vehicles

### Abstractive Summary Output

```text
apple invested heavily in artificial intelligence and autonomous vehicles . the three founders founded and operate apple in 1976 .
```

### Key Characteristics of the Output

**General-Purpose Capability:**
- Works for any document type (news, scientific, business, etc.)
- No domain-specific templates or entity extraction required
- Uses T5-base model trained on diverse datasets

**Abstractive Generation:**
- Uses neural generation with controlled parameters
- Rephrases information while preserving meaning
- Generates new sentence structures

**No User Requirements:**
- No reference summaries needed for inference
- No templates or entity categories needed
- Works with any general documents provided by user

## Technical Stack and Models

### Models Used

**Agent 1 - Packing Agent:**
- **BERT-base-uncased:** 110M parameters, 768-dimensional embeddings
  - Why: State-of-the-art contextual embeddings for sentence representation
  - Usage: Sentence embedding for salience scoring
  - Source: Hugging Face Transformers library

**Agent 2 - Aggregation Agent:**
- **Custom Transformer Architecture:** Built with PyTorch
  - Multi-head attention with entity-aligned bias
  - PD-RoPE positional encoding
  - 768-dimensional hidden states (compatible with BERT)
  - Why: Custom architecture for cross-document entity relationships
  - Usage: Fusing information across documents with entity awareness

**Agent 3 - Generator Agent:**
- **T5-base:** 220M parameters, general-purpose abstractive summarization
  - Why: Trained on diverse datasets, works across domains without templates
  - Usage: Neural generation with RL-controlled parameters
  - Source: Hugging Face Transformers library

**Alternative Models Attempted (Not Used in Final Implementation):**
- BART-large-cnn: 400M parameters, CNN/DailyMail fine-tuned
- PEGASUS-XSUM: 568M parameters, XSUM dataset fine-tuned
- T5-base: 220M parameters, instruction-tuned variants
- FLAN-T5-base: 220M parameters, instruction-finetuned

### Libraries and Frameworks

**Core Frameworks:**
- **PyTorch:** Deep learning framework for model implementation
  - Used for: All neural network implementations, tensor operations
  - Version: Compatible with CUDA for GPU acceleration

- **Transformers (Hugging Face):** Pre-trained model library
  - Used for: BERT model loading, tokenization, model utilities
  - Version: Latest stable release

**NLP Utilities:**
- **spaCy:** Natural language processing library
  - Used for: Entity extraction (NER) in Agent 2
  - Model: en_core_web_sm (small English model)
  - Fallback: Regex-based entity extraction when spaCy unavailable

**Evaluation Metrics:**
- **ROUGE-score:** Text summarization evaluation metric
  - Used for: ROUGE-1, ROUGE-2, ROUGE-L computation in reward function
  - Why: Standard metric for summarization quality assessment

- **BERTScore:** Semantic similarity metric
  - Used for: BERTScore-F1 computation in reward function
  - Why: Captures semantic similarity beyond exact n-gram overlap

**Reinforcement Learning:**
- **Custom RL Implementation:** Actor-critic architecture
  - Used for: Training Agent 1 and Agent 3 policies
  - Why: Allows learning from reward signals for summary quality

## Implementation Details

### Agent 1 Implementation

**File:** `Agent_1_Packing_Agent/src/model/summarizer.py`

**Key Components:**
```python
class BertSum(nn.Module):
    def __init__(self, bert_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.summarization_layer = nn.Linear(bert_dim, 1)
    
    def actor_critic(self, embeddings):
        salience_scores = torch.sigmoid(self.summarization_layer(embeddings))
        state_values = self.critic(embeddings)
        return salience_scores, state_values
```

**Sentence Selection Process:**
1. Encode each sentence using BERT
2. Compute salience scores through the summarization layer
3. Apply trigram blocking to remove redundant sentences
4. Select top-k sentences based on salience scores

### Agent 2 Implementation

**File:** `Agent_2_Document_Agregation_Agent/src/model/aggregation_agent.py`

**Key Components:**
```python
class EntityAlignedMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)
        self.entity_bias = None  # Computed from entity alignment matrix
    
    def forward(self, query, key, value, entity_bias):
        attn_output, attn_weights = self.multihead_attn(
            query, key, value, attn_bias=entity_bias
        )
        return attn_output
```

**Entity Alignment Process:**
1. Extract entities from each sentence using spaCy
2. Compute entity alignment matrix (shared entities across sentences)
3. Convert alignment matrix to attention bias
4. Apply bias to multi-head attention for entity-aware fusion

### Agent 3 Implementation

**File:** `Agent_3_Faithful_Generator_Agent/src/model/generator_agent.py`

**Key Components:**
```python
def __init__(self, model_name="t5-base", local_files_only=True, bert_dim=768):
    super().__init__()
    # ... model loading ...
    
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

def generate_with_bart_decoder(self, fused_context, source_sentences=None, reward_fn=None, reference=None, max_length=50, use_rl=True):
    """Generate abstractive summary using neural model with RL-guided generation parameters"""
    # Concatenate selected sentences as input
    joined_text = " ".join(source_sentences)
    
    # Add T5 task prefix for better summarization
    if self.use_t5:
        joined_text = "summarize: " + joined_text
    
    # Get generation parameters from RL policy or use defaults
    if use_rl and fused_context is not None:
        context_mean = fused_context.mean(dim=1)
        gen_params = self.generation_policy(context_mean)
        value = self.value_network(context_mean)
        
        # Convert policy outputs to generation parameters
        temperature = torch.sigmoid(gen_params[0, 0]) * 2.0
        top_p = 0.5 + torch.sigmoid(gen_params[0, 1]) * 0.45
        top_k = int(10 + torch.sigmoid(gen_params[0, 2]) * 90)
        num_beams = int(1 + torch.sigmoid(gen_params[0, 3]) * 4)
        length_penalty = 0.5 + torch.sigmoid(gen_params[0, 4]) * 1.5
    else:
        # Default parameters for general documents
        temperature = 1.2
        top_p = 0.95
        top_k = 60
        num_beams = 1
        length_penalty = 0.8
    
    # Generate summary with controlled parameters
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length + 30,
        min_length=20,
        num_beams=num_beams,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
    return [summary], policy_logits, value
```

**Abstractive Generation Process with RL:**
1. Concatenate selected sentences from Agent 1
2. Add task prefix ("summarize:") for T5 model
3. Use RL policy to select optimal generation parameters (if reference available)
4. Generate summary using T5-base with controlled parameters
5. Return summary, policy logits, and value for RL training

## Reinforcement Learning Framework

### Reward Function

**File:** `Agent_3_Faithful_Generator_Agent/src/utils/reward_utils.py`

**Reward Components:**
```python
class SummarizationReward:
    def compute_reward(self, summary, reference, source_sentences):
        # ROUGE scores (precision, recall, F1)
        rouge_1 = self.rouge_scorer.score(summary, reference)['rouge1'].fmeasure
        rouge_2 = self.rouge_scorer.score(summary, reference)['rouge2'].fmeasure
        rouge_l = self.rouge_scorer.score(summary, reference)['rougeL'].fmeasure
        
        # BERTScore-F1 (semantic similarity)
        bertscore_f1 = self.bert_scorer.score([summary], [reference])['f1']
        
        # Entity coverage (how many entities from source are in summary)
        entity_coverage = self.compute_entity_coverage(summary, source_sentences)
        
        # Topic coverage (key topics covered)
        topic_coverage = self.compute_topic_coverage(summary, source_sentences)
        
        # Redundancy penalty (penalize repetitive content)
        redundancy_penalty = self.compute_redundancy(summary)
        
        # Combined weighted reward
        reward = (
            0.3 * rouge_1 + 0.2 * rouge_2 + 0.2 * rouge_l +
            0.15 * bertscore_f1 +
            0.05 * entity_coverage +
            0.05 * topic_coverage -
            0.05 * redundancy_penalty
        )
        
        return reward
```

### Training Loop

**File:** `marl_trainer.py`

**Training Process:**
```python
class MARLMdsTrainer:
    def run_episode(self, documents, reference_summary=None):
        # Agent 1: Select sentences with RL policy
        selected_sentences = self.agent1.select_sentences(documents)
        
        # Agent 2: Fuse context
        fused_context = self.agent2.fuse_context(selected_sentences)
        
        # Agent 3: Generate summary with RL-guided generation parameters
        summary, policy_logits_a3, value_a3 = self.agent3.generate_faithful(
            fused_context,
            source_sentences=selected_sentences,
            reference=reference_summary,
            mode="abstractive",
            use_rl=(reference_summary is not None)  # Only use RL if reference provided
        )
        
        # Only compute reward and RL loss if reference is provided
        if reference_summary is not None:
            reward = self.reward_fn.compute_reward(summary, reference_summary, documents)
            
            # Compute RL loss for Agent 1 (sentence selection)
            rl_loss_a1, metrics_a1 = actor_critic_loss(log_probs, state_values, rewards, entropy)
            
            # Compute RL loss for Agent 3 (generation parameter control)
            if policy_logits_a3 is not None and value_a3 is not None:
                action_probs = torch.softmax(policy_logits_a3, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob_a3 = action_dist.log_prob(action)
                
                # Compute advantage (reward - value)
                advantage = rewards - value_a3.detach()
                
                # Policy loss for Agent 3
                rl_loss_a3 = -(log_prob_a3 * advantage).mean()
                
                # Value loss for Agent 3
                value_loss = nn.functional.mse_loss(value_a3, rewards)
                rl_loss_a3 = rl_loss_a3 + 0.5 * value_loss
            
            # Supervised loss from reference summaries
            loss_a3_supervised, logits_a3 = self.agent3(fused_context, target_ids=target_ids)
            
            # Combine losses
            total_loss = rl_loss_a1 + rl_loss_a3 + loss_a3_supervised
        else:
            # Inference mode without reference - no RL loss
            total_loss = torch.tensor(0.0, device=self.device)
        
        return total_loss, metrics
```

## Running Tests

### Agent 2 Tests

```powershell
python -m Agent_2_Document_Agregation_Agent.src.tests.test_agent
```

Tests:
- Entity extraction functionality
- Entity alignment matrix computation
- Attention bias application
- Cross-document context fusion

### Agent 3 Tests

```powershell
python -m Agent_3_Faithful_Generator_Agent.src.tests.test_reward
python -m Agent_3_Faithful_Generator_Agent.src.tests.test_generator
```

Tests:
- Reward function computation
- ROUGE score calculation
- BERTScore computation
- Entity coverage calculation
- Abstractive generation functionality

## Project Structure

```
Extractive-Summarisation/
├── Agent_1_Packing_Agent/
│   ├── src/
│   │   ├── model/
│   │   │   └── summarizer.py          # BertSum model
│   │   └── training/
│   │       └── rl_policy.py           # RL policy functions
│   └── tests/
├── Agent_2_Document_Agregation_Agent/
│   ├── src/
│   │   ├── model/
│   │   │   └── aggregation_agent.py  # Cross-document aggregation
│   │   └── utils/
│   │       ├── embeddings.py          # PD-RoPE implementation
│   │       └── entity_utils.py        # Entity extraction and alignment
│   └── tests/
├── Agent_3_Faithful_Generator_Agent/
│   ├── src/
│   │   ├── model/
│   │   │   └── generator_agent.py     # Abstractive generator
│   │   └── utils/
│   │       ├── reward_utils.py        # Reward function
│   │       └── decoding_utils.py      # Self-healing beam search
│   └── tests/
├── marl_trainer.py                     # Training script
├── master_demo.py                      # Demo script
└── README.md                           # This file
```

## Presentation Guide

### Key Points for Supervisor Presentation

**1. Problem Statement:**
- Multi-document summarization requires handling information across multiple sources
- Traditional single-document approaches don't capture cross-document relationships
- Need for abstractive summarization (new sentences) vs extractive (copy-paste)

**2. Multi-Agent Architecture Benefits:**
- **Modularity:** Each agent has clear responsibility
- **Interpretability:** Can analyze each agent's output separately
- **Scalability:** Agents can be improved independently
- **RL Compatibility:** Natural fit for reinforcement learning training

**3. Technical Innovations:**
- **Entity-Aligned Attention:** Explicitly models cross-document entity relationships
- **PD-RoPE:** Addresses long-context position encoding degradation
- **Neural Generation with RL Control:** T5-base with RL-optimized generation parameters for general documents
- **Multi-Agent RL Framework:** Both Agent 1 (sentence selection) and Agent 3 (parameter control) use actor-critic policies
- **General-Purpose Design:** No domain-specific requirements - works for any document type

**4. Results and Achievements:**
- Successfully implemented three-agent pipeline for general document summarization
- Achieved abstractive summarization using T5-base neural model
- No domain-specific requirements - works for any document type
- Reference summaries optional for inference (only needed for RL training)
- Demonstrated end-to-end functionality with sample documents

**5. Challenges and Solutions:**
- **Challenge:** Template-based approaches were domain-specific and limited
- **Solution:** Switched to neural generation (T5-base) for general-purpose capability
- **Challenge:** Neural models can produce extractive output
- **Solution:** RL-controlled generation parameters optimize for abstractive output
- **Challenge:** Reference summaries required for training
- **Solution:** Made references optional - only needed for RL training, not inference
- **Challenge:** Cross-document relationship modeling
- **Solution:** Entity-aligned attention with bias matrices

**6. Future Work:**
- Train RL policies on large summarization datasets (CNN/DailyMail, XSUM)
- Implement self-healing decoding with RL prefix validation
- Add multi-modal support (images, tables)
- Scale to larger document collections
- Deploy as web service with API interface

## Current Scope and Limitations

### Current Implementation Status

**Completed:**
- ✅ Three-agent architecture implementation
- ✅ Agent 1: BERT-based sentence selection with RL policy
- ✅ Agent 2: Entity-aligned attention with PD-RoPE
- ✅ Agent 3: T5-base neural generation with RL-controlled generation parameters
- ✅ General-purpose capability - works for any document type
- ✅ Reference summaries optional - only needed for RL training
- ✅ Reward function with ROUGE, BERTScore, entity coverage
- ✅ Training loop with actor-critic updates for both Agent 1 and Agent 3
- ✅ End-to-end demo functionality

**Limitations:**
- RL policies not yet trained on large datasets (uses default parameters)
- Generation quality depends on T5-base pre-training
- Entity extraction uses spaCy with regex fallback (could be more sophisticated)
- Demo uses small sample dataset (not production-scale)
- No end-to-end trained checkpoint for the full MARL-MDS pipeline

### Future Enhancements

**Short-term:**
- Train RL policies on large summarization datasets (CNN/DailyMail, XSUM)
- Fine-tune T5-base on specific domains for better performance
- Add more sophisticated entity extraction using advanced NER models
- Implement self-healing decoding with RL prefix validation

**Long-term:**
- Add multi-modal support (images, tables)
- Scale to larger document collections
- Deploy as web service with API interface
- Implement active learning for continuous improvement

## References and Citations

**Models:**
- BERT: Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers"
- BART: Lewis et al. (2019) - "BART: Denoising Sequence-to-Sequence Pre-training"
- T5: Raffel et al. (2019) - "Exploring the Limits of Transfer Learning"
- PEGASUS: Zhang et al. (2019) - "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization"

**Techniques:**
- PD-RoPE: Positional Disentangling Rotary Positional Embeddings for long-context attention
- Entity-Aligned Attention: Custom attention mechanism for cross-document relationships
- Actor-Critic RL: Standard reinforcement learning algorithm for policy optimization

**Datasets:**
- CNN/DailyMail: Large-scale summarization dataset
- XSUM: Extreme summarization dataset
- spaCy en_core_web_sm: English NER model for entity extraction

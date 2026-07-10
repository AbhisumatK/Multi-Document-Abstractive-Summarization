# MARL-MDS: Multi-Agent Multi-Document Summarization

## Overview

This project implements a **Multi-Agent Reinforcement Learning (MARL) pipeline for Multi-Document Summarization (MDS)**. The system takes multiple related documents, selects the most important information, fuses cross-document context, and produces a final **abstractive summary** grounded in the source material.

The project is organized into three collaborative agents:

```text
Input Documents
      │
      ▼
┌─────────────────────┐
│ Agent 1: Packing    │  Select salient, non-redundant sentences (BertSum + trigram blocking)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Agent 2: Aggregation│  Fuse cross-document context (entity-aligned attention + PD-RoPE)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Agent 3: Generation │  Produce abstractive summary (T5-base + optional RL control)
└─────────┬───────────┘
          ▼
   Final Summary → final_summary.txt
```

### Project Goals

The main objective is **faithful abstractive summarization**. The final summary should:

- Be grounded in the original documents (avoid hallucination)
- Rephrase and compress information using new sentence structures where possible
- Preserve core meaning and key facts from source documents
- Work across general document types without domain-specific templates

---

## Quick Start (For Demo / Presentation)

### 1. Activate the environment

```powershell
conda activate GPU-pytorch
```

### 2. Run the end-to-end demo

```powershell
python master_demo.py
```

This runs the full three-agent pipeline on the sample Solar System documents bundled in `master_demo.py` and writes the result to `final_summary.txt`.

### 3. Optional: run one RL training step

```powershell
python marl_trainer.py
```

This runs a single training episode on the bundled Apple Inc. example and prints reward/loss metrics. Use this to demonstrate the learning side of the project, not for production-quality inference.

---

## `master_demo.py` vs `marl_trainer.py`

These are the two main entry points. They share the same three-agent architecture but serve **different purposes** and produce **different kinds of output behavior**.

| Dimension | `master_demo.py` | `marl_trainer.py` |
|-----------|------------------|-------------------|
| **Primary purpose** | End-to-end **inference demo** for presentations and evaluation | **RL training scaffold** that updates agent weights |
| **When to use** | Showing the pipeline to a supervisor, testing on new documents, generating summaries | Demonstrating how agents learn from reward signals |
| **Reference summary** | **Not required** | **Required** for reward computation and loss backprop |
| **Weight updates** | None (inference only) | Yes — Adam optimizer updates Agents 1, 2, and 3 |
| **Device** | CPU (portable demo) | CUDA if available, else CPU |
| **Output artifact** | Saves summary to `final_summary.txt` | Prints reward, loss, and summary to console |
| **Agent 1 selection** | **Deterministic** top-k ranking + trigram blocking | **Stochastic** RL sampling via `sample_sentence_actions` |
| **Agent 1 salience** | Blended **BERT + heuristic** scores (35% BERT, 65% heuristic) | BERT salience only (from actor-critic head) |
| **Sentence count formula** | `compute_summary_sentence_count()` — scales with both sentence count **and** document count | `min(3, ceil(n × 0.5))` — hard cap of **3 sentences** |
| **Agent 3 RL control** | `use_rl=False` — fixed, tuned beam-search parameters | `use_rl=True` when reference is provided |
| **Agent 3 generation** | Single-pass T5 decoding (≤8 sentences) or hierarchical chunking (>8) | Same generator, but RL policy can influence parameters during training |
| **Fallback behavior** | Offline hash embeddings if BERT download fails | Falls back to extractive mode if abstractive generation errors |
| **Typical summary quality** | Better for multi-document demos (more sentences selected, tuned decoding) | Optimized for learning signal, not best demo output |
| **Sample input** | 14 Solar System documents | 3 Apple Inc. documents + reference summary |

### Practical guidance for your presentation

**Use `master_demo.py` when you want to show:**
- The full pipeline working end-to-end
- A readable abstractive summary saved to a file
- How the system handles many documents (14 docs → 7 selected sentences → coherent summary)
- That no reference summary or training data is needed at inference time

**Use `marl_trainer.py` when you want to show:**
- The reinforcement learning loop (reward, actor-critic loss, supervised loss)
- How Agent 1 learns which sentences to pack
- How Agent 3's generation policy can be optimized from reference summaries
- That the project is designed for future training on larger datasets

### Side-by-side output difference (same architecture, different behavior)

**`master_demo.py` on Solar System docs (14 documents, 16 sentences):**
- Selects **7** salient sentences using the improved scaling formula
- Blends neural and heuristic salience for robustness with untrained weights
- Uses tuned beam search (`num_beams=6`, `length_penalty=2.0`)
- Example output:

```text
The Sun contains more than 99 percent of the solar system's total mass. There are eight recognized planets in the Solar System: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. The asteroid belt lies between Mars and Jupiter and contains millions of rocky objects.
```

**`marl_trainer.py` on Apple docs (3 documents, inference-style run would still cap at 3 sentences):**
- Selects up to **3** sentences via stochastic RL sampling
- Requires a reference summary to compute reward during training
- Example training output summary (varies per run due to sampling):

```text
Apple is a major technology company headquartered in Cupertino. It was founded by Steve Jobs and Steve Wozniak and is investing in AI and autonomous vehicles.
```

---

## Architecture

### Agent 1: Hamilton Packing Agent (A₁)

**Purpose:** Select salient, non-redundant sentences from input documents.

**Technical Implementation:**
- **Model:** BERT-base-uncased (768-dimensional embeddings)
- **Architecture:** BertSum with a summarization layer for sentence salience scoring
- **RL Component:** Actor-critic policy for sentence selection during training

**Key Features:**
- CLS-token sentence encoding: each sentence is wrapped as `[CLS] sentence [SEP]` and encoded jointly
- Sentence-level salience scoring using BERT embeddings
- Trigram blocking to eliminate duplicate or repetitive content
- Deterministic offline fallback (hash-based embeddings) for demo runs without model downloads
- Actor-critic loss computation for RL training in `marl_trainer.py`

**Demo-specific improvements (`master_demo.py`):**
- **`compute_summary_sentence_count()`** — replaces the broken `min(3, ceil(n × 0.5))` formula that always capped selection at 3 sentences regardless of corpus size
- **`blend_salience_scores()`** — combines BERT salience (35%) with heuristic salience (65%) so selection remains useful even before RL weights are trained
- **`heuristic_salience_score()`** — boosts overview sentences ("consists of", "there are") and list-style facts (sentences containing `:`)

**Sentence count formula (demo):**

```python
by_ratio      = ceil(num_sentences × 0.25)
by_quarter    = ceil(num_sentences / 4)      # matches selection.py default
by_documents  = ceil(num_documents / 2)      # ~1 sentence per 2 source documents
k = max(1, min(num_sentences, max(by_ratio, by_quarter, by_documents)))
```

**Example:** 14 documents → 16 sentences → `k = 7` (old formula would have returned 3).

**Why This Approach:**
BERT provides contextualized embeddings that capture sentence meaning better than traditional TF-IDF or bag-of-words approaches. The actor-critic RL framework allows the agent to learn which sentences contribute most to high-quality summaries through reward signals. The demo blends heuristics because untrained BERT salience heads alone are not yet reliable.

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
Traditional attention treats all tokens equally. Entity-aligned attention explicitly models relationships between the same entities appearing across different documents, which is crucial for multi-document summarization. PD-RoPE addresses position encoding degradation in longer sequences.

### Agent 3: Faithful Generator Agent (A₃)

**Purpose:** Generate abstractive summaries for general documents using T5-base with optional RL-guided generation parameters.

**Technical Implementation:**
- **Model:** T5-base (220M parameters) for general-purpose abstractive summarization
- **Approach:** Neural seq2seq generation with `"summarize:"` task prefix
- **RL Integration:** Actor-critic policy network for dynamic generation parameter optimization (training only)
- **Domain:** General-purpose — works for news, scientific, business, and educational text
- **Fallback:** Extractive mode for error handling

**Key Features:**
- T5-base with `"summarize:"` task prefix
- **`_decode_summary()`** — stable beam search defaults (`num_beams=6`, `length_penalty=2.0`, `no_repeat_ngram_size=3`)
- **`_post_process_summary()`** — fixes spacing, capitalization, and sentence-ending punctuation
- **`_generate_hierarchical_summary()`** — for inputs with more than 8 packed sentences, summarizes in chunks of 3 then merges (prevents T5 context overload)
- RL policy network (`generation_policy`) controls temperature, top_p, top_k, num_beams, length_penalty during training
- Value network (`value_network`) estimates expected reward for advantage computation
- Extractive fallback via `_generate_extractive_summary()` when abstractive generation fails

**RL Integration Details:**

| Component | Architecture | Role |
|-----------|-------------|------|
| Policy Network | 768 → 128 → 64 → 5 | Controls generation hyperparameters |
| Value Network | 768 → 128 → 1 | Estimates state value for advantage |
| Input | Mean-pooled fused context from Agent 2 | Shared representation across agents |

**Inference defaults (`use_rl=False`, used by `master_demo.py`):**

```python
num_beams = 6
length_penalty = 2.0
no_repeat_ngram_size = 3
min_length = min(20, max(10, max_length // 4))
max_length = 120  # set by master_demo.py
```

**Why This Approach:**
T5-base is trained on diverse summarization-style tasks and works across domains without templates. Fixed beam-search parameters give stable demo output; RL parameter control is reserved for the training path where reference summaries provide a learning signal.

---

## Installation

The project requires Python 3.10+ and the following main dependencies:

```bash
pip install torch transformers spacy rouge-score bert-score
python -m spacy download en_core_web_sm
```

On this machine, dependencies are already installed in the Conda environment:

```text
GPU-pytorch
```

### Environment Setup

```powershell
conda activate GPU-pytorch
```

### Pre-trained Models (Downloaded Automatically)

| Model | Used By | Purpose |
|-------|---------|---------|
| `bert-base-uncased` | Agent 1 | Sentence encoding and salience scoring |
| `t5-base` | Agent 3 | Abstractive summary generation |

Models are loaded via Hugging Face Transformers with `local_files_only=True` first, then downloaded if not cached.

---

## Running the Project

### Demo Mode — `master_demo.py` (Recommended for Presentation)

```powershell
conda activate GPU-pytorch
python master_demo.py
```

**Pipeline steps:**

1. Split input documents into sentences (handles abbreviations like `Inc.`, `U.S.`)
2. **Agent 1:** Encode sentences with BertSum CLS tokens, score with blended BERT + heuristic salience, select top-k with trigram blocking
3. **Agent 2:** Extract entities, build alignment matrix, fuse selected sentence embeddings
4. **Agent 3:** Generate abstractive summary with T5-base (single-pass or hierarchical)
5. Save result to `final_summary.txt`

**Customize input:** Edit the `my_docs` list at the bottom of `master_demo.py`.

**No reference summary required.**

### Training Mode — `train_multinews.py`

Train the MARL-MDS framework on the XSUM dataset:

```powershell
conda activate GPU-pytorch
python train_multinews.py
```

**Training process:**

1. Load XSUM dataset from HuggingFace (100 samples by default)
2. Initialize all three agents and Adam optimizer
3. Run training episodes with reference summaries
4. Compute reward (ROUGE, BERTScore, entity coverage) against reference summary
5. Backpropagate combined loss: `RL_loss_A1 + RL_loss_A3 + supervised_loss_A3`
6. Save trained checkpoint to `checkpoints/marl_mds_multinews.pt`

**Trained model location:** `checkpoints/marl_mds_multinews.pt`

**Note:** Training requires GPU for reasonable speed. The script uses the GPU-pytorch conda environment.

### Inference Mode — `marl_trainer.py` (Using Trained Model)

Run inference with the trained checkpoint:

```powershell
python marl_trainer.py
```

**Inference process:**

1. Load trained checkpoint from `checkpoints/marl_mds_multinews.pt`
2. Run inference on sample documents
3. Generate abstractive summaries using the trained model
4. Print summary and selected sentences

**No reference summaries required** - the trained model has learned what good summarization looks like from the training phase.

---

## Example Input and Output

### Demo Example — Solar System (bundled in `master_demo.py`)

**Input:** 14 short documents covering the Sun, eight planets, asteroid belt, Kuiper Belt, comets, and space exploration missions.

**Agent 1 selects 7 sentences**, including:
- Sun mass and life-sustaining energy
- The eight recognized planets
- Jupiter's Great Red Spot
- Mars as the Red Planet
- Mercury's orbital period
- Asteroid belt location
- Recent space mission goals

**Final abstractive summary (`final_summary.txt`):**

```text
The Sun contains more than 99 percent of the solar system's total mass. There are eight recognized planets in the Solar System: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. The asteroid belt lies between Mars and Jupiter and contains millions of rocky objects.
```

### Training Example — Apple Inc. (bundled in `marl_trainer.py`)

**Input:** 3 documents about Apple (headquarters, founding, AI/autonomous vehicle investments).

**Reference summary (used for reward only):**

```text
Apple is a major technology company headquartered in Cupertino. It was founded by Steve Jobs and Steve Wozniak and is investing in AI and autonomous vehicles.
```

**Typical demo output on Apple docs via `master_demo.py`:**

```text
Apple is an american multinational technology company headquartered in Cupertino, California. Recent reports suggest apple is investing heavily in artificial intelligence and autonomous vehicles.
```

---

## Technical Stack and Models

### Models Used

**Agent 1 — Packing Agent:**
- **BERT-base-uncased:** 110M parameters, 768-dimensional embeddings
- Usage: CLS-token sentence encoding and salience scoring
- Source: Hugging Face Transformers

**Agent 2 — Aggregation Agent:**
- **Custom Transformer:** Entity-aligned multi-head attention + PD-RoPE
- 768-dimensional hidden states (BERT-compatible)
- Usage: Cross-document fusion with entity bias

**Agent 3 — Generator Agent:**
- **T5-base:** 220M parameters, general-purpose seq2seq
- Usage: Abstractive summarization with `"summarize:"` prefix
- Source: Hugging Face Transformers

**Alternative Models Considered (not used in final implementation):**
- BART-large-cnn — CNN/DailyMail fine-tuned
- PEGASUS-XSUM — extreme summarization dataset fine-tuned
- FLAN-T5-base — instruction-tuned variant

### Libraries and Frameworks

| Library | Role |
|---------|------|
| **PyTorch** | Neural network implementation, training, inference |
| **Transformers (Hugging Face)** | BERT, T5 loading and tokenization |
| **spaCy** | Named entity recognition in Agent 2 (`en_core_web_sm`) |
| **ROUGE-score** | Summarization evaluation in reward function |
| **BERTScore** | Semantic similarity in reward function |

---

## Implementation Details

### Agent 1 — Sentence Selection Flow

**Files:**
- `Agent_1_Packing_Agent/src/model/summarizer.py` — BertSum model
- `Agent_1_Packing_Agent/src/utils/selection.py` — Trigram blocking and top-k selection
- `master_demo.py` — Demo-specific encoding and scoring helpers

**Process:**

```text
Documents → split_sentences()
         → encode with [CLS] sent [SEP] tokens
         → BertSum actor_critic salience scores
         → blend with heuristic scores (demo only)
         → rank + trigram blocking
         → top-k sentence indices
         → CLS embeddings for selected sentences
```

**Key code (`master_demo.py`):**

```python
summary_sentence_count = compute_summary_sentence_count(
    len(all_sentences),
    len(documents),
)
indices, selected_sentences = select_indices_with_trigram_blocking(
    all_sentences, scores, k=summary_sentence_count
)
packed_embeddings = cls_embs[indices].unsqueeze(0)  # [1, k, 768]
```

### Agent 2 — Cross-Document Fusion

**File:** `Agent_2_Document_Agregation_Agent/src/model/aggregation_agent.py`

**Process:**
1. Extract entities from each selected sentence (spaCy or regex fallback)
2. Build entity alignment matrix (shared entities → attention bias)
3. Apply entity-aligned multi-head attention with PD-RoPE
4. Return fused context tensor `[batch, k, 768]`

### Agent 3 — Abstractive Generation

**File:** `Agent_3_Faithful_Generator_Agent/src/model/generator_agent.py`

**Process:**

```text
Selected sentences
      │
      ├─ ≤8 sentences → single-pass T5 beam search
      │
      └─ >8 sentences → hierarchical chunk summarization
              │
              ├─ chunk 1 (3 sents) → partial summary
              ├─ chunk 2 (3 sents) → partial summary
              └─ merge partials → final T5 pass
      │
      ▼
_post_process_summary() → final text
```

**Key generation method:**

```python
def _decode_summary(self, text, device, max_length=80, num_beams=6, length_penalty=2.0):
    # Tokenize with "summarize:" prefix (T5)
    # Beam search with no_repeat_ngram_size=3
    # Post-process capitalization and punctuation
    return self._post_process_summary(decoded_text)
```

---

## Reinforcement Learning Framework

### Reward Function

**File:** `Agent_3_Faithful_Generator_Agent/src/utils/reward_utils.py`

**Reward components:**

| Component | Weight | Purpose |
|-----------|--------|---------|
| ROUGE-1 | 0.30 | Unigram overlap with reference |
| ROUGE-2 | 0.20 | Bigram overlap with reference |
| ROUGE-L | 0.20 | Longest common subsequence |
| BERTScore F1 | 0.15 | Semantic similarity |
| Entity coverage | 0.05 | Key entities preserved |
| Topic coverage | 0.05 | Topic overlap with source |
| Redundancy penalty | −0.05 | Penalize repetitive output |

### Training Loop

**File:** `marl_trainer.py`

```python
class MARLMdsTrainer:
    def train_step(self, documents, reference_summary):
        # 1. Agent 1: stochastic sentence selection (RL)
        # 2. Agent 2: fuse selected embeddings
        # 3. Agent 3: generate summary (RL params if reference given)
        # 4. Compute reward vs reference
        # 5. Backprop: rl_loss_a1 + rl_loss_a3 + supervised_loss_a3
        return metrics  # reward, loss, summary, selected_sentences
```

**Loss composition (when reference provided):**

```text
total_loss = RL_loss_Agent1 + RL_loss_Agent3 + supervised_loss_Agent3
```

---

## Project Structure

```text
Extractive-Summarisation/
├── Agent_1_Packing_Agent/
│   ├── src/
│   │   ├── model/
│   │   │   └── summarizer.py          # BertSum model
│   │   ├── utils/
│   │   │   └── selection.py           # Trigram blocking + top-k selection
│   │   └── training/
│   │       └── rl_policy.py           # RL policy functions
│   └── tests/
├── Agent_2_Document_Agregation_Agent/
│   ├── src/
│   │   ├── model/
│   │   │   └── aggregation_agent.py   # Cross-document aggregation
│   │   └── utils/
│   │       ├── embeddings.py          # PD-RoPE implementation
│   │       └── entity_utils.py        # Entity extraction and alignment
│   └── tests/
├── Agent_3_Faithful_Generator_Agent/
│   ├── src/
│   │   ├── model/
│   │   │   └── generator_agent.py     # T5 abstractive generator
│   │   └── utils/
│   │       ├── reward_utils.py        # Reward function
│   │       └── decoding_utils.py      # Self-healing beam search
│   └── tests/
├── marl_trainer.py                     # RL training entry point
├── master_demo.py                      # Inference demo entry point
├── final_summary.txt                   # Latest demo output
└── README.md                           # This file
```

---

## Running Tests

### Agent 2 Tests

```powershell
python -m Agent_2_Document_Agregation_Agent.src.tests.test_agent
```

### Agent 3 Tests

```powershell
python -m Agent_3_Faithful_Generator_Agent.src.tests.test_reward
python -m Agent_3_Faithful_Generator_Agent.src.tests.test_generator
```

---

## Presentation Guide (For Internship Coordinator)

### 1. Problem Statement

- **Multi-document summarization** requires combining facts spread across many sources
- Single-document summarizers miss cross-document relationships
- **Abstractive** summarization (rephrasing) is harder but more readable than extractive copy-paste
- The system must stay **faithful** — no hallucinated facts

### 2. Solution: Three-Agent Pipeline

| Agent | Role | Key Technique |
|-------|------|---------------|
| Agent 1 | Pack salient sentences | BertSum + trigram blocking |
| Agent 2 | Fuse cross-doc context | Entity-aligned attention + PD-RoPE |
| Agent 3 | Generate summary | T5-base abstractive generation |

Explain the flow with the diagram at the top of this README, then live-run:

```powershell
conda activate GPU-pytorch
python master_demo.py
type final_summary.txt
```

### 3. Technical Innovations to Highlight

- **Entity-aligned attention** — explicitly links sentences that mention the same entities across documents
- **PD-RoPE** — improved positional encoding for longer fused contexts
- **Blended salience scoring** — robust sentence selection even before RL training converges
- **Adaptive sentence count formula** — scales with document count instead of a fixed cap of 3
- **Hierarchical T5 decoding** — handles large packed inputs without garbled output
- **MARL framework** — both packing (Agent 1) and generation parameters (Agent 3) can be optimized via reward

### 4. Demo vs Training — What to Say

> "For the live demo I use `master_demo.py`, which runs inference only and produces the best summary output. For the research/training side, `marl_trainer.py` shows how the agents learn from reference summaries using reinforcement learning. Same architecture, different entry points."

### 5. Results to Show

- **Input:** 14 Solar System documents (~16 sentences total)
- **Agent 1 output:** 7 selected salient sentences (print from console)
- **Final summary:** Sun mass + eight planets + asteroid belt in 3 coherent sentences
- **Compare** with what the old broken formula produced (only 3 sentences selected, poorer coverage)

### 6. Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Sentence selection capped at 3 regardless of input size | `compute_summary_sentence_count()` scales with documents and sentences |
| Untrained BERT salience head gives noisy scores | Blend BERT (35%) with heuristic salience (65%) |
| T5 produces garbled run-on text on long inputs | Tuned beam search + hierarchical chunk summarization for >8 sentences |
| Cross-document entity relationships ignored | Entity-aligned attention bias in Agent 2 |
| Domain-specific templates don't generalize | T5-base general-purpose generation |

### 7. Limitations (Be Transparent)

- RL policies are not yet trained on large datasets (CNN/DailyMail, XSUM)
- Demo uses CPU; training benefits from GPU
- T5-base output quality depends on pre-training — not fine-tuned on this project's data
- `marl_trainer.py` still uses the older `min(3, …)` selection cap (intentional for training scaffold)
- No deployed web API yet

### 8. Future Work

- Train RL policies on large summarization benchmarks
- Align `marl_trainer.py` sentence count formula with demo improvements
- Fine-tune T5-base on domain-specific corpora
- Implement self-healing decoding with prefix validation
- Deploy as a web service with REST API
- Add GPU support to `master_demo.py` for faster inference

---

## Current Scope and Status

### Completed

- Three-agent architecture (Packing → Aggregation → Generation)
- Agent 1: BERT-based sentence selection with RL policy and demo heuristics
- Agent 2: Entity-aligned attention with PD-RoPE
- Agent 3: T5-base generation with post-processing and hierarchical decoding
- Reward function (ROUGE + BERTScore + entity/topic coverage)
- Training loop with actor-critic updates (`marl_trainer.py`)
- End-to-end demo with improved selection and generation (`master_demo.py`)

### Known Limitations

- RL weights not trained on production-scale data
- `marl_trainer.py` sentence cap (`min(3, …)`) differs from demo formula
- Entity extraction relies on spaCy small model or regex fallback
- No saved checkpoint for a fully trained end-to-end pipeline

---

## References

**Models:**
- BERT: Devlin et al. (2019) — "BERT: Pre-training of Deep Bidirectional Transformers"
- T5: Raffel et al. (2019) — "Exploring the Limits of Transfer Learning"
- BART: Lewis et al. (2019) — "BART: Denoising Sequence-to-Sequence Pre-training"
- PEGASUS: Zhang et al. (2020) — "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization"

**Techniques:**
- PD-RoPE: Positional Disentangling Rotary Positional Embeddings
- Entity-Aligned Attention: Custom cross-document attention bias
- Actor-Critic RL: Policy gradient with value baseline

**Datasets (for future training):**
- CNN/DailyMail
- XSUM (Extreme Summarization)
- Multi-News

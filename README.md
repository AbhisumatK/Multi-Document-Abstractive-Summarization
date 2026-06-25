# MARL-MDS: Multi-Agent Reinforcement Learning for Multi-Document Summarization

## Overview
This repository contains the implementation of a robust and faithful Multi-Document Summarization (MDS) framework. Unlike traditional single-model approaches, this system utilizes three specialized agents working together under a Reinforcement Learning (RL) structure to ensure coherence, entity consistency, and factual faithfulness.

## Architecture

The framework is divided into three distinct agents:

### 1. Agent 1: Packing Agent (Hamilton Packing)
- **Objective**: Select the most salient sentences across multiple documents to fit within the transformer's context window.
- **Mechanism**: Utilizes BertSum for saliency scoring combined with Trigram Blocking to reduce redundancy.

### 2. Agent 2: Cross-Document Aggregation Agent (A₂)
- **Objective**: Fuse information across documents and resolve inter-document inconsistencies.
- **Key Technologies**:
  - **PD-RoPE**: Positional Disentangling Rotary Positional Embeddings for stable long-context attention.
  - **Entity-Aligned Attention**: Biases the model to prioritize relationships between shared entities (people, places, organizations) across documents.

### 3. Agent 3: Faithful Generator Agent (A₃)
- **Objective**: Generate a coherent abstractive summary while maintaining factual consistency.
- **Mechanism**: 
  - **BART-Style Decoder**: Specifically configured for conditional generation from fused representations.
  - **Self-Healing RL Decoding**: A custom decoding strategy that filters out "hallucinations" (unfactual tokens) in real-time during the generation process.

## Installation

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install torch transformers spacy rouge-score bert-score
python -m spacy download en_core_web_sm
```

## Running the Project

### End-to-End Demo
To test the full pipeline (Agent 1 -> Agent 2 -> Agent 3) on your own documents:
1. Open `master_demo.py` and modify the `my_docs` list with your text.
2. Run the demo:
   ```bash
   python master_demo.py
   ```
3. The result will be saved to `final_summary.txt`.

### Running Verification Tests
Each module has its own test suite:
- **Agent 2 Tests**: 
  ```bash
  python -m Agent_2_Document_Agregation_Agent.src.tests.test_agent
  ```
- **Agent 3 Tests**:
  ```bash
  python -m Agent_3_Faithful_Generator_Agent.src.tests.test_reward
  python -m Agent_3_Faithful_Generator_Agent.src.tests.test_generator
  ```

## Reinforcement Learning Training
The agents are designed to be fine-tuned using the `SummarizationReward` utility (located in `Agent_3_Faithful_Generator_Agent/src/utils/reward_utils.py`), which combines ROUGE and BERTScore to provide a dense feedback signal for the MARL training loop.

---
**Note**: The demo script uses pre-trained weights for the encoder/decoder and transparent initialization for the custom aggregation layers. For production-grade summaries, the full RL training pass on your specific dataset is recommended to align the agents' shared embedding space.

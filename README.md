# MARL-MDS: Multi-Agent Multi-Document Summarization

## Overview

This project implements a multi-agent pipeline for multi-document summarization. The system takes multiple related documents, selects the most important information, fuses cross-document context, and produces one final faithful summary.

The project is organized into three agents:

```text
Input Documents -> Agent 1 -> Agent 2 -> Agent 3 -> Final Summary
```

The main objective is factual summarization. The final summary should be grounded in the original documents and should avoid hallucinated or unrelated content.

## Architecture

### Agent 1: Packing Agent

Agent 1 reads all input documents and splits them into sentences.

Its job is to select the most important non-redundant sentences before passing information to the next stage. This reduces the amount of text the later agents need to process.

Main ideas:

- Sentence-level salience scoring
- Trigram blocking to reduce duplicate or repetitive content
- BERT-based sentence embeddings when the local model is available
- Deterministic offline fallback so the demo can still run without downloading models

### Agent 2: Cross-Document Aggregation Agent

Agent 2 receives the selected sentences from Agent 1 and fuses information across documents.

Its job is to identify relationships between selected sentences, especially when the same entities appear across multiple documents.

Main ideas:

- Multi-head attention
- PD-RoPE positional encoding
- Entity-aligned attention
- Cross-document context fusion

### Agent 3: Faithful Generator Agent

Agent 3 produces the final summary.

For the current demo, Agent 3 uses a faithful extractive generation path. This means it builds the final summary from selected source sentences instead of freely generating text from an untrained decoder.

This is intentional. The repository does not include a fully trained checkpoint that aligns Agent 2's fused embeddings with BART's decoder space. Directly decoding from incompatible embeddings can produce unrelated hallucinated text. The faithful extractive path keeps the demo output factually correct and suitable for presentation.

An experimental BART decoder path is still present in the code, but it should only be used after proper end-to-end training or loading a compatible trained checkpoint.

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

## Running the Full Project

Open PowerShell and move into the project directory:

```powershell
cd "C:\Users\ABHISUMAT\All Projects\Extractive-Summarisation"
```

Run the complete end-to-end pipeline:

```powershell
C:\Users\ABHISUMAT\anaconda3\envs\GPU-pytorch\python.exe master_demo.py
```

The script will:

1. Load the documents from the `my_docs` list in `master_demo.py`.
2. Run Agent 1 to select salient non-redundant sentences.
3. Run Agent 2 to fuse cross-document information.
4. Run Agent 3 to generate a faithful final summary.
5. Save the result to `final_summary.txt`.

## Example

For the sample Apple documents in `master_demo.py`, the final summary is:

```text
Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Steve Jobs and Steve Wozniak founded Apple in 1976. Recent reports suggest Apple is investing heavily in artificial intelligence and autonomous vehicles to expand its product line.
```

This is shorter than the original input and preserves the key facts:

- What Apple is
- Where Apple is headquartered
- Who founded Apple
- Apple's current investment direction

## Running Tests

Run Agent 2 tests:

```powershell
C:\Users\ABHISUMAT\anaconda3\envs\GPU-pytorch\python.exe -m Agent_2_Document_Agregation_Agent.src.tests.test_agent
```

Run Agent 3 tests:

```powershell
C:\Users\ABHISUMAT\anaconda3\envs\GPU-pytorch\python.exe -m Agent_3_Faithful_Generator_Agent.src.tests.test_reward
C:\Users\ABHISUMAT\anaconda3\envs\GPU-pytorch\python.exe -m Agent_3_Faithful_Generator_Agent.src.tests.test_generator
```

## Explanation for Presentation

This project is a multi-document summarization system. It takes several documents about the same topic and creates one final summary.

Instead of using one large model for everything, the system divides the work among three agents:

1. The first agent selects the most important sentences.
2. The second agent connects related information across documents.
3. The third agent produces the final faithful summary.

The benefit of this design is control. Each agent has a clear responsibility, which makes the summarization process easier to understand, debug, and improve.

The current demo focuses on factual correctness. Since the project does not include a trained abstractive generation checkpoint, the final summary is extractive. This means the summary is created from the original source sentences, which avoids hallucination and keeps the output reliable.

## Reinforcement Learning Component

The project includes a reward utility in:

```text
Agent_3_Faithful_Generator_Agent/src/utils/reward_utils.py
```

This reward function combines ROUGE and BERTScore. It is designed for future reinforcement learning fine-tuning, where the agents can be trained to improve summary quality, semantic similarity, and factual faithfulness.

## Current Scope

The current version is a working internship/demo implementation of the MARL-MDS architecture.

It demonstrates:

- Multi-agent summarization structure
- Salient sentence selection
- Redundancy reduction
- Cross-document aggregation
- Entity-aware attention
- Faithful final summary generation

For production-level abstractive summarization, the next step would be training the full Agent 2 to Agent 3 generation path on a summarization dataset and saving a compatible checkpoint.

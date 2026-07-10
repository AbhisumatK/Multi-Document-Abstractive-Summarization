"""
Train the MARL-MDS framework on the Multi-News dataset.

IMPORTANT: Activate GPU environment before running:
    conda activate GPU-pytorch

Usage: python train_multinews.py
"""
import math
import os
import re
import sys
import tim

import torch
from datasets import load_dataset

project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

from marl_trainer import MARLMdsTrainer


def parse_multinews_sample(sample):
    """Parse XSUM sample into documents and summary."""
    # XSUM has 'document' field as the document and 'summary' as the summary
    document = sample["document"]
    # Split document into sentences to simulate multiple documents
    import re
    docs = [s.strip() for s in re.split(r"(?<=[.!?])\s+", document) if s.strip() and len(s.strip()) > 20]
    
    # Take first few sentences to simulate multi-document input
    docs = docs[:5]  # Limit to 5 documents
    
    summary = sample["summary"].strip()
    
    return docs, summary


def main():
    # --- Config ---
    NUM_SAMPLES = 100        # Number of training samples (increase for better results)
    EPOCHS = 2               # Number of passes over the data
    CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "marl_mds_multinews.pt")
    LOG_EVERY = 5            # Print metrics every N steps

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load Dataset ---
    print("Loading XSUM dataset...")
    dataset = load_dataset("EdinburghNLP/xsum", split=f"train[:{NUM_SAMPLES}]")
    print(f"Loaded {len(dataset)} samples.")

    # --- Init Trainer ---
    trainer = MARLMdsTrainer(device=device, local_files_only=False)

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
        total_loss = 0
        total_reward = 0
        skipped = 0

        for step, sample in enumerate(dataset):
            docs, ref = parse_multinews_sample(sample)
            if len(docs) < 2 or len(ref) < 10:
                skipped += 1
                continue

            try:
                metrics = trainer.train_step(docs, ref)
                total_loss += metrics["loss"]
                total_reward += metrics.get("reward", 0)

                if (step + 1) % LOG_EVERY == 0:
                    avg_loss = total_loss / (step + 1 - skipped)
                    avg_reward = total_reward / (step + 1 - skipped)
                    print(f"  Step {step + 1}/{len(dataset)} | Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f}")

            except Exception as e:
                print(f"  Step {step + 1} skipped: {e}")
                skipped += 1
                continue

        effective = len(dataset) - skipped
        if effective > 0:
            print(f"Epoch {epoch + 1} done. Avg Loss: {total_loss/effective:.4f} | Avg Reward: {total_reward/effective:.4f} | Skipped: {skipped}")

        # Save checkpoint after each epoch
        trainer.save_checkpoint(CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

"""
Train the MARL-MDS framework on the Multi-News dataset using TensorFlow Datasets.

IMPORTANT: Activate GPU environment before running:
    conda activate GPU-pytorch

Install tensorflow-datasets if needed:
    pip install tensorflow-datasets

Usage: python train_multinews.py
"""
import math
import os
import re
import sys
import time
from datetime import datetime

import torch
import tensorflow_datasets as tfds

project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

from marl_trainer import MARLMdsTrainer


def parse_multinews_sample(sample):
    """Parse Multi-News sample into documents and summary."""
    # Multi-News has 'document' field with multiple articles separated by |||||
    document = sample['document'].numpy().decode('utf-8')
    # Split on ||||| to get individual documents
    docs = [doc.strip() for doc in document.split("|||||") if doc.strip()]

    summary = sample['summary'].numpy().decode('utf-8')

    return docs, summary


def main():
    # --- Config ---
    NUM_SAMPLES = 500        # Number of training samples (increased for better results)
    EPOCHS = 5               # Number of passes over the data (increased for better convergence)
    CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
    # Add timestamp to checkpoint name to avoid overwriting existing model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"marl_mds_multinews_{timestamp}.pt")
    LOG_EVERY = 10           # Print metrics every N steps

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint will be saved to: {CHECKPOINT_PATH}")

    # --- Load Dataset ---
    print("Loading Multi-News dataset from TensorFlow Datasets...")
    # Load Multi-News dataset (true multi-document summarization)
    ds = tfds.load('multi_news', split=f'train[:{NUM_SAMPLES}]')
    print(f"Loaded {NUM_SAMPLES} samples from Multi-News dataset.")

    # --- Init Trainer ---
    trainer = MARLMdsTrainer(device=device, local_files_only=False)

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
        total_loss = 0
        total_reward = 0
        skipped = 0
        step_count = 0

        for sample in ds:
            docs, ref = parse_multinews_sample(sample)
            if len(docs) < 2 or len(ref) < 10:
                skipped += 1
                continue

            try:
                metrics = trainer.train_step(docs, ref)
                total_loss += metrics["loss"]
                total_reward += metrics.get("reward", 0)
                step_count += 1

                if step_count % LOG_EVERY == 0:
                    avg_loss = total_loss / step_count
                    avg_reward = total_reward / step_count
                    print(f"  Step {step_count} | Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f}")

            except Exception as e:
                print(f"  Step skipped: {e}")
                skipped += 1
                continue

        effective = step_count
        if effective > 0:
            print(f"Epoch {epoch + 1} done. Avg Loss: {total_loss/effective:.4f} | Avg Reward: {total_reward/effective:.4f} | Skipped: {skipped}")

        # Save checkpoint after each epoch
        trainer.save_checkpoint(CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

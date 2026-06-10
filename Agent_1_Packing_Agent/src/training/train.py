import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data_preprocessing.loader import get_dataset, preprocess_data
from src.model.summarizer import BertSum
from src.utils.selection import greedy_selection_with_trigram_blocking, select_indices_with_trigram_blocking
import torch.optim as optim
from src.training.evaluate import evaluate
import multiprocessing
from tqdm import tqdm
from rouge_score import rouge_scorer

def train(model, dataloader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        if batch is None:
            continue

        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        cls_positions = batch['cls_positions'].to(device)
        labels = batch['labels'].to(device)

        salience_scores = model(input_ids, attention_mask, cls_positions)

        min_len = min(salience_scores.size(1), labels.size(1))
        salience_scores = salience_scores[:, :min_len]
        labels = labels[:, :min_len]

        loss = loss_function(salience_scores, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # Compute ROUGE between model summary and oracle summary (logging only)
        batch_oracle = batch['oracle_summary']
        batch_sentences = batch['original_sentences']
        batch_scores = salience_scores.detach().cpu().tolist()
        rouge_vals = []
        for sentences, scores, oracle_summary in zip(batch_sentences, batch_scores, batch_oracle):
            _, selected_sentences = select_indices_with_trigram_blocking(sentences, scores)
            model_summary = " ".join(selected_sentences)
            rouge_vals.append(rouge.score(oracle_summary, model_summary)['rouge1'].fmeasure)
        avg_rouge = sum(rouge_vals) / max(len(rouge_vals), 1)
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'rouge1': f"{avg_rouge:.4f}"})

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0

if __name__ == '__main__':
    # Hyperparameters
    EPOCHS = 1 # For demonstration
    BATCH_SIZE = 1 # Reduced from 4 due to OOM (Out Of Memory, Exit Code 137)
    LEARNING_RATE = 2e-5

    # Load data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Check if Mac GPU (MPS) is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading and preprocessing data... (This will be fast if cached)")
    
    # Use all available cores for mapping. The results are automatically cached.
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores for preprocessing.")

    train_dataset_raw = get_dataset('train')
    train_dataset = train_dataset_raw.map(
        lambda x: preprocess_data(x, tokenizer),
        num_proc=num_cores
    )
    
    # Filter out examples that couldn't be processed
    train_dataset = train_dataset.filter(lambda x: x is not None)
    # Use only the first 20,000 rows for faster training
    train_dataset = train_dataset.select(range(min(20000, len(train_dataset))))

    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
        cls_positions = [torch.tensor(item['cls_positions'], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item['labels'], dtype=torch.float32) for item in batch]

        max_seq = max(t.size(0) for t in input_ids)
        max_sents = max(t.size(0) for t in cls_positions)

        input_ids_padded = [torch.nn.functional.pad(t, (0, max_seq - t.size(0)), value=0) for t in input_ids]
        attention_mask_padded = [torch.nn.functional.pad(t, (0, max_seq - t.size(0)), value=0) for t in attention_mask]
        cls_positions_padded = [torch.nn.functional.pad(t, (0, max_sents - t.size(0)), value=0) for t in cls_positions]
        labels_padded = [torch.nn.functional.pad(t, (0, max_sents - t.size(0)), value=0.0) for t in labels]

        return {
            'input_ids': torch.stack(input_ids_padded),
            'attention_mask': torch.stack(attention_mask_padded),
            'cls_positions': torch.stack(cls_positions_padded),
            'labels': torch.stack(labels_padded),
            'original_sentences': [item['original_sentences'] for item in batch],
            'oracle_summary': [item['oracle_summary'] for item in batch],
        }

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = BertSum().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.BCELoss()

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train(model, train_dataloader, optimizer, loss_function, device)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}")

    # Example of how to use the trained model for prediction
    model.eval()
    
    # Get a single sample for prediction demonstration
    sample = train_dataset[0]

    with torch.no_grad():
        sample_input_ids = torch.tensor(sample['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        sample_attention_mask = torch.tensor(sample['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
        sample_cls_positions = torch.tensor(sample['cls_positions'], dtype=torch.long).unsqueeze(0).to(device)
        salience_scores = model(
            sample_input_ids,
            sample_attention_mask,
            sample_cls_positions
        ).cpu()
    
    original_sentences = sample['original_sentences']
    # Use the new selection logic
    summary = greedy_selection_with_trigram_blocking(original_sentences, salience_scores.squeeze(0).tolist())
    
    print("\n--- Example Summary ---")
    print("Original sentence count:", len(original_sentences))
    print("Selected sentence count:", len(summary.split('. '))) # Approximate
    print("\nSummary:\n", summary)

    # --- Evaluation Phase ---
    print("\nStarting evaluation on the test set...")
    avg_rouge_scores = evaluate(model, tokenizer, device)

    print("\n--- Evaluation Results ---")
    print(f"ROUGE-1: {avg_rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {avg_rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {avg_rouge_scores['rougeL']:.4f}")

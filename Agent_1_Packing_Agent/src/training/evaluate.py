import torch
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from tqdm import tqdm

from src.data_preprocessing.loader import get_dataset, preprocess_data
from src.utils.selection import greedy_selection_with_trigram_blocking

def evaluate(model, tokenizer, device):
    """
    Evaluates the model on the test set.
    """
    model.eval()
    
    # Load test data
    test_dataset = get_dataset('test').map(lambda x: preprocess_data(x, tokenizer))
    test_dataset = test_dataset.filter(lambda x: x is not None and len(x['cls_positions']) > 0 and len(x['original_sentences']) > 0)

    def collate_fn(batch):
        # This collate function will handle single items since batch_size is 1
        item = batch[0]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long).unsqueeze(0),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long).unsqueeze(0),
            'cls_positions': torch.tensor(item['cls_positions'], dtype=torch.long).unsqueeze(0),
            'original_sentences': [item['original_sentences']],
            'oracle_summary': [item['oracle_summary']]
        }

    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cls_positions = batch['cls_positions'].to(device)
            original_sentences = batch['original_sentences'][0]
            oracle_summary = batch['oracle_summary'][0]

            salience_scores = model(input_ids, attention_mask, cls_positions).squeeze(0).tolist()
            
            generated_summary = greedy_selection_with_trigram_blocking(original_sentences, salience_scores)
            
            scores = scorer.score(oracle_summary, generated_summary)
            total_scores['rouge1'] += scores['rouge1'].fmeasure
            total_scores['rouge2'] += scores['rouge2'].fmeasure
            total_scores['rougeL'] += scores['rougeL'].fmeasure

    num_samples = len(test_dataloader)
    avg_scores = {key: value / num_samples for key, value in total_scores.items()}
    
    return avg_scores

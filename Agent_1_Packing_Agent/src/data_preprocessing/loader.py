from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from src.utils.label_generator import create_labels
import nltk
import ssl
import torch

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # For Python versions that don't have _create_unverified_context
    pass
else:
    # For others, like macOS, where certificate verification fails
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def get_dataset(split='train'):
    """
    Loads and preprocesses the CNN/Daily Mail dataset.
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
    return dataset

def preprocess_data(example, tokenizer):
    """
    Tokenizes article for BERTSUM and creates labels.
    """
    sentences = sent_tokenize(example['article'])
    if not sentences:
        return None

    # Create labels + oracle summary using per-sentence ROUGE + trigram blocking
    labels, oracle_summary, sentence_scores = create_labels(sentences, example['highlights'])

    # Build CLS/SEP formatted input string
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    input_text = " ".join([f"{cls_token} {s} {sep_token}" for s in sentences])

    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors="pt",
        add_special_tokens=False,
    )

    cls_positions = (inputs['input_ids'][0] == tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
    num_sents = len(cls_positions)
    if num_sents == 0:
        return None

    # Align lengths with CLS positions in case of truncation
    sentences = sentences[:num_sents]
    labels = labels[:num_sents]
    sentence_scores = sentence_scores[:num_sents]
    oracle_summary = " ".join([s for s, l in zip(sentences, labels) if l == 1])

    example['input_ids'] = inputs['input_ids'].squeeze(0)
    example['attention_mask'] = inputs['attention_mask'].squeeze(0)
    example['cls_positions'] = cls_positions
    example['original_sentences'] = sentences
    example['labels'] = torch.tensor(labels, dtype=torch.float32)
    example['oracle_summary'] = oracle_summary
    example['sentence_scores'] = sentence_scores

    return example

if __name__ == '__main__':
    # Example of loading and preprocessing
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = get_dataset('train')
    sample = train_dataset[0]
    processed_sample = preprocess_data(sample, tokenizer)
    
    print("Input IDs shape:", processed_sample['input_ids'].shape)
    print("Attention mask shape:", processed_sample['attention_mask'].shape)
    print("CLS positions:", processed_sample['cls_positions'])
    print("Original sentences:", processed_sample['original_sentences'])
    print("Labels shape:", processed_sample['labels'].shape)
    print("Labels:", processed_sample['labels'])

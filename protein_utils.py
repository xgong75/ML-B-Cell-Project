import pandas as pd
import torch 
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# Store and format data for DataLoader
# class ProteinDataset(Dataset):
#     def __init__(self, sequences, peptides, start_positions, end_positions, labels, tokenizer, covariates, max_len=512):
#         self.sequences = sequences
#         self.peptides = peptides
#         self.start_positions = start_positions
#         self.end_positions = end_positions
#         self.labels = labels
#         self.covariates = covariates
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         # Tokenize parent sequence
#         parent_tokens = self.tokenizer(
#             self.sequences[idx],
#             max_length=self.max_len,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         )

#         # Tokenize peptide sequence
#         peptide_tokens = self.tokenizer(
#             self.peptides[idx],
#             max_length=self.max_len,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         )

#         # Extract covariates and labels
#         covariates = torch.tensor(self.covariates[idx], dtype=torch.float32)
#         label = self.labels[idx]

#         return {
#             "parent_tokens": {key: val.squeeze(0) for key, val in parent_tokens.items()},
#             "peptide_tokens": {key: val.squeeze(0) for key, val in peptide_tokens.items()},
#             "start_positions": self.start_positions[idx],
#             "end_positions": self.end_positions[idx],
#             "covariates": covariates,
#             "labels": label,
#         }


def load_data(filepath):
    """Load protein and peptide sequences along with metadata."""
    data = pd.read_csv(filepath)
    return data['protein_seq'].tolist(), data['peptide_seq'].tolist(), data['start_position'].tolist(), data['end_position'].tolist(), data['target'].tolist()

def align_positions_to_tokens(parent_seq, peptide_seq, tokenizer):
    """
    Align raw start and end positions to tokenized sequence positions.
    Args:
        parent_seq (str): Raw parent sequence.
        peptide_seq (str): Raw peptide subsequence.
        tokenizer: Tokenizer (e.g., BERT tokenizer).
    Returns:
        (start_idx, end_idx): Tokenized start and end indices.
    """
    parent_tokens = tokenizer.tokenize(parent_seq)
    peptide_tokens = tokenizer.tokenize(peptide_seq)

    # Locate the start and end indices of the peptide tokens in the parent tokens
    try:
        start_idx = parent_tokens.index(peptide_tokens[0])  # Start of the peptide in tokenized parent
        end_idx = start_idx + len(peptide_tokens) - 1       # End of the peptide in tokenized parent
        return start_idx, end_idx
    except ValueError:
        print(f"Error: Peptide sequence {peptide_seq} not found in parent sequence {parent_seq}")
        return None, None

def preprocess_data(parent_sequences, peptide_sequences, tokenizer):
    """
    Preprocess data to align subsequence positions with tokenized parent sequence.
    Args:
        parent_sequences: List of parent sequences (raw).
        peptide_sequences: List of peptide subsequences (raw).
        tokenizer: Tokenizer (e.g., BERT tokenizer).
    Returns:
        aligned_start_positions: List of aligned start positions.
        aligned_end_positions: List of aligned end positions.
    """
    aligned_start_positions = []
    aligned_end_positions = []
    for parent_seq, peptide_seq in zip(parent_sequences, peptide_sequences):
        start_idx, end_idx = align_positions_to_tokens(parent_seq, peptide_seq, tokenizer)
        if start_idx is not None and end_idx is not None:
            aligned_start_positions.append(start_idx)
            aligned_end_positions.append(end_idx)
        else:
            raise ValueError(f"Alignment failed for parent: {parent_seq}, peptide: {peptide_seq}")
    return aligned_start_positions, aligned_end_positions

# Output embeddings to a file    
def save_embeddings(embeddings, filepath):
    """Save embeddings to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(embeddings, filepath)

def mask_non_peptide_regions(input_ids, attention_mask, start_positions, end_positions):
        """Mask all tokens outside the peptide region."""
        masked_attention = attention_mask.clone()
        for i, (start, end) in enumerate(zip(start_positions, end_positions)):
            masked_attention[i, :start] = 0
            masked_attention[i, end + 1:] = 0
        return input_ids, masked_attention

import torch
from tqdm import tqdm

def extract_final_embeddings(model, dataloader, mode, device="cpu"):
    """
    Extract embeddings after the attention layer but before concatenating with covariates.
    
    Args:
        model: The trained model instance.
        dataloader: DataLoader for the dataset.
        mode: The embedding mode ("peptide_only", "subsequence", etc.).
        device: The device to run the model on ("cpu" or "cuda").
    
    Returns:
        final_embeddings: Tensor of embeddings for all sequences.
    """
    model.eval()  # Set the model to evaluation mode
    final_embeddings = []

    # Progress bar for extraction
    progress_bar = tqdm(dataloader, desc="Extracting Final Embeddings", unit="batch", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            # Move input data to the device
            parent_tokens = {k: v.to(device) for k, v in batch["parent_tokens"].items()}
            peptide_tokens = {k: v.to(device) for k, v in batch["peptide_tokens"].items()}
            start_positions = batch["start_positions"]
            end_positions = batch["end_positions"]

            # Get embeddings after the attention layer
            logits, _ = model(
                mode=mode,
                parent_tokens=parent_tokens,
                peptide_tokens=peptide_tokens,
                start_positions=start_positions,
                end_positions=end_positions,
                covariates=None  # Exclude covariates
            )
            embeddings = model.apply_attention(logits)  # Extract embeddings after attention
            final_embeddings.append(embeddings.cpu())  # Move to CPU for storage

    # Combine all embeddings into a single tensor
    final_embeddings = torch.cat(final_embeddings, dim=0)  # Combine all batches
    return final_embeddings

# def evaluate_model(model, dataloader, mode, device="cpu",covariates=None):
#     model.eval()
#     correct = 0
#     total = 0
#     total_loss = 0

#     criterion = nn.BCEWithLogitsLoss()
#     progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch", leave=True)

#     with torch.no_grad():
#         for batch in progress_bar:
#             # Move data to device
#             parent_tokens = {k: v.to(device) for k, v in batch["parent_tokens"].items()}
#             peptide_tokens = {k: v.to(device) for k, v in batch["peptide_tokens"].items()}
#             start_positions = batch["start_positions"]
#             end_positions = batch["end_positions"]
#             covariates = batch["covariates"].to(device)  # Ensure covariates are on the same device
#             labels = batch["labels"].to(device)

#             # Forward pass
#             logits, _ = model(
#                 mode=mode,
#                 parent_tokens=parent_tokens,
#                 peptide_tokens=peptide_tokens,
#                 start_positions=start_positions,
#                 end_positions=end_positions,
#                 covariates=covariates
#             )

#             # Compute loss
#             loss = criterion(logits.squeeze(), labels.float())
#             total_loss += loss.item()

#             # Convert logits to probabilities
#             probabilities = torch.sigmoid(logits).squeeze()
#             predictions = (probabilities > 0.5).float()

#             # Calculate accuracy
#             correct += (predictions == labels).sum().item()
#             total += labels.size(0)

#     accuracy = 100 * correct / total
#     avg_loss = total_loss / len(dataloader)

#     print(f"Evaluation Accuracy: {accuracy:.2f}%")
#     print(f"Average Loss: {avg_loss:.4f}")

#     return accuracy, avg_loss
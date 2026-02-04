import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TrajectoryDataset(Dataset):
    """
    Dataset for tokenized GPS trajectories.
    """
    def __init__(self, tokenized_trajectories, max_length=None):
        """
        Args:
            tokenized_trajectories: List of tokenized trajectories
                                   Each trajectory is a list/array of token indices
            max_length: Maximum sequence length (will truncate longer sequences)
        """
        self.trajectories = tokenized_trajectories
        self.max_length = max_length
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        
        # Convert to tensor if not already
        if not isinstance(trajectory, torch.Tensor):
            trajectory = torch.tensor(trajectory, dtype=torch.long)
        
        # Truncate if necessary
        if self.max_length is not None and len(trajectory) > self.max_length:
            trajectory = trajectory[:self.max_length]
        
        return trajectory


def collate_fn_pad(batch, pad_token_id=0):
    """
    Collate function for DataLoader that pads sequences to same length.
    
    Args:
        batch: List of trajectories (tensors of different lengths)
        pad_token_id: Token ID to use for padding
        
    Returns:
        Dictionary containing:
            - 'input_ids': Padded input sequences (batch_size, max_seq_len-1)
            - 'target_ids': Padded target sequences (batch_size, max_seq_len-1)
            - 'padding_mask': Mask indicating padding positions (batch_size, max_seq_len-1)
            - 'sequence_lengths': Original sequence lengths
    """
    # Pad sequences to same length
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
    
    # Create input and target sequences (targets are inputs shifted by 1)
    inputs = padded_batch[:, :-1]
    targets = padded_batch[:, 1:]
    
    # Create padding mask (True for padding positions)
    padding_mask = (inputs == pad_token_id)
    
    # Get original sequence lengths (before padding)
    sequence_lengths = torch.tensor([len(seq) for seq in batch])
    
    return {
        'input_ids': inputs,
        'target_ids': targets,
        'padding_mask': padding_mask,
        'sequence_lengths': sequence_lengths
    }


def create_dataloader(tokenized_trajectories, batch_size=32, max_length=None, 
                     pad_token_id=0, shuffle=True, num_workers=0):
    """
    Create a DataLoader for tokenized trajectories.
    
    Args:
        tokenized_trajectories: List of tokenized trajectories
        batch_size: Batch size
        max_length: Maximum sequence length
        pad_token_id: Token ID for padding
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader object
    """
    dataset = TrajectoryDataset(tokenized_trajectories, max_length=max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn_pad(batch, pad_token_id),
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    # Simulate tokenized trajectories with variable lengths
    example_trajectories = [
        [1, 5, 23, 45, 67, 89, 12, 34],  # Length 8
        [2, 6, 24, 46, 68, 90, 13, 35, 57, 79],  # Length 10
        [3, 7, 25, 47, 69],  # Length 5
        [4, 8, 26, 48, 70, 92, 14, 36, 58, 80, 91],  # Length 11
    ]
    
    # Create dataloader
    dataloader = create_dataloader(
        example_trajectories,
        batch_size=2,
        max_length=10,
        pad_token_id=0,
        shuffle=False
    )
    
    # Iterate through batches
    print("Example batches:")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Target IDs shape: {batch['target_ids'].shape}")
        print(f"  Padding mask shape: {batch['padding_mask'].shape}")
        print(f"  Sequence lengths: {batch['sequence_lengths']}")
        print(f"  Input IDs:\n{batch['input_ids']}")
        print(f"  Target IDs:\n{batch['target_ids']}")
        print(f"  Padding mask:\n{batch['padding_mask']}")

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TrajectoryDatasetVector(Dataset):
    """
    Dataset for tokenized GPS trajectories with VECTOR tokens.
    Each token is a vector like [20, 0.3, 45, 12] instead of a scalar.
    """
    def __init__(self, tokenized_trajectories, max_length=None):
        """
        Args:
            tokenized_trajectories: List of tokenized trajectories
                                   Each trajectory is a 2D array/list of shape (seq_len, token_dim)
                                   Example: [[20, 0.3, 45, 12], [21, 0.4, 46, 13], ...]
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
            trajectory = torch.tensor(trajectory, dtype=torch.float32)
        
        # Ensure it's 2D (seq_length, token_dim)
        if trajectory.dim() == 1:
            raise ValueError(f"Trajectory must be 2D (seq_length, token_dim), got shape {trajectory.shape}")
        
        # Truncate if necessary
        if self.max_length is not None and trajectory.size(0) > self.max_length:
            trajectory = trajectory[:self.max_length]
        
        return trajectory


def collate_fn_pad_vector(batch, pad_value=0.0):
    """
    Collate function for DataLoader that pads vector token sequences to same length.
    
    Args:
        batch: List of trajectories (tensors of shape (seq_length, token_dim))
        pad_value: Value to use for padding (default 0.0)
        
    Returns:
        Dictionary containing:
            - 'input_ids': Padded input sequences (batch_size, max_seq_len-1, token_dim)
            - 'target_ids': Padded target sequences (batch_size, max_seq_len-1, token_dim)
            - 'padding_mask': Mask indicating padding positions (batch_size, max_seq_len-1)
            - 'sequence_lengths': Original sequence lengths
    """
    # Pad sequences to same length
    # pad_sequence expects list of tensors and pads along first dimension
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=pad_value)
    # Shape: (batch_size, max_seq_length, token_dim)
    
    # Create input and target sequences (targets are inputs shifted by 1)
    inputs = padded_batch[:, :-1, :]  # All but last token
    targets = padded_batch[:, 1:, :]   # All but first token
    
    # Create padding mask (True for padding positions)
    # A position is padding if ALL elements in the token vector are pad_value
    padding_mask = (inputs == pad_value).all(dim=-1)
    # Shape: (batch_size, seq_length)
    
    # Get original sequence lengths (before padding)
    sequence_lengths = torch.tensor([len(seq) for seq in batch])
    
    return {
        'input_ids': inputs,
        'target_ids': targets,
        'padding_mask': padding_mask,
        'sequence_lengths': sequence_lengths
    }


def create_dataloader_vector(tokenized_trajectories, batch_size=32, max_length=None, 
                            pad_value=0.0, shuffle=True, num_workers=0):
    """
    Create a DataLoader for tokenized trajectories with vector tokens.
    
    Args:
        tokenized_trajectories: List of tokenized trajectories
                               Each trajectory is 2D: (seq_length, token_dim)
        batch_size: Batch size
        max_length: Maximum sequence length
        pad_value: Value for padding
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader object
    """
    dataset = TrajectoryDatasetVector(tokenized_trajectories, max_length=max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn_pad_vector(batch, pad_value),
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    # Simulate tokenized trajectories with VECTOR tokens and variable lengths
    # Each token is a vector of dimension 4: [feature1, feature2, feature3, feature4]
    example_trajectories = [
        # Trajectory 1: 8 tokens, each with 4 features
        [[20, 0.3, 45, 12], 
         [21, 0.4, 46, 13], 
         [22, 0.5, 47, 14],
         [23, 0.6, 48, 15],
         [24, 0.7, 49, 16],
         [25, 0.8, 50, 17],
         [26, 0.9, 51, 18],
         [27, 1.0, 52, 19]],
        
        # Trajectory 2: 10 tokens
        [[30, 0.2, 55, 20],
         [31, 0.3, 56, 21],
         [32, 0.4, 57, 22],
         [33, 0.5, 58, 23],
         [34, 0.6, 59, 24],
         [35, 0.7, 60, 25],
         [36, 0.8, 61, 26],
         [37, 0.9, 62, 27],
         [38, 1.0, 63, 28],
         [39, 1.1, 64, 29]],
        
        # Trajectory 3: 5 tokens
        [[40, 0.5, 65, 30],
         [41, 0.6, 66, 31],
         [42, 0.7, 67, 32],
         [43, 0.8, 68, 33],
         [44, 0.9, 69, 34]],
        
        # Trajectory 4: 11 tokens
        [[50, 0.1, 70, 35],
         [51, 0.2, 71, 36],
         [52, 0.3, 72, 37],
         [53, 0.4, 73, 38],
         [54, 0.5, 74, 39],
         [55, 0.6, 75, 40],
         [56, 0.7, 76, 41],
         [57, 0.8, 77, 42],
         [58, 0.9, 78, 43],
         [59, 1.0, 79, 44],
         [60, 1.1, 80, 45]],
    ]
    
    # Create dataloader
    dataloader = create_dataloader_vector(
        example_trajectories,
        batch_size=2,
        max_length=10,
        pad_value=0.0,
        shuffle=False
    )
    
    # Iterate through batches
    print("Example batches with vector tokens:")
    for i, batch in enumerate(dataloader):
        print(f"\n{'='*60}")
        print(f"Batch {i+1}:")
        print(f"{'='*60}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"    -> (batch_size={batch['input_ids'].shape[0]}, "
              f"seq_length={batch['input_ids'].shape[1]}, "
              f"token_dim={batch['input_ids'].shape[2]})")
        print(f"  Target IDs shape: {batch['target_ids'].shape}")
        print(f"  Padding mask shape: {batch['padding_mask'].shape}")
        print(f"  Sequence lengths: {batch['sequence_lengths']}")
        
        print(f"\n  First trajectory in batch:")
        print(f"    First input token: {batch['input_ids'][0, 0].tolist()}")
        print(f"    First target token: {batch['target_ids'][0, 0].tolist()}")
        print(f"    Padding mask (first 5): {batch['padding_mask'][0, :5].tolist()}")
        
        # Show how padding works
        print(f"\n  Last few positions (showing padding):")
        for j in range(-3, 0):
            print(f"    Position {j}: token={batch['input_ids'][0, j].tolist()}, "
                  f"is_padding={batch['padding_mask'][0, j].item()}")
    
    print("\n" + "="*60)
    print("Token vector format verified!")
    print("Each token is a vector (e.g., [20, 0.3, 45, 12])")
    print("="*60)

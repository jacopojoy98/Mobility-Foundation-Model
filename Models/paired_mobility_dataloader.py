import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class PairedMobilityDataset(Dataset):
    """
    Dataset for paired trajectory and location visit sequences.
    Each sample contains both representations of the same mobility data.
    """
    def __init__(
        self, 
        trajectory_sequences,      # List of trajectory token sequences
        visit_sequences,           # List of visit token sequences
        max_trajectory_length=None,
        max_visit_length=None
    ):
        """
        Args:
            trajectory_sequences: List of trajectory sequences
                                 Each is 2D array (seq_len, trajectory_token_dim)
            visit_sequences: List of visit sequences (same length as trajectory_sequences)
                            Each is 2D array (seq_len, visit_token_dim)
            max_trajectory_length: Max length for trajectory sequences
            max_visit_length: Max length for visit sequences
        """
        assert len(trajectory_sequences) == len(visit_sequences), \
            "Must have same number of trajectory and visit sequences"
        
        self.trajectory_sequences = trajectory_sequences
        self.visit_sequences = visit_sequences
        self.max_trajectory_length = max_trajectory_length
        self.max_visit_length = max_visit_length
    
    def __len__(self):
        return len(self.trajectory_sequences)

    def __getitem__(self, idx):
        # Get paired sequences

        trajectory = self.trajectory_sequences[idx]
        visit = self.visit_sequences[idx]
         # Randomly select a different index for the visit sequence
        other_idx = idx
        while other_idx == idx:
            other_idx = torch.randint(0, len(self.visit_sequences), (1,)).item()
        unpaired_visit = self.visit_sequences[other_idx]

        # Convert to tensors if not already
        if not isinstance(trajectory, torch.Tensor):
            trajectory = torch.tensor(trajectory, dtype=torch.float32)
        if not isinstance(visit, torch.Tensor):
            visit = torch.tensor(visit, dtype=torch.float32)
        if not isinstance(unpaired_visit, torch.Tensor):
            unpaired_visit = torch.tensor(unpaired_visit, dtype=torch.float32)
        
        # Ensure 2D shape
        if trajectory.dim() == 1:
            raise ValueError(f"Trajectory must be 2D, got shape {trajectory.shape}")
        if visit.dim() == 1:
            # If visit is 1D, we can unsqueeze to make it (1, visit_token_dim)
            visit = visit.unsqueeze(0)
        if unpaired_visit.dim() == 1:
            unpaired_visit = unpaired_visit.unsqueeze(0)
        
        # Truncate if necessary
        if self.max_trajectory_length is not None and trajectory.size(0) > self.max_trajectory_length:
            trajectory = trajectory[:self.max_trajectory_length]
        
        if self.max_visit_length is not None and visit.size(0) > self.max_visit_length:
            visit = visit[:self.max_visit_length]
        if self.max_visit_length is not None and unpaired_visit.size(0) > self.max_visit_length:
            unpaired_visit = unpaired_visit[:self.max_visit_length]
        
        return {
            'trajectory': trajectory,
            'visit': visit,
            'unpaired_visit': unpaired_visit  # For negative sampling in CLIP loss
        }
    

def collate_fn_paired(batch, pad_value=0.0):
    """
    Collate function for paired trajectory and visit sequences.
    
    Args:
        batch: List of dicts with 'trajectory','visit' and 'unpaired_visit' keys
        pad_value: Value to use for padding
        
    Returns:
        Dictionary containing:
            - 'trajectory_inputs': (batch_size, max_traj_len-1, traj_token_dim)
            - 'trajectory_targets': (batch_size, max_traj_len-1, traj_token_dim)
            - 'trajectory_padding_mask': (batch_size, max_traj_len-1)
            - 'visit_inputs': (batch_size, max_visit_len, visit_token_dim)
            - 'visit_padding_mask': (batch_size, max_visit_len)
            - 'visit_inputs_unpaired': (batch_size, max_visit_len, visit_token_dim)
            - 'visit_padding_mask_unpaired': (batch_size, max_visit_len)
            - 'trajectory_lengths': Original trajectory lengths
            - 'visit_lengths': Original visit lengths
            - 'visit_lengths_unpaired': Original unpaired visit lengths
    """
    # Separate trajectories and visits
    trajectories = [item['trajectory'] for item in batch]
    visits = [item['visit'] for item in batch]
    visits_unpaired = [item['unpaired_visit'] for item in batch]  # For negative sampling
    
    # Pad trajectory sequences
    padded_trajectories = pad_sequence(trajectories, batch_first=True, padding_value=pad_value)
    # Shape: (batch_size, max_seq_length, trajectory_token_dim)
    
    # Create trajectory input/target pairs (shifted by 1 for next-token prediction)
    trajectory_inputs = padded_trajectories[:, :-1, :]
    trajectory_targets = padded_trajectories[:, 1:, :]
    
    # Create trajectory padding mask
    trajectory_padding_mask = (trajectory_inputs == pad_value).all(dim=-1)
    
    # Pad visit sequences (no shifting needed, just encoding)
    padded_visits = pad_sequence(visits, batch_first=True, padding_value=pad_value)
    # Shape: (batch_size, max_seq_length, visit_token_dim)
    
    # Create visit padding mask
    visit_padding_mask = (padded_visits == pad_value).all(dim=-1)
    
    padded_visits_unpaired = pad_sequence(visits_unpaired, batch_first=True, padding_value=pad_value)
    visit_padding_mask_unpaired = (padded_visits_unpaired == pad_value).all(dim=-1)
    # Get original lengths
    trajectory_lengths = torch.tensor([len(traj) for traj in trajectories])
    visit_lengths = torch.tensor([len(visit) for visit in visits])
    visit_lengths_unpaired = torch.tensor([len(visit) for visit in visits_unpaired])
    return {
        'trajectory_inputs': trajectory_inputs,
        'trajectory_targets': trajectory_targets,
        'trajectory_padding_mask': trajectory_padding_mask,
        'visit_inputs': padded_visits,
        'visit_padding_mask': visit_padding_mask,
        'visit_inputs_unpaired': padded_visits_unpaired,
        'visit_padding_mask_unpaired': visit_padding_mask_unpaired,
        'trajectory_lengths': trajectory_lengths,
        'visit_lengths': visit_lengths,
        'visit_lengths_unpaired': visit_lengths_unpaired
    }


def create_paired_dataloader(
    trajectory_sequences,
    visit_sequences,
    batch_size=32,
    max_trajectory_length=None,
    max_visit_length=None,
    pad_value=0.0,
    shuffle=True,
    num_workers=0
):
    """
    Create a DataLoader for paired trajectory and visit sequences.
    
    Args:
        trajectory_sequences: List of trajectory token sequences
        visit_sequences: List of visit token sequences
        batch_size: Batch size
        max_trajectory_length: Max length for trajectories
        max_visit_length: Max length for visits
        pad_value: Value for padding
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader object
    """
    dataset = PairedMobilityDataset(
        trajectory_sequences,
        visit_sequences,
        max_trajectory_length,
        max_visit_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn_paired(batch, pad_value),
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Simulate paired trajectory and visit sequences
    # These represent THE SAME mobility data in two different formats
    
    num_samples = 100
    trajectory_token_dim = 4  # [lat, lon, time, speed]
    visit_token_dim = 6       # [poi_id, dwell_time, visit_count, time_of_day, day_of_week, duration]
    
    print("Creating paired mobility sequences...")
    print(f"  Trajectory token dim: {trajectory_token_dim}")
    print(f"  Visit token dim: {visit_token_dim}")
    
    trajectory_sequences = []
    visit_sequences = []
    
    for i in range(num_samples):
        # Random sequence length
        num_trajectory_points = np.random.randint(20, 100)
        num_visits = np.random.randint(5, 30)  # Usually fewer visits than trajectory points
        
        # Create trajectory sequence (continuous GPS points)
        trajectory = np.random.randn(num_trajectory_points, trajectory_token_dim)
        
        # Create visit sequence (discrete location visits)
        # Represents the same person's movements but as location visits
        visit = np.random.randn(num_visits, visit_token_dim)
        
        trajectory_sequences.append(trajectory)
        visit_sequences.append(visit)
    
    # Example: Show what paired data looks like
    print(f"\nExample paired sequences:")
    print(f"  Trajectory sequence 0 shape: {trajectory_sequences[0].shape}")
    print(f"  Visit sequence 0 shape: {visit_sequences[0].shape}")
    print(f"\n  First trajectory token: {trajectory_sequences[0][0]}")
    print(f"  First visit token: {visit_sequences[0][0]}")
    
    # Create dataloader
    print("\n" + "="*60)
    print("Creating DataLoader...")
    print("="*60)
    
    dataloader = create_paired_dataloader(
        trajectory_sequences,
        visit_sequences,
        batch_size=8,
        max_trajectory_length=100,
        max_visit_length=50,
        pad_value=0.0,
        shuffle=True,
        num_workers=0
    )
    
    # Iterate through a batch
    print("\nExample batch:")
    for batch in dataloader:
        print(f"\nBatch structure:")
        print(f"  trajectory_inputs shape: {batch['trajectory_inputs'].shape}")
        print(f"  trajectory_targets shape: {batch['trajectory_targets'].shape}")
        print(f"  trajectory_padding_mask shape: {batch['trajectory_padding_mask'].shape}")
        print(f"  visit_inputs shape: {batch['visit_inputs'].shape}")
        print(f"  visit_padding_mask shape: {batch['visit_padding_mask'].shape}")
        
        print(f"\n  Trajectory lengths: {batch['trajectory_lengths'].tolist()}")
        print(f"  Visit lengths: {batch['visit_lengths'].tolist()}")
        
        # Show padding information
        print(f"\n  Sample 0:")
        print(f"    Trajectory padding (first 10): {batch['trajectory_padding_mask'][0, :10].tolist()}")
        print(f"    Visit padding (first 10): {batch['visit_padding_mask'][0, :10].tolist()}")
        
        # Show how the data aligns for CLIP
        print(f"\n  CLIP alignment:")
        print(f"    Each row in the batch represents ONE person's data")
        print(f"    Row 0 trajectory should align with Row 0 visit")
        print(f"    This is enforced by the diagonal in CLIP loss")
        
        break  # Just show first batch
    
    print("\n" + "="*60)
    print("Paired dataloader verified!")
    print("="*60)
    print("\nKey points:")
    print("  1. Each sample has BOTH trajectory and visit representations")
    print("  2. They come from the SAME mobility data")
    print("  3. CLIP will learn to align these representations")
    print("  4. Trajectory sequences are shifted for next-token prediction")
    print("  5. Visit sequences are just encoded (no shifting)")

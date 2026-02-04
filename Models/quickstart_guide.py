"""
Quick Start Guide: Mobility Transformer for GPS Trajectory Prediction
=====================================================================

This guide shows how to use the mobility transformer with your tokenized GPS trajectories.
"""

import torch
from mobility_transformer import MobilityTransformer
from trajectory_dataloader import create_dataloader
from train_mobility_model import TrajectoryTrainer


# ============================================================
# STEP 1: Prepare Your Tokenized Trajectories
# ============================================================
# Your tokenization process should produce a list of sequences
# where each sequence is a list/array of integer token IDs

# Example format (replace with your actual tokenized data):
# your_tokenized_trajectories = [
#     [45, 67, 89, 123, 456, 789],           # Trajectory 1
#     [12, 34, 56, 78, 90, 111, 222],        # Trajectory 2
#     [98, 76, 54, 32, 10],                  # Trajectory 3
#     ...
# ]

# For this example, let's simulate some data:
def load_your_tokenized_data():
    """
    Replace this function with your actual data loading logic.
    
    Returns:
        train_trajectories: List of tokenized training trajectories
        val_trajectories: List of tokenized validation trajectories
        vocab_size: Number of unique tokens in your vocabulary
        max_seq_length: Maximum trajectory length in your dataset
    """
    # REPLACE THIS WITH YOUR ACTUAL DATA
    vocab_size = 5000  # Your actual vocabulary size
    max_seq_length = 200  # Your actual max trajectory length
    
    # Example: Generate dummy data (REPLACE WITH YOUR DATA)
    train_trajectories = [
        torch.randint(1, vocab_size, (torch.randint(20, max_seq_length, (1,)).item(),)).tolist()
        for _ in range(1000)
    ]
    
    val_trajectories = [
        torch.randint(1, vocab_size, (torch.randint(20, max_seq_length, (1,)).item(),)).tolist()
        for _ in range(200)
    ]
    
    return train_trajectories, val_trajectories, vocab_size, max_seq_length


# ============================================================
# STEP 2: Load Your Data
# ============================================================
print("Loading tokenized trajectories...")
train_trajectories, val_trajectories, VOCAB_SIZE, MAX_SEQ_LENGTH = load_your_tokenized_data()

print(f"Dataset statistics:")
print(f"  - Vocabulary size: {VOCAB_SIZE}")
print(f"  - Max sequence length: {MAX_SEQ_LENGTH}")
print(f"  - Training trajectories: {len(train_trajectories)}")
print(f"  - Validation trajectories: {len(val_trajectories)}")


# ============================================================
# STEP 3: Configure Model Hyperparameters
# ============================================================
# Model architecture
D_MODEL = 256              # Embedding dimension (try 128, 256, 512)
NHEAD = 8                  # Attention heads (must divide D_MODEL evenly)
NUM_LAYERS = 6             # Transformer layers (try 4, 6, 8, 12)
DIM_FEEDFORWARD = 1024     # FFN dimension (typically 4 * D_MODEL)
DROPOUT = 0.1              # Dropout rate

# Training parameters
BATCH_SIZE = 32            # Batch size (adjust based on GPU memory)
LEARNING_RATE = 1e-4       # Learning rate
NUM_EPOCHS = 50            # Number of training epochs


# ============================================================
# STEP 4: Create DataLoaders
# ============================================================
print("\nCreating data loaders...")
train_loader = create_dataloader(
    train_trajectories,
    batch_size=BATCH_SIZE,
    max_length=MAX_SEQ_LENGTH,
    pad_token_id=0,         # Token ID used for padding
    shuffle=True,
    num_workers=0           # Set > 0 for multi-process data loading
)

val_loader = create_dataloader(
    val_trajectories,
    batch_size=BATCH_SIZE,
    max_length=MAX_SEQ_LENGTH,
    pad_token_id=0,
    shuffle=False,
    num_workers=0
)


# ============================================================
# STEP 5: Create the Model
# ============================================================
print("\nCreating mobility transformer model...")
model = MobilityTransformer(
    vocab_size=VOCAB_SIZE,           # YOUR vocabulary size
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    max_seq_length=MAX_SEQ_LENGTH,   # YOUR max trajectory length
    learned_pos_encoding=False       # Sinusoidal encoding (as requested)
)

# Print model info
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model created with {num_params:,} trainable parameters")


# ============================================================
# STEP 6: Train the Model
# ============================================================
print("\nInitializing trainer...")
trainer = TrajectoryTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_dir='./runs/mobility_experiment'
)

print(f"Training on: {trainer.device}")
print("\nStarting training...")
trainer.train(
    num_epochs=NUM_EPOCHS,
    save_dir='./checkpoints/mobility_model'
)


# ============================================================
# STEP 7: Use the Trained Model for Prediction
# ============================================================
print("\n" + "="*60)
print("Example: Using the model for prediction")
print("="*60)

# Load the best checkpoint
checkpoint_path = './checkpoints/mobility_model/best_model.pt'
try:
    epoch, loss = trainer.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    
    # Example: Predict next token for a sample trajectory
    sample_trajectory = torch.tensor([train_trajectories[0][:10]]).to(trainer.device)
    
    model.eval()
    with torch.no_grad():
        # Get next token prediction
        next_token = model.predict_next_token(
            sample_trajectory, 
            temperature=0.8,  # Lower = more deterministic, higher = more random
            top_k=50          # Only sample from top 50 tokens
        )
        
        print(f"\nInput trajectory: {sample_trajectory[0].tolist()}")
        print(f"Predicted next token: {next_token.item()}")
        
        # Generate a continuation (example: generate 5 more tokens)
        print("\nGenerating trajectory continuation...")
        current_seq = sample_trajectory.clone()
        generated_tokens = []
        
        for i in range(5):
            next_token = model.predict_next_token(current_seq, temperature=0.8, top_k=50)
            generated_tokens.append(next_token.item())
            
            # Append to sequence for next prediction
            current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
        
        print(f"Generated continuation: {generated_tokens}")
        print(f"Full generated trajectory: {current_seq[0].tolist()}")
        
except FileNotFoundError:
    print(f"Checkpoint not found at {checkpoint_path}")
    print("Train the model first to generate predictions.")


# ============================================================
# OPTIONAL: Save Model for Deployment
# ============================================================
print("\n" + "="*60)
print("Saving model for deployment")
print("="*60)

# Save just the model weights (for deployment)
deployment_path = './deployed_model.pt'
torch.save(model.state_dict(), deployment_path)
print(f"Model weights saved to {deployment_path}")

# To load later:
# loaded_model = MobilityTransformer(vocab_size=VOCAB_SIZE, max_seq_length=MAX_SEQ_LENGTH, ...)
# loaded_model.load_state_dict(torch.load(deployment_path))
# loaded_model.eval()


print("\n" + "="*60)
print("Training complete! Check ./runs/mobility_experiment for TensorBoard logs")
print("Run: tensorboard --logdir=./runs/mobility_experiment")
print("="*60)

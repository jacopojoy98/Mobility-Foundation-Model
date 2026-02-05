"""
Quick Start Guide: Mobility Transformer with VECTOR Tokens
===========================================================

This guide shows how to use the mobility transformer when your tokens are VECTORS
(e.g., [20, 0.3, 45, 12]) instead of scalar IDs (e.g., 100).

VECTOR TOKEN FORMAT:
Instead of:     trajectory = [45, 67, 89, 123]  (scalar tokens)
You have:       trajectory = [[20, 0.3, 45, 12],  (vector tokens)
                              [21, 0.4, 46, 13],
                              [22, 0.5, 47, 14],
                              [23, 0.6, 48, 15]]
"""

import torch
import numpy as np
from Models.mobility_transformer import MobilityTransformerVector
from Models.trajectory_dataloader import create_dataloader_vector
from Models.train_mobility_model import TrajectoryTrainerVector
import pickle

# ============================================================
# STEP 1: Prepare Your Vector Token Trajectories
# ============================================================

def load_your_vector_tokenized_data():
    """
    Replace this function with your actual data loading logic.
    
    Your tokenization should produce trajectories where each token is a VECTOR.
    
    Returns:
        train_trajectories: List of trajectories, each is 2D array (seq_len, token_dim)
        val_trajectories: List of validation trajectories
        token_dim: Dimension of each token vector
        max_seq_length: Maximum trajectory length in your dataset
    """
    
    # EXAMPLE: Your actual data might look like this:
    # Each trajectory is a 2D array/list where:
    #   - First dimension: sequence of locations
    #   - Second dimension: features per location
    
    # Example with 4-dimensional token vectors:
    # Feature 0: grid_x coordinate
    # Feature 1: grid_y coordinate  
    # Feature 2: time_of_day (normalized)
    # Feature 3: day_of_week (normalized)
    with open("preliminary_tokens.pkl", "rb") as f:
        complete_dataset = pickle.load(f)
    
    max_seq_length = max[[len(Trajectory) for Trajectory in complete_dataset]]
    token_dim = len(complete_dataset[0][0])
    # REPLACE THIS with loading your actual data
    # Example format:
    train_trajectories = complete_dataset[:int(len(complete_dataset)*0.8)]
    val_trajectories = complete_dataset[int(len(complete_dataset)*0.8):]
    # Generate more dummy data for demonstrat
    
    return train_trajectories, val_trajectories, token_dim, max_seq_length


# ============================================================
# STEP 2: Load Your Data
# ============================================================
print("Loading vector tokenized trajectories...")
train_trajectories, val_trajectories, TOKEN_DIM, MAX_SEQ_LENGTH = load_your_vector_tokenized_data()

print(f"Dataset statistics:")
print(f"  - Token dimension: {TOKEN_DIM}")
print(f"  - Max sequence length: {MAX_SEQ_LENGTH}")
print(f"  - Training trajectories: {len(train_trajectories)}")
print(f"  - Validation trajectories: {len(val_trajectories)}")

# Show example of data format
print(f"\nExample trajectory (first 3 tokens):")
for i, token in enumerate(train_trajectories[0][:3]):
    print(f"  Token {i}: {token}")


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
LOSS_FN = 'mse'            # Loss function: 'mse', 'mae', or 'huber'


# ============================================================
# STEP 4: Create DataLoaders
# ============================================================
print("\nCreating data loaders for vector tokens...")
train_loader = create_dataloader_vector(
    train_trajectories,
    batch_size=BATCH_SIZE,
    max_length=MAX_SEQ_LENGTH,
    pad_value=0.0,          # Value used for padding
    shuffle=True,
    num_workers=0           # Set > 0 for multi-process data loading
)

val_loader = create_dataloader_vector(
    val_trajectories,
    batch_size=BATCH_SIZE,
    max_length=MAX_SEQ_LENGTH,
    pad_value=0.0,
    shuffle=False,
    num_workers=0
)


# ============================================================
# STEP 5: Create the Model
# ============================================================
print("\nCreating mobility transformer model for vector tokens...")
model = MobilityTransformerVector(
    token_dim=TOKEN_DIM,             # YOUR token vector dimension
    output_dim=TOKEN_DIM,            # Output same dimension
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    max_seq_length=MAX_SEQ_LENGTH,   # YOUR max trajectory length
    learned_pos_encoding=False       # Sinusoidal encoding
)

# Print model info
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model created with {num_params:,} trainable parameters")


# ============================================================
# STEP 6: Train the Model
# ============================================================
print("\nInitializing trainer...")
trainer = TrajectoryTrainerVector(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    loss_fn=LOSS_FN,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_dir='./runs/mobility_vector_experiment'
)

print(f"Training on: {trainer.device}")
print("\nStarting training...")
trainer.train(
    num_epochs=NUM_EPOCHS,
    save_dir='./checkpoints/mobility_vector_model'
)


# ============================================================
# STEP 7: Use the Trained Model for Prediction
# ============================================================
print("\n" + "="*60)
print("Example: Using the model for prediction")
print("="*60)

# Load the best checkpoint
checkpoint_path = './checkpoints/mobility_vector_model/best_model.pt'
try:
    epoch, loss = trainer.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    
    # Example: Predict next token for a sample trajectory
    sample_trajectory = torch.tensor([train_trajectories[0][:10]], dtype=torch.float32).to(trainer.device)
    
    model.eval()
    with torch.no_grad():
        # Get next token prediction
        next_token = model.predict_next_token(sample_trajectory)
        
        print(f"\nInput trajectory (first 3 and last token):")
        print(f"  Token 0: {sample_trajectory[0, 0].tolist()}")
        print(f"  Token 1: {sample_trajectory[0, 1].tolist()}")
        print(f"  Token 2: {sample_trajectory[0, 2].tolist()}")
        print(f"  ...")
        print(f"  Token -1: {sample_trajectory[0, -1].tolist()}")
        print(f"\nPredicted next token: {next_token[0].tolist()}")
        
        # Generate a trajectory continuation (5 more tokens)
        print("\n" + "="*60)
        print("Generating trajectory continuation (5 tokens)...")
        print("="*60)
        
        generated = model.generate_trajectory(
            sample_trajectory, 
            num_steps=5
        )
        
        print(f"Original length: {sample_trajectory.shape[1]}")
        print(f"Generated length: {generated.shape[1]}")
        print(f"\nGenerated tokens:")
        for i in range(sample_trajectory.shape[1], generated.shape[1]):
            print(f"  Token {i}: {generated[0, i].tolist()}")
        
except FileNotFoundError:
    print(f"Checkpoint not found at {checkpoint_path}")
    print("Train the model first to generate predictions.")


# ============================================================
# STEP 8: Visualize Predictions (Optional)
# ============================================================
print("\n" + "="*60)
print("Example: Comparing prediction vs ground truth")
print("="*60)

# Get a batch from validation set
val_batch = next(iter(val_loader))
inputs = val_batch['input_ids'][:1].to(trainer.device)  # Take first sample
targets = val_batch['target_ids'][:1].to(trainer.device)

model.eval()
with torch.no_grad():
    predictions = model(inputs)
    
    # Compare first position prediction with ground truth
    print(f"\nPosition 0:")
    print(f"  Input:      {inputs[0, 0].tolist()}")
    print(f"  Predicted:  {predictions[0, 0].tolist()}")
    print(f"  Ground truth: {targets[0, 0].tolist()}")
    
    # Compute error
    error = torch.abs(predictions[0, 0] - targets[0, 0])
    print(f"  Absolute error: {error.tolist()}")
    print(f"  Mean absolute error: {error.mean().item():.4f}")


# ============================================================
# OPTIONAL: Save Model for Deployment
# ============================================================
print("\n" + "="*60)
print("Saving model for deployment")
print("="*60)

# Save just the model weights (for deployment)
deployment_path = './deployed_vector_model.pt'
torch.save(model.state_dict(), deployment_path)
print(f"Model weights saved to {deployment_path}")

# To load later:
# loaded_model = MobilityTransformerVector(
#     token_dim=TOKEN_DIM, 
#     output_dim=TOKEN_DIM,
#     max_seq_length=MAX_SEQ_LENGTH,
#     ...
# )
# loaded_model.load_state_dict(torch.load(deployment_path))
# loaded_model.eval()


print("\n" + "="*60)
print("KEY DIFFERENCES FROM SCALAR TOKENS:")
print("="*60)
print("1. Tokens are vectors: [[20, 0.3, 45, 12], ...]")
print("2. Model uses Linear projection instead of Embedding lookup")
print("3. Loss is MSE/MAE instead of Cross-Entropy")
print("4. Output is continuous vector instead of discrete class")
print("5. No need for vocabulary size parameter")
print("="*60)

print("\n" + "="*60)
print("Training complete! Check ./runs/mobility_vector_experiment for TensorBoard logs")
print("Run: tensorboard --logdir=./runs/mobility_vector_experiment")
print("="*60)

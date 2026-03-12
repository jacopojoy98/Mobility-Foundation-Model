"""
Quick Start Guide: CLIP-Style Dual-View Mobility Model
=======================================================

This guide shows how to train a model that learns aligned representations
from TWO different views of mobility data:
1. Trajectory sequences (GPS points over time)
2. Location visit sequences (discrete location visits)

The model learns to align these representations using CLIP-style contrastive learning.
"""

import torch
import numpy as np
from clip_mobility_model import CLIPMobilityModel
from paired_mobility_dataloader import create_paired_dataloader
from train_clip_mobility import CLIPMobilityTrainer
from Data_visualization import visualize_distribution
import pickle

# ============================================================
# STEP 1: Prepare Your Paired Data
# ============================================================
print("\n" + "="*70)
print("STEP 1: Data Preparation")
print("="*70)

# Load data
print("\nLoading paired mobility data...")
with open('Data/Mobility/trajectory_tokens_split_0.pkl', 'rb') as f:
    train_trajectories = pickle.load(f) 
with open('Data/Mobility/visit_tokens_split_0.pkl', 'rb') as f:
    train_visits = pickle.load(f)
with open('Data/Mobility/trajectory_tokens_split_1.pkl', 'rb') as f:
    val_trajectories = pickle.load(f) 
with open('Data/Mobility/visit_tokens_split_1.pkl', 'rb') as f:
    val_visits = pickle.load(f)

trajectories = []
token_dim = 0
for user_trajectories in train_trajectories:
    for traj in user_trajectories:
        trajectories.append(traj)
        if token_dim == 0:
            token_dim = len(traj[0])
        else:
            for token in traj:
                assert len(token) == token_dim, f"Token dimension mismatch: expected {token_dim}, got {len(token)}"

# print(f"Length of loaded dataset (number of users): {len(train_trajectories)}")
# print(f"Second dimension (number of trajectories per user) is inhomogeneous\
#     \n statistics: \n \t mean: {np.mean([len(user_traj) for user_traj in train_trajectories]):.2f}\n\
#     \t min: {np.min([len(user_traj) for user_traj in train_trajectories])}\n\
#     \t max: {np.max([len(user_traj) for user_traj in train_trajectories])}")
# print(f"Lenght of trajectory (inhomogeneous): \n statistics: \n \t mean: {np.mean([len(traj) for traj in trajectories]):.2f}\n\
#     \t min: {np.min([len(traj) for traj in trajectories])}\n\
#     \t max: {np.max([len(traj) for traj in trajectories])}")
# print(f"Dimension of tokens: { len(train_trajectories[0][0][0])}")
# input("Check trajectory data shapes above. Press Enter to continue...")

# print(f"Length of loaded dataset(number_of_users): {len(train_visits)}")
# print(f"Second dimension (number of trajectories per user) is inhomogeneous\
#     \n statistics: \n \t mean: {np.mean([len(user_visit) for user_visit in train_visits]):.2f}\n\
#     \t min: {np.min([len(user_visit) for user_visit in train_visits])}\n\
#     \t max: {np.max([len(user_visit) for user_visit in train_visits])}")
# print(f"Dimension of tokens: { len(train_visits[0][0])}")
# input("Check visit data shapes above. Press Enter to continue...")

TRAJECTORY_TOKEN_DIM = 8
VISIT_TOKEN_DIM = 9

# ============================================================
# STEP 2: Configure Model
# ============================================================
print("\n" + "="*70)
print("STEP 2: CONFIGURE MODEL")
print("="*70)

# Model architecture (shared between both transformers)
D_MODEL = 256
NHEAD =  8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# CLIP-specific parameters
EMBEDDING_DIM = 128        # Dimension of aligned embedding space
TEMPERATURE = 0.07         # CLIP temperature (controls sharpness)

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200

# Loss weights
PREDICTION_WEIGHT = 1.0    # Weight for trajectory prediction
CLIP_WEIGHT = 1.0          # Weight for CLIP alignment

# Max sequence lengths
TRAJECTORY_MAX_LEN = 200
VISIT_MAX_LEN = 100

print(f"Model configuration:")
print(f"  Shared hidden dim (d_model): {D_MODEL}")
print(f"  CLIP embedding dim: {EMBEDDING_DIM}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Loss weights: Prediction={PREDICTION_WEIGHT}, CLIP={CLIP_WEIGHT}")


# ============================================================
# STEP 3: Create DataLoaders
# ============================================================
print("\n" + "="*70)
print("STEP 3: CREATE DATALOADERS")
print("="*70)

train_trajectories = [np.array(traj).astype(np.float32) for user_trajectories in train_trajectories for traj in user_trajectories]
train_visits = [np.array(visit).astype(np.float32) for user_visits in train_visits for visit in user_visits]

val_trajectories = [np.array(traj).astype(np.float32) for user_trajectories in val_trajectories for traj in user_trajectories]
val_visits = [np.array(visit).astype(np.float32) for user_visits in val_visits for visit in user_visits]

print(len(train_trajectories), len(train_visits))  # Should be the same number of trajectories and visits   
print(len(train_trajectories[0]), len(train_visits[0]))  # Length of first trajectory and visit sequence
# print(np.array(train_visits).shape)
# input()
# print(len(train_trajectories), len(train_visits))  # Should be the same number of trajectories and visits
# print(len(train_trajectories[0]), len(train_visits[0]))  # Length of first trajectory and visit sequence
# input("Check data shapes above. Press Enter to continue...")
# print(train_visits)    

train_loader = create_paired_dataloader(
    train_trajectories,
    train_visits,
    batch_size=BATCH_SIZE,
    max_trajectory_length=TRAJECTORY_MAX_LEN,
    max_visit_length=VISIT_MAX_LEN,
    pad_value=0.0,
    shuffle=True
)

val_loader = create_paired_dataloader(
    val_trajectories,
    val_visits,
    batch_size=BATCH_SIZE,
    max_trajectory_length=TRAJECTORY_MAX_LEN,
    max_visit_length=VISIT_MAX_LEN,
    pad_value=0.0,
    shuffle=False
)

print(f"DataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")


# ============================================================
# STEP 4: Create CLIP Model
# ============================================================
print("\n" + "="*70)
print("STEP 4: CREATE CLIP MODEL")
print("="*70)

model = CLIPMobilityModel(
    # Trajectory transformer parameters
    trajectory_token_dim=TRAJECTORY_TOKEN_DIM,
    trajectory_max_seq_length=TRAJECTORY_MAX_LEN,
    
    # Visit transformer parameters
    visit_token_dim=VISIT_TOKEN_DIM,
    visit_max_seq_length=VISIT_MAX_LEN,
    
    # Shared architecture parameters
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    learned_pos_encoding=False,  # Sinusoidal encoding
    
    # CLIP parameters
    embedding_dim=EMBEDDING_DIM,
    temperature=TEMPERATURE
)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"CLIP Mobility Model created with {num_params:,} parameters")

print(f"\nModel components:")
print(f"  1. Trajectory Transformer: processes GPS sequences")
print(f"  2. Visit Transformer: processes location visit sequences")
print(f"  3. CLIP projections: align the two representations")


# ============================================================
# STEP 5: Train the Model
# ============================================================
print("\n" + "="*70)
print("STEP 5: TRAIN MODEL")
print("="*70)

trainer = CLIPMobilityTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    prediction_loss_fn='mse',
    clip_weight=CLIP_WEIGHT,
    prediction_weight=PREDICTION_WEIGHT,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_dir='./runs/clip_mobility_guide'
)

print(f"Trainer configured:")
print(f"  Device: {trainer.device}")
print(f"  Prediction loss: MSE")
print(f"  CLIP weight: {CLIP_WEIGHT}")
print(f"  Prediction weight: {PREDICTION_WEIGHT}")

print("\nStarting training...")
trainer.train(
    num_epochs=NUM_EPOCHS,
    save_dir='./checkpoints/clip_mobility_guide'
)


# ============================================================
# STEP 6: Use the Trained Model
# ============================================================
print("\n" + "="*70)
print("STEP 6: USING THE TRAINED MODEL")
print("="*70)

# Load best checkpoint
checkpoint_path = './checkpoints/clip_mobility_guide/best_model.pt'
try:
    epoch, loss = trainer.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    
    model.eval()
    
    # Example 1: Get aligned embeddings
    print("\n" + "-"*70)
    print("Example 1: Encoding sequences into aligned embedding space")
    print("-"*70)
    
    sample_trajectory = torch.tensor([train_trajectories[0][:200]], dtype=torch.float32).to(trainer.device)
    print(sample_trajectory.shape)  # Should be (1, seq_len, token_dim)
    sample_visit = torch.tensor([train_visits[:200]], dtype=torch.float32).to(trainer.device)
    print(sample_visit.shape)
    input()
    # sample_visit = sample_visit.unsqueeze(0)
    with torch.no_grad():
        traj_embedding = model.encode_trajectory(sample_trajectory)
        visit_embedding = model.encode_visits(sample_visit)
        
        # Compute similarity
        similarity = (traj_embedding @ visit_embedding.t()).item()
        
        print(f"Trajectory embedding shape: {traj_embedding.shape}")
        print(f"Visit embedding shape: {visit_embedding.shape}")
        print(f"Cosine similarity (should be high): {similarity:.4f}")
    
    # Example 2: Cross-modal retrieval
    print("\n" + "-"*70)
    print("Example 2: Cross-modal retrieval")
    print("-"*70)
    
    # Get a batch of validation data
    val_batch = next(iter(val_loader))
    
    with torch.no_grad():
        traj_inputs = val_batch['trajectory_inputs'].to(trainer.device)
        visit_inputs = val_batch['visit_inputs'].to(trainer.device)
        traj_mask = val_batch['trajectory_padding_mask'].to(trainer.device)
        visit_mask = val_batch['visit_padding_mask'].to(trainer.device)
        
        # Encode all trajectories and visits
        traj_embeddings = model.encode_trajectory(traj_inputs, traj_mask)
        visit_embeddings = model.encode_visits(visit_inputs, visit_mask)
        visualize_distribution(traj_embeddings.cpu().numpy(),outdir="figures/traj_embedding_distribution")
        visualize_distribution(visit_embeddings.cpu().numpy(),outdir="figures/visit_embedding_distribution")

        # Compute similarity matrix
        similarity_matrix = traj_embeddings @ visit_embeddings.t()
        
        # For each trajectory, find most similar visit
        most_similar = similarity_matrix.argmax(dim=1)
        
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"\nRetrieval accuracy (diagonal matches):")
        correct = (most_similar == torch.arange(len(most_similar), device=trainer.device)).sum()
        accuracy = correct / len(most_similar)
        print(f"  {correct}/{len(most_similar)} = {accuracy:.2%}")
        
        print(f"\nTop-5 retrieval accuracy:")
        top5 = similarity_matrix.topk(5, dim=1).indices
        correct_indices = torch.arange(len(most_similar), device=trainer.device).unsqueeze(1)
        top5_correct = (top5 == correct_indices).any(dim=1).sum()
        top5_accuracy = top5_correct / len(most_similar)
        print(f"  {top5_correct}/{len(most_similar)} = {top5_accuracy:.2%}")
    
    # Example 3: Trajectory prediction
    print("\n" + "-"*70)
    print("Example 3: Next-token trajectory prediction")
    print("-"*70)
    
    with torch.no_grad():
        predictions = model.trajectory_transformer(traj_inputs[:1], traj_mask[:1])
        
        print(f"Input shape: {traj_inputs[:1].shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"\nFirst token:")
        print(f"  Input: {traj_inputs[0, 0].cpu().tolist()}")
        print(f"  Predicted next: {predictions[0, 0].cpu().tolist()}")
        print(f"  Actual next: {val_batch['trajectory_targets'][0, 0].tolist()}")
    
    # Example 4: Generate trajectory continuation
    print("\n" + "-"*70)
    print("Example 4: Generate trajectory continuation")
    print("-"*70)
    
    with torch.no_grad():
        initial_seq = traj_inputs[0:1, :10, :]  # First 10 tokens
        generated = model.trajectory_transformer.generate_trajectory(
            initial_seq,
            num_steps=5
        )
        
        print(f"Initial sequence length: {initial_seq.shape[1]}")
        print(f"Generated sequence length: {generated.shape[1]}")
        print(f"\nGenerated tokens:")
        for i in range(10, 15):
            print(f"  Token {i}: {generated[0, i].cpu().tolist()}")

except FileNotFoundError:
    print(f"Checkpoint not found at {checkpoint_path}")
    print("Train the model first!")


# ============================================================
# UNDERSTANDING CLIP ALIGNMENT
# ============================================================
print("\n" + "="*70)
print("HOW CLIP ALIGNMENT WORKS")
print("="*70)

print("""
1. ENCODING:
   - Trajectory → Trajectory Transformer → Embedding (128-dim)
   - Visit → Visit Transformer → Embedding (128-dim)

2. SIMILARITY MATRIX:
   Batch of 4 samples:
   
              Visit0  Visit1  Visit2  Visit3
   Traj0  [   0.95    0.12    0.08    0.15  ]  ← High on diagonal
   Traj1  [   0.10    0.92    0.14    0.09  ]  ← (matched pairs)
   Traj2  [   0.11    0.13    0.89    0.12  ]
   Traj3  [   0.14    0.11    0.10    0.91  ]

3. CLIP LOSS:
   - Encourages diagonal to be high (matched pairs)
   - Encourages off-diagonal to be low (mismatched pairs)
   - Symmetric: trajectory→visit AND visit→trajectory

4. RESULT:
   - Model learns that different views of SAME data are similar
   - Can retrieve one view from another
   - Embeddings capture shared mobility patterns
""")


print("\n" + "="*70)
print("PRACTICAL APPLICATIONS")
print("="*70)

print("""
1. CROSS-MODAL RETRIEVAL:
   - Given GPS trajectory, find similar visit patterns
   - Given visit pattern, find similar GPS trajectories

2. TRANSFER LEARNING:
   - Pre-train with CLIP on large dataset
   - Fine-tune on specific tasks with limited data

3. MULTI-VIEW LEARNING:
   - Learn robust representations from multiple data sources
   - Handle missing modalities at inference time

4. SIMILARITY SEARCH:
   - Find similar users based on either view
   - Cluster mobility patterns in aligned space

5. DATA AUGMENTATION:
   - Use one view when other is unavailable
   - Improve robustness by training on both views
""")


print("\n" + "="*70)
print("Training complete! Check TensorBoard:")
print("  tensorboard --logdir=./runs/clip_mobility_guide")
print("="*70)

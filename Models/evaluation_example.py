"""
Complete Evaluation Example
============================

This script demonstrates how to:
1. Load a trained foundation model
2. Extract evaluation labels from trajectory data
3. Run comprehensive evaluation on downstream tasks
"""

import pickle

import pickle

import torch
import numpy as np
from clip_mobility_model import CLIPMobilityModel
from label_extraction import LabelExtractor
from evaluation_framework import MobilityEvaluationSuite
from train_clip_mobility import CLIPMobilityTrainer


def prepare_trajectories_for_model(trajectories, max_length=None):
    """
    Convert list of numpy arrays to padded tensor.
    
    Args:
        trajectories: List of numpy arrays (seq_len, token_dim)
        max_length: Maximum sequence length (truncate if longer)
    
    Returns:
        padded_tensor: (num_samples, max_seq_len, token_dim)
        padding_mask: (num_samples, max_seq_len)
    """
    # Truncate if needed
    if max_length is not None:
        trajectories = [traj[:max_length] if len(traj) > max_length else traj 
                       for traj in trajectories]
    
    # Find max length
    max_len = max(len(traj) for traj in trajectories)
    token_dim = trajectories[0].shape[1]
    
    # Create padded tensor
    padded = np.zeros((len(trajectories), max_len, token_dim))
    padding_mask = np.ones((len(trajectories), max_len), dtype=bool)
    
    for i, traj in enumerate(trajectories):
        length = len(traj)
        padded[i, :length] = traj
        padding_mask[i, :length] = False
    
    return torch.FloatTensor(padded), torch.BoolTensor(padding_mask)


def main():
    print("="*70)
    print("COMPLETE EVALUATION PIPELINE")
    print("="*70)
    
    # ============================================================
    # STEP 1: Load Trained Foundation Model
    # ============================================================
    print("\n" + "="*70)
    print("STEP 1: LOAD TRAINED FOUNDATION MODEL")
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

    TRAJECTORY_TOKEN_DIM = 8
    VISIT_TOKEN_DIM = 9
       # Weight for CLIP alignment

    # Max sequence lengths
    TRAJECTORY_MAX_LEN = 200
    VISIT_MAX_LEN = 100
    
    # Create model
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load checkpoint
    checkpoint_path = './checkpoints/clip_mobility/best_model.pt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    except FileNotFoundError:
        print(f"⚠ Checkpoint not found at {checkpoint_path}")
        print("  Using randomly initialized model for demonstration")
    
    model.eval()
    model.to(device)
    
    # ============================================================
    # STEP 2: Generate/Load Trajectory Data
    # ============================================================
    print("\n" + "="*70)
    print("STEP 2: LOAD TRAJECTORY DATA")
    print("="*70)
    
    # In practice, load your real trajectory data here
    # For demonstration, we generate synthetic data
    
    with open('Data/Mobility/trajectory_tokens_split_0.pkl', 'rb') as f:
        train_trajectories = pickle.load(f) 
    with open('Data/Mobility/visit_tokens_split_0.pkl', 'rb') as f:
        train_visits = pickle.load(f)
    with open('Data/Mobility/trajectory_tokens_split_1.pkl', 'rb') as f:
        val_trajectories = pickle.load(f) 
    with open('Data/Mobility/visit_tokens_split_1.pkl', 'rb') as f:
        val_visits = pickle.load(f)
    

    trajectories = [np.array(traj).astype(np.float32) for user_trajectories in train_trajectories for traj in user_trajectories]
    # ============================================================
    # STEP 3: Extract Evaluation Labels
    # ============================================================
    print("\n" + "="*70)
    print("STEP 3: EXTRACT EVALUATION LABELS")
    print("="*70)
    with open('Data/Labels/evaluation_labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    # ============================================================
    # STEP 4: Split Data
    # ============================================================
    print("\n" + "="*70)
    print("STEP 4: SPLIT DATA")
    print("="*70)
    
    train_data, val_data = LabelExtractor.split_data(
        trajectories,
        labels,
        train_ratio=0.8,
        random_seed=42
    )
    
    # ============================================================
    # STEP 5: Prepare Data for Model
    # ============================================================
    print("\n" + "="*70)
    print("STEP 5: PREPARE DATA FOR MODEL")
    print("="*70)
    
    print("Converting to tensors...")
    
    # Prepare training data
    train_traj_tensor, train_traj_mask = prepare_trajectories_for_model(
        train_data['trajectories'], 
        max_length=200
    )
    
    # Prepare validation data
    val_traj_tensor, val_traj_mask = prepare_trajectories_for_model(
        val_data['trajectories'],
        max_length=200
    )
    
    
    print(f"✓ Train trajectories shape: {train_traj_tensor.shape}")
    print(f"✓ Val trajectories shape: {val_traj_tensor.shape}")
    
    # ============================================================
    # STEP 6: Create Evaluation Suite
    # ============================================================
    print("\n" + "="*70)
    print("STEP 6: CREATE EVALUATION SUITE")
    print("="*70)
    
    evaluator = MobilityEvaluationSuite(model, device=device)
    print("✓ Evaluation suite created")
    
    # ============================================================
    # STEP 7: Prepare Evaluation Data Dictionary
    # ============================================================
    print("\n" + "="*70)
    print("STEP 7: PREPARE EVALUATION DATA")
    print("="*70)
    
    evaluation_data = {}


    # Task 1: Destination Prediction
    evaluation_data['destination'] = {
        'train_trajectories': train_traj_tensor,
        'train_destinations': torch.LongTensor(train_data['labels']['destination']['destination_labels']),
        'val_trajectories': val_traj_tensor,
        'val_destinations': torch.LongTensor(val_data['labels']['destination']['destination_labels']),
        'num_destinations': train_data['labels']['destination']['num_destinations'],
        'sequence_type': 'trajectory',
        'probe_type': 'mlp'
    }

    # Task 2: Time of Arrival
    evaluation_data['time_of_arrival'] = {
        'train_trajectories': train_traj_tensor,
        'train_arrival_times': train_data['labels']['time_of_arrival']['arrival_times'],
        'val_trajectories': val_traj_tensor,
        'val_arrival_times': val_data['labels']['time_of_arrival']['arrival_times'],
        'sequence_type': 'trajectory',
        'probe_type': 'mlp'
    }
    
    # Task 4: Trip Purpose Classification

    evaluation_data['trip_purpose'] = {
        'train_trajectories': train_traj_tensor,
        'train_purposes': torch.LongTensor(train_data['labels']['trip_purpose']['purpose_labels']),
        'val_trajectories': val_traj_tensor,
        'val_purposes': torch.LongTensor(val_data['labels']['trip_purpose']['purpose_labels']),
        'num_purposes': train_data['labels']['trip_purpose']['num_purposes'],
        'sequence_type': 'trajectory',
        'probe_type': 'mlp'
    }
    
    # # Task 5: User Identification
    # if 'user_ids' in train_data['labels']:
    #     evaluation_data['user_identification'] = {
    #         'train_trajectories': train_traj_tensor,
    #         'train_user_ids': torch.LongTensor(train_data['labels']['user_ids']),
    #         'val_trajectories': val_traj_tensor,
    #         'val_user_ids': torch.LongTensor(val_data['labels']['user_ids']),
    #         'num_users': train_data['labels']['num_users'],
    #         'sequence_type': 'trajectory',
    #         'probe_type': 'mlp'
    #     }
    
    print("✓ Evaluation data prepared")
    print(f"  Tasks configured: {len(evaluation_data)}")
    
    # ============================================================
    # STEP 8: Run Full Evaluation
    # ============================================================
    print("\n" + "="*70)
    print("STEP 8: RUN FULL EVALUATION")
    print("="*70)
    
    print("\nRunning all evaluation tasks...")
    print("This will train linear probes on frozen embeddings for each task.")
    print()
    
    all_results = evaluator.run_full_evaluation(
        evaluation_data,
        save_path='./evaluation_results.json'
    )
    
    # ============================================================
    # STEP 9: Print Summary
    # ============================================================
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    if 'destination_prediction' in all_results:
        print(f"\n1. DESTINATION PREDICTION")
        print(f"   Accuracy: {all_results['destination_prediction']['accuracy']:.4f}")
        print(f"   F1 Score: {all_results['destination_prediction']['f1_score']:.4f}")
    
    if 'time_of_arrival' in all_results:
        print(f"\n2. TIME OF ARRIVAL")
        print(f"   MAE: {all_results['time_of_arrival']['mae']:.4f}")
        print(f"   RMSE: {all_results['time_of_arrival']['rmse']:.4f}")
    
    if 'next_location' in all_results:
        print(f"\n3. NEXT LOCATION PREDICTION")
        print(f"   Top-1 Accuracy: {all_results['next_location']['accuracy']:.4f}")
        print(f"   Top-5 Accuracy: {all_results['next_location']['top5_accuracy']:.4f}")
    
    if 'trip_purpose' in all_results:
        print(f"\n4. TRIP PURPOSE CLASSIFICATION")
        print(f"   Accuracy: {all_results['trip_purpose']['accuracy']:.4f}")
        print(f"   F1 Score: {all_results['trip_purpose']['f1_score']:.4f}")
    
    if 'user_identification' in all_results:
        print(f"\n5. USER IDENTIFICATION")
        print(f"   Accuracy: {all_results['user_identification']['accuracy']:.4f}")
        print(f"   F1 Score: {all_results['user_identification']['f1_score']:.4f}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nDetailed results saved to: ./evaluation_results.json")
    print("\nThese metrics evaluate the quality of learned embeddings.")
    print("Higher scores indicate better representation learning.")
    
    # ============================================================
    # OPTIONAL: Individual Task Evaluation
    # ============================================================
    print("\n" + "="*70)
    print("OPTIONAL: RUN INDIVIDUAL TASKS")
    print("="*70)
    print("\nYou can also run individual tasks:")
    print()
    print("# Example: Only destination prediction")
    print("results = evaluator.evaluate_destination_prediction(")
    print("    train_trajectories=train_traj_tensor,")
    print("    train_destinations=torch.LongTensor(train_data['labels']['destinations']),")
    print("    val_trajectories=val_traj_tensor,")
    print("    val_destinations=torch.LongTensor(val_data['labels']['destinations']),")
    print("    num_destinations=train_data['labels']['num_destinations'],")
    print("    sequence_type='trajectory',")
    print("    probe_type='mlp'")
    print(")")


if __name__ == "__main__":
    main()

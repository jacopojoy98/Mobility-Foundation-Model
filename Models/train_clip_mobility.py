import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import random
from clip_mobility_model import CLIPMobilityModel, compute_combined_loss
from paired_mobility_dataloader import create_paired_dataloader


class CLIPMobilityTrainer:
    """
    Trainer for CLIP-style dual-view mobility model.
    Jointly trains trajectory prediction and cross-modal alignment.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=1e-4,
        weight_decay=0.01,
        prediction_loss_fn='mse',
        clip_weight=1.0,
        prediction_weight=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='./runs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.prediction_loss_fn = prediction_loss_fn
        self.clip_weight = clip_weight
        self.prediction_weight = prediction_weight
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        self.global_step = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_pred_loss = 0
        total_clip_loss = 0
        num_batches = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            trajectory_inputs = batch['trajectory_inputs'].to(self.device)
            trajectory_targets = batch['trajectory_targets'].to(self.device)
            trajectory_padding_mask = batch['trajectory_padding_mask'].to(self.device)
            
            visit_inputs = batch['visit_inputs'].to(self.device)
            visit_padding_mask = batch['visit_padding_mask'].to(self.device)

            visit_inputs_unpaired = batch['visit_inputs_unpaired'].to(self.device)
            visit_padding_mask_unpaired = batch['visit_padding_mask_unpaired'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            if random.random() < 0.5:
                # 50% of the time, use unpaired visits for negative sampling
                visit_inputs_to_use = visit_inputs_unpaired
                visit_padding_mask_to_use = visit_padding_mask_unpaired
                clip_weight = -self.clip_weight
            else:
                visit_inputs_to_use = visit_inputs
                visit_padding_mask_to_use = visit_padding_mask
                clip_weight=self.clip_weight

            outputs = self.model(
                trajectory_tokens=trajectory_inputs,
                visit_tokens=visit_inputs_to_use,
                trajectory_padding_mask=trajectory_padding_mask,
                visit_padding_mask=visit_padding_mask_to_use,
                trajectory_targets=trajectory_targets,
                compute_alignment=True
            )

            total_loss_batch, loss_dict = compute_combined_loss(
                outputs,
                trajectory_targets,
                trajectory_padding_mask,
                self.prediction_loss_fn,
                clip_weight,
                self.prediction_weight
            )
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss_dict['total_loss']
            total_pred_loss += loss_dict['prediction_loss']
            total_clip_loss += loss_dict['clip_loss']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'total': f"{loss_dict['total_loss']:.4f}",
                'pred': f"{loss_dict['prediction_loss']:.4f}",
                'clip': f"{loss_dict['clip_loss']:.4f}"
            })
            
            # Log to TensorBoard
            self.writer.add_scalar('Train/TotalLoss', loss_dict['total_loss'], self.global_step)
            self.writer.add_scalar('Train/PredictionLoss', loss_dict['prediction_loss'], self.global_step)
            self.writer.add_scalar('Train/CLIPLoss', loss_dict['clip_loss'], self.global_step)
            self.writer.add_scalar('Train/Temperature', self.model.logit_scale.exp().item(), self.global_step)
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        avg_pred_loss = total_pred_loss / num_batches
        avg_clip_loss = total_clip_loss / num_batches
        
        return avg_loss, avg_pred_loss, avg_clip_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        if self.val_loader is None:
            return None, None, None
        
        self.model.eval()
        total_loss = 0
        total_pred_loss = 0
        total_clip_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move to device
            trajectory_inputs = batch['trajectory_inputs'].to(self.device)
            trajectory_targets = batch['trajectory_targets'].to(self.device)
            trajectory_padding_mask = batch['trajectory_padding_mask'].to(self.device)
            
            visit_inputs = batch['visit_inputs'].to(self.device)
            visit_padding_mask = batch['visit_padding_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                trajectory_tokens=trajectory_inputs,
                visit_tokens=visit_inputs,
                trajectory_padding_mask=trajectory_padding_mask,
                visit_padding_mask=visit_padding_mask,
                trajectory_targets=trajectory_targets,
                compute_alignment=True
            )
            
            # Compute combined loss
            _, loss_dict = compute_combined_loss(
                outputs,
                trajectory_targets,
                trajectory_padding_mask,
                self.prediction_loss_fn,
                self.clip_weight,
                self.prediction_weight
            )
            
            total_loss += loss_dict['total_loss']
            total_pred_loss += loss_dict['prediction_loss']
            total_clip_loss += loss_dict['clip_loss']
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_pred_loss = total_pred_loss / num_batches
        avg_clip_loss = total_clip_loss / num_batches
        
        return avg_loss, avg_pred_loss, avg_clip_loss
    
    def train(self, num_epochs, save_dir='./checkpoints'):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss, train_pred, train_clip = self.train_epoch(epoch)
            print(f"Train - Total: {train_loss:.4f} | Pred: {train_pred:.4f} | CLIP: {train_clip:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_loss, val_pred, val_clip = self.validate()
                print(f"Val   - Total: {val_loss:.4f} | Pred: {val_pred:.4f} | CLIP: {val_clip:.4f}")
                
                # Log validation metrics
                self.writer.add_scalar('Val/TotalLoss', val_loss, epoch)
                self.writer.add_scalar('Val/PredictionLoss', val_pred, epoch)
                self.writer.add_scalar('Val/CLIPLoss', val_clip, epoch)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(save_dir, 'best_model.pt')
                    self.save_checkpoint(checkpoint_path, epoch, val_loss)
                    print(f"✓ Saved best model with val loss: {val_loss:.4f}")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                self.save_checkpoint(checkpoint_path, epoch, train_loss)
            
            # Log current temperature
            print(f"Temperature: {self.model.logit_scale.exp().item():.4f}")
        
        self.writer.close()
        print("\n" + "="*70)
        print("Training completed!")
        print("="*70)
    
    def save_checkpoint(self, path, epoch, loss):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'global_step': self.global_step
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        return checkpoint['epoch'], checkpoint['loss']


def main():
    """Example training script for CLIP mobility model"""
    
    # ============================================================
    # PRIMARY CONFIGURATION
    # ============================================================
    # Trajectory representation parameters
    TRAJECTORY_TOKEN_DIM = 4       # e.g., [lat, lon, time, speed]
    TRAJECTORY_MAX_LEN = 200
    
    # Visit representation parameters
    VISIT_TOKEN_DIM = 6            # e.g., [poi_id, dwell_time, visit_count, ...]
    VISIT_MAX_LEN = 100
    
    # Training configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    # Loss weights
    PREDICTION_WEIGHT = 1.0        # Weight for trajectory prediction loss
    CLIP_WEIGHT = 1.0              # Weight for CLIP alignment loss
    PREDICTION_LOSS_FN = 'mse'     # 'mse', 'mae', or 'huber'
    
    # Model architecture
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    EMBEDDING_DIM = 128            # Dimension of aligned embeddings
    TEMPERATURE = 0.07             # CLIP temperature
    
    # Generate dummy paired data
    print("Generating dummy paired mobility data...")
    import numpy as np
    
    num_train = 1000
    num_val = 200
    
    train_trajectories = []
    train_visits = []
    for _ in range(num_train):
        traj_len = np.random.randint(20, TRAJECTORY_MAX_LEN)
        visit_len = np.random.randint(10, VISIT_MAX_LEN)
        
        train_trajectories.append(np.random.randn(traj_len, TRAJECTORY_TOKEN_DIM))
        train_visits.append(np.random.randn(visit_len, VISIT_TOKEN_DIM))
    
    val_trajectories = []
    val_visits = []
    for _ in range(num_val):
        traj_len = np.random.randint(20, TRAJECTORY_MAX_LEN)
        visit_len = np.random.randint(10, VISIT_MAX_LEN)
        
        val_trajectories.append(np.random.randn(traj_len, TRAJECTORY_TOKEN_DIM))
        val_visits.append(np.random.randn(visit_len, VISIT_TOKEN_DIM))
    
    # Create dataloaders
    print("Creating paired dataloaders...")
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
    
    # Create CLIP model
    print("Creating CLIP mobility model...")
    model = CLIPMobilityModel(
        trajectory_token_dim=TRAJECTORY_TOKEN_DIM,
        trajectory_max_seq_length=TRAJECTORY_MAX_LEN,
        visit_token_dim=VISIT_TOKEN_DIM,
        visit_max_seq_length=VISIT_MAX_LEN,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        learned_pos_encoding=False,
        embedding_dim=EMBEDDING_DIM,
        temperature=TEMPERATURE
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Create trainer
    trainer = CLIPMobilityTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        prediction_loss_fn=PREDICTION_LOSS_FN,
        clip_weight=CLIP_WEIGHT,
        prediction_weight=PREDICTION_WEIGHT,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='./runs/clip_mobility_experiment'
    )
    
    print(f"\nTraining on device: {trainer.device}")
    print(f"Loss configuration:")
    print(f"  Prediction loss: {PREDICTION_LOSS_FN}")
    print(f"  Prediction weight: {PREDICTION_WEIGHT}")
    print(f"  CLIP weight: {CLIP_WEIGHT}")
    
    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=NUM_EPOCHS, save_dir='./checkpoints/clip_mobility')


if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from mobility_transformer import MobilityTransformer, compute_loss
from trajectory_dataloader import create_dataloader


class TrajectoryTrainer:
    """
    Trainer class for the mobility transformer model.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=1e-4,
        weight_decay=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='./runs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
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
            patience=5,
            verbose=True
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        self.global_step = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            inputs = batch['input_ids'].to(self.device)
            targets = batch['target_ids'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = compute_loss(self.model, inputs, targets, padding_mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move to device
            inputs = batch['input_ids'].to(self.device)
            targets = batch['target_ids'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            
            # Forward pass
            loss = compute_loss(self.model, inputs, targets, padding_mask)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
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
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Val Loss: {val_loss:.4f}")
                
                # Log validation loss
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(save_dir, 'best_model.pt')
                    self.save_checkpoint(checkpoint_path, epoch, val_loss)
                    print(f"Saved best model with val loss: {val_loss:.4f}")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                self.save_checkpoint(checkpoint_path, epoch, train_loss)
        
        self.writer.close()
        print("\nTraining completed!")
    
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
    """Example training script"""
    
    # ============================================================
    # PRIMARY CONFIGURATION - CUSTOMIZE FOR YOUR DATA
    # ============================================================
    VOCAB_SIZE = 10000        # YOUR vocabulary size (number of unique tokens)
    MAX_SEQ_LENGTH = 128      # YOUR maximum trajectory length
    
    # Training configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    # Model architecture hyperparameters
    D_MODEL = 256             # Embedding dimension
    NHEAD = 8                 # Number of attention heads (must divide D_MODEL)
    NUM_LAYERS = 6            # Number of transformer layers
    DIM_FEEDFORWARD = 1024    # Feedforward network dimension
    DROPOUT = 0.1             # Dropout rate
    
    # Generate dummy data (replace with your actual tokenized trajectories)
    print("Generating dummy data...")
    train_trajectories = [
        torch.randint(1, VOCAB_SIZE, (torch.randint(20, MAX_SEQ_LENGTH, (1,)).item(),))
        for _ in range(1000)
    ]
    val_trajectories = [
        torch.randint(1, VOCAB_SIZE, (torch.randint(20, MAX_SEQ_LENGTH, (1,)).item(),))
        for _ in range(200)
    ]
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(
        train_trajectories,
        batch_size=BATCH_SIZE,
        max_length=MAX_SEQ_LENGTH,
        pad_token_id=0,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        val_trajectories,
        batch_size=BATCH_SIZE,
        max_length=MAX_SEQ_LENGTH,
        pad_token_id=0,
        shuffle=False
    )
    
    # Create model
    print("Creating model...")
    model = MobilityTransformer(
        vocab_size=VOCAB_SIZE,              # Your vocabulary size
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_length=MAX_SEQ_LENGTH,      # Your max trajectory length
        learned_pos_encoding=False          # Using sinusoidal positional encoding
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Create trainer
    trainer = TrajectoryTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=NUM_EPOCHS, save_dir='./checkpoints')


if __name__ == "__main__":
    main()

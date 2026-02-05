import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as in "Attention is All You Need".
    Can also use learned positional embeddings.
    """
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1, learned=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learned = learned
        
        if learned:
            # Learned positional embeddings
            self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        else:
            # Sinusoidal positional encoding
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # Shape: (1, max_seq_length, d_model)
            
            # Register as buffer (not a parameter, but should be saved with model)
            self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        Returns:
            Tensor of shape (batch_size, seq_length, d_model)
        """
        if self.learned:
            seq_length = x.size(1)
            positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
            pos_embed = self.pos_embedding(positions)
            x = x + pos_embed
        else:
            x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)


class MobilityTransformerVector(nn.Module):
    """
    Transformer model for next-token prediction in GPS trajectory sequences.
    This version handles VECTOR tokens (e.g., [20, 0.3, 45, 12]) instead of scalar IDs.
    Similar to GPT architecture with causal masking.
    """
    def __init__(
        self,
        token_dim,              # NEW: dimension of input token vectors
        output_dim,             # NEW: dimension of output token vectors
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_length=5000,
        learned_pos_encoding=False
    ):
        """
        Args:
            token_dim: Dimension of input token vectors (e.g., 4 for [20, 0.3, 45, 12])
            output_dim: Dimension of output token vectors (usually same as token_dim)
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            learned_pos_encoding: Use learned vs sinusoidal positional encoding
        """
        super().__init__()
        
        self.d_model = d_model
        self.token_dim = token_dim
        self.output_dim = output_dim
        
        # Token projection: project input vectors to d_model dimension
        # Input: (batch, seq_len, token_dim) -> Output: (batch, seq_len, d_model)
        self.token_projection = nn.Linear(token_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model, 
            max_seq_length, 
            dropout, 
            learned=learned_pos_encoding
        )
        
        # Transformer encoder layers (used autoregressively)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Input shape: (batch, seq, feature)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection: project back to token vector dimension
        # Output: (batch, seq_len, d_model) -> (batch, seq_len, output_dim)
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        initrange = 0.1
        self.token_projection.weight.data.uniform_(-initrange, initrange)
        self.token_projection.bias.data.zero_()
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def generate_causal_mask(self, seq_length, device):
        """
        Generate causal mask for autoregressive prediction.
        Prevents attending to future tokens.
        
        Args:
            seq_length: Length of sequence
            device: Device to create mask on
            
        Returns:
            Mask of shape (seq_length, seq_length) with True for masked positions
        """
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, src, src_padding_mask=None):
        """
        Forward pass of the model.
        
        Args:
            src: Input token vectors, shape (batch_size, seq_length, token_dim)
            src_padding_mask: Optional padding mask, shape (batch_size, seq_length)
                             True for positions that should be masked (padding)
        
        Returns:
            Predicted token vectors, shape (batch_size, seq_length, output_dim)
        """
        seq_length = src.size(1)
        device = src.device
        
        # Project token vectors to d_model dimension
        src = self.token_projection(src)
        
        # Scale by sqrt(d_model) as in original transformer
        src = src * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Generate causal mask
        causal_mask = self.generate_causal_mask(seq_length, device)
        
        # Apply transformer with causal masking
        output = self.transformer_encoder(
            src,
            mask=causal_mask,
            src_key_padding_mask=src_padding_mask
        )
        
        # Project back to token vector dimension
        predictions = self.output_projection(output)
        
        return predictions
    
    def predict_next_token(self, src, src_padding_mask=None):
        """
        Predict the next token vector given a sequence.
        
        Args:
            src: Input token vectors, shape (batch_size, seq_length, token_dim)
            src_padding_mask: Optional padding mask
            
        Returns:
            Next token vector predictions, shape (batch_size, output_dim)
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(src, src_padding_mask)
            # Get prediction for last position
            next_token = predictions[:, -1, :]
            
        return next_token
    
    def generate_trajectory(self, initial_tokens, num_steps, src_padding_mask=None):
        """
        Generate a trajectory by autoregressively predicting tokens.
        
        Args:
            initial_tokens: Starting tokens, shape (batch_size, seq_length, token_dim)
            num_steps: Number of steps to generate
            src_padding_mask: Optional padding mask
            
        Returns:
            Generated trajectory, shape (batch_size, seq_length + num_steps, token_dim)
        """
        self.eval()
        current_seq = initial_tokens.clone()
        
        with torch.no_grad():
            for _ in range(num_steps):
                # Predict next token
                next_token = self.predict_next_token(current_seq, src_padding_mask)
                
                # Append to sequence
                next_token = next_token.unsqueeze(1)  # Add sequence dimension
                current_seq = torch.cat([current_seq, next_token], dim=1)
                
                # Update padding mask if needed
                if src_padding_mask is not None:
                    new_mask = torch.zeros(src_padding_mask.size(0), 1, 
                                          dtype=torch.bool, device=src_padding_mask.device)
                    src_padding_mask = torch.cat([src_padding_mask, new_mask], dim=1)
        
        return current_seq


def compute_loss(model, src, targets, padding_mask=None, loss_fn='mse'):
    """
    Compute loss for next-token prediction with vector tokens.
    
    Args:
        model: MobilityTransformerVector model
        src: Input sequences, shape (batch_size, seq_length, token_dim)
        targets: Target sequences (shifted by 1), shape (batch_size, seq_length, token_dim)
        padding_mask: Optional padding mask, shape (batch_size, seq_length)
        loss_fn: Loss function to use ('mse', 'mae', or 'huber')
        
    Returns:
        Average loss value
    """
    predictions = model(src, padding_mask)
    
    # Compute loss based on specified function
    if loss_fn == 'mse':
        loss = F.mse_loss(predictions, targets, reduction='none')
    elif loss_fn == 'mae':
        loss = F.l1_loss(predictions, targets, reduction='none')
    elif loss_fn == 'huber':
        loss = F.huber_loss(predictions, targets, reduction='none', delta=1.0)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")
    
    # Average over token dimensions
    loss = loss.mean(dim=-1)  # Shape: (batch_size, seq_length)
    
    # Apply padding mask if provided
    if padding_mask is not None:
        loss = loss.masked_fill(padding_mask, 0.0)
        # Compute mean only over non-padded positions
        num_valid = (~padding_mask).sum()
        loss = loss.sum() / num_valid
    else:
        loss = loss.mean()
    
    return loss


# Example usage
if __name__ == "__main__":
    # Hyperparameters for VECTOR tokens
    TOKEN_DIM = 4           # Your token vectors have 4 dimensions
    OUTPUT_DIM = 4          # Output same dimension
    BATCH_SIZE = 32
    SEQ_LENGTH = 128
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    
    # Create model for vector tokens
    model = MobilityTransformerVector(
        token_dim=TOKEN_DIM,
        output_dim=OUTPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_length=512,
        learned_pos_encoding=False
    )
    
    # Example batch of tokenized trajectories with VECTOR tokens
    # Shape: (batch_size, seq_length, token_dim)
    batch_tokens = torch.randn(BATCH_SIZE, SEQ_LENGTH, TOKEN_DIM)
    
    # For next-token prediction, targets are input shifted by 1
    # Input: [token_0, token_1, token_2, ..., token_n-1]
    # Target: [token_1, token_2, token_3, ..., token_n]
    inputs = batch_tokens[:, :-1, :]
    targets = batch_tokens[:, 1:, :]
    
    # Optional: create padding mask if sequences have variable length
    padding_mask = None
    
    # Forward pass
    predictions = model(inputs, padding_mask)
    print(f"Input shape: {inputs.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Compute loss (MSE for continuous vectors)
    loss = compute_loss(model, inputs, targets, padding_mask, loss_fn='mse')
    print(f"Loss: {loss.item():.4f}")
    
    # Example prediction
    sample_sequence = batch_tokens[0:1, :10, :]  # First 10 tokens
    next_token = model.predict_next_token(sample_sequence)
    print(f"\nNext token prediction shape: {next_token.shape}")
    print(f"Next token prediction: {next_token[0].tolist()}")
    
    # Example trajectory generation
    print("\n--- Example Trajectory Generation ---")
    initial_seq = batch_tokens[0:1, :5, :]  # Start with 5 tokens
    generated = model.generate_trajectory(initial_seq, num_steps=10)
    print(f"Generated trajectory shape: {generated.shape}")
    print(f"Initial length: 5, Final length: {generated.shape[1]}")
    
    # Training loop example
    print("\n--- Example Training Loop ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        loss = compute_loss(model, inputs, targets, padding_mask, loss_fn='mse')
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

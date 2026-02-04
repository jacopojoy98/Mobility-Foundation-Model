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


class MobilityTransformer(nn.Module):
    """
    Transformer model for next-token prediction in GPS trajectory sequences.
    Similar to GPT architecture with causal masking.
    """
    def __init__(
        self,
        vocab_size,
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
            vocab_size: Number of unique tokens in vocabulary
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
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
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
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
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
            src: Input token indices, shape (batch_size, seq_length)
            src_padding_mask: Optional padding mask, shape (batch_size, seq_length)
                             True for positions that should be masked (padding)
        
        Returns:
            Logits for next token prediction, shape (batch_size, seq_length, vocab_size)
        """
        seq_length = src.size(1)
        device = src.device
        
        # Embed tokens and scale by sqrt(d_model) as in original transformer
        src = self.token_embedding(src) * math.sqrt(self.d_model)
        
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
        
        # Project to vocabulary size
        logits = self.output_projection(output)
        
        return logits
    
    def predict_next_token(self, src, src_padding_mask=None, temperature=1.0, top_k=None):
        """
        Predict the next token given a sequence.
        
        Args:
            src: Input token indices, shape (batch_size, seq_length)
            src_padding_mask: Optional padding mask
            temperature: Sampling temperature (lower = more deterministic)
            top_k: If specified, only sample from top k tokens
            
        Returns:
            Next token predictions, shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(src, src_padding_mask)
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                # Zero out logits for all but top k tokens
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            
        return next_token


def compute_loss(model, src, targets, padding_mask=None):
    """
    Compute cross-entropy loss for next-token prediction.
    
    Args:
        model: MobilityTransformer model
        src: Input sequences, shape (batch_size, seq_length)
        targets: Target sequences (shifted by 1), shape (batch_size, seq_length)
        padding_mask: Optional padding mask
        
    Returns:
        Average loss value
    """
    logits = model(src, padding_mask)
    
    # Reshape for cross entropy: (batch * seq_length, vocab_size)
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    
    # Compute cross entropy loss
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=-100,  # Standard ignore index for padding
        reduction='mean'
    )
    
    return loss


# Example usage and training loop
if __name__ == "__main__":
    # Hyperparameters - CUSTOMIZE THESE FOR YOUR DATA
    VOCAB_SIZE = 10000  # Set this to your actual vocabulary size
    MAX_SEQ_LENGTH = 512  # Set this to your maximum trajectory length
    BATCH_SIZE = 32
    SEQ_LENGTH = 128  # Typical sequence length for this example
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    
    # Create model with your vocabulary size and max sequence length
    model = MobilityTransformer(
        vocab_size=VOCAB_SIZE,  # Your vocabulary size
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_length=MAX_SEQ_LENGTH,  # Your max trajectory length
        learned_pos_encoding=False  # Using sinusoidal encoding
    )
    
    # Example batch of tokenized trajectories
    # In practice, load your actual data
    batch_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    
    # For next-token prediction, targets are input shifted by 1
    # Input: [token_0, token_1, token_2, ..., token_n-1]
    # Target: [token_1, token_2, token_3, ..., token_n]
    inputs = batch_tokens[:, :-1]
    targets = batch_tokens[:, 1:]
    
    # Optional: create padding mask if sequences have variable length
    # padding_mask = (inputs == PAD_TOKEN_ID)  # True for padding positions
    padding_mask = None
    
    # Forward pass
    logits = model(inputs, padding_mask)
    print(f"Input shape: {inputs.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Compute loss
    loss = compute_loss(model, inputs, targets, padding_mask)
    print(f"Loss: {loss.item():.4f}")
    
    # Example prediction
    sample_sequence = batch_tokens[0:1, :10]  # First 10 tokens
    next_token = model.predict_next_token(sample_sequence, temperature=0.8)
    print(f"Next token prediction: {next_token.item()}")
    
    # Training loop example
    print("\n--- Example Training Loop ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        loss = compute_loss(model, inputs, targets, padding_mask)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

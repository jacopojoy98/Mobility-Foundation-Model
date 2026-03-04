import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mobility_transformer_vector import MobilityTransformerVector, PositionalEncoding


class LocationVisitTransformer(nn.Module):
    """
    Transformer for processing sequences of location visits.
    This processes a different representation than the trajectory transformer.
    """
    def __init__(
        self,
        visit_token_dim,        # Dimension of location visit token vectors
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
            visit_token_dim: Dimension of visit token vectors
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
        self.visit_token_dim = visit_token_dim
        
        # Token projection for visit sequences
        self.visit_projection = nn.Linear(visit_token_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model, 
            max_seq_length, 
            dropout, 
            learned=learned_pos_encoding
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        initrange = 0.1
        self.visit_projection.weight.data.uniform_(-initrange, initrange)
        self.visit_projection.bias.data.zero_()
    
    def generate_causal_mask(self, seq_length, device):
        """Generate causal mask for autoregressive prediction"""
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, src, src_padding_mask=None, return_all_tokens=False):
        """
        Forward pass of the visit transformer.
        
        Args:
            src: Input visit token vectors, shape (batch_size, seq_length, visit_token_dim)
            src_padding_mask: Optional padding mask
            return_all_tokens: If True, return all token embeddings; if False, return only [CLS] or mean
        
        Returns:
            If return_all_tokens=False: (batch_size, d_model)
            If return_all_tokens=True: (batch_size, seq_length, d_model)
        """
        seq_length = src.size(1)
        device = src.device
        
        # Project visit vectors to d_model dimension
        src = self.visit_projection(src)
        
        # Scale by sqrt(d_model)
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
        
        if return_all_tokens:
            return output
        else:
            # Return mean pooling over sequence (ignoring padding)
            if src_padding_mask is not None:
                # Mask out padding tokens
                mask = (~src_padding_mask).unsqueeze(-1).float()
                output = (output * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                output = output.mean(dim=1)
            return output


class CLIPMobilityModel(nn.Module):
    """
    Dual-view mobility model with CLIP-style contrastive learning.
    Aligns trajectory representations with location visit representations.
    """
    def __init__(
        self,
        # Trajectory transformer parameters
        trajectory_token_dim,
        trajectory_max_seq_length,
        # Visit transformer parameters
        visit_token_dim,
        visit_max_seq_length,
        # Shared parameters
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        learned_pos_encoding=False,
        # CLIP parameters
        embedding_dim=256,          # Dimension of aligned embeddings
        temperature=0.07            # Temperature for contrastive loss
    ):
        """
        Args:
            trajectory_token_dim: Dimension of trajectory token vectors
            trajectory_max_seq_length: Max length for trajectory sequences
            visit_token_dim: Dimension of visit token vectors
            visit_max_seq_length: Max length for visit sequences
            d_model: Hidden dimension for transformers
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            learned_pos_encoding: Use learned vs sinusoidal encoding
            embedding_dim: Dimension of final aligned embeddings
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # Trajectory transformer (for next-token prediction)
        self.trajectory_transformer = MobilityTransformerVector(
            token_dim=trajectory_token_dim,
            output_dim=trajectory_token_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=trajectory_max_seq_length,
            learned_pos_encoding=learned_pos_encoding
        )
        
        # Visit transformer (for sequence encoding)
        self.visit_transformer = LocationVisitTransformer(
            visit_token_dim=visit_token_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=visit_max_seq_length,
            learned_pos_encoding=learned_pos_encoding
        )
        
        # Projection heads for CLIP alignment
        # Projects from d_model to embedding_dim
        self.trajectory_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, embedding_dim)
        )
        
        self.visit_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, embedding_dim)
        )
        
        # Learnable temperature parameter (optional)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))
    
    def encode_trajectory(self, trajectory_tokens, padding_mask=None):
        """
        Encode trajectory sequence into embedding space.
        
        Args:
            trajectory_tokens: (batch_size, seq_length, trajectory_token_dim)
            padding_mask: (batch_size, seq_length)
            
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        # Get all token representations from trajectory transformer
        # We need the internal representations, not the output predictions
        seq_length = trajectory_tokens.size(1)
        device = trajectory_tokens.device
        
        # Forward through trajectory transformer (replicating internal forward)
        src = self.trajectory_transformer.token_projection(trajectory_tokens)
        src = src * math.sqrt(self.trajectory_transformer.d_model)
        src = self.trajectory_transformer.pos_encoder(src)
        
        causal_mask = self.trajectory_transformer.generate_causal_mask(seq_length, device)
        output = self.trajectory_transformer.transformer_encoder(
            src,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        
        # Mean pooling over sequence (ignoring padding)
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = output.mean(dim=1)
        
        # Project to embedding space
        embeddings = self.trajectory_projection(pooled)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def encode_visits(self, visit_tokens, padding_mask=None):
        """
        Encode visit sequence into embedding space.
        
        Args:
            visit_tokens: (batch_size, seq_length, visit_token_dim)
            padding_mask: (batch_size, seq_length)
            
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        # Forward through visit transformer
        pooled = self.visit_transformer(visit_tokens, padding_mask, return_all_tokens=False)
        
        # Project to embedding space
        embeddings = self.visit_projection(pooled)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def compute_clip_loss(self, trajectory_embeddings, visit_embeddings):
        """
        Compute CLIP-style contrastive loss.
        
        Args:
            trajectory_embeddings: (batch_size, embedding_dim)
            visit_embeddings: (batch_size, embedding_dim)
            
        Returns:
            loss: scalar
        """
        batch_size = trajectory_embeddings.size(0)
        
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_trajectory = logit_scale * trajectory_embeddings @ visit_embeddings.t()
        logits_per_visit = logits_per_trajectory.t()
        
        # Symmetric cross-entropy loss
        labels = torch.arange(batch_size, device=trajectory_embeddings.device)
        
        loss_trajectory = F.cross_entropy(logits_per_trajectory, labels)
        loss_visit = F.cross_entropy(logits_per_visit, labels)
        
        clip_loss = (loss_trajectory + loss_visit) / 2
        
        return clip_loss
    
    def forward(
        self, 
        trajectory_tokens, 
        visit_tokens,
        trajectory_padding_mask=None,
        visit_padding_mask=None,
        trajectory_targets=None,
        compute_alignment=True
    ):
        """
        Full forward pass with both trajectory prediction and CLIP alignment.
        
        Args:
            trajectory_tokens: (batch_size, traj_seq_len, trajectory_token_dim)
            visit_tokens: (batch_size, visit_seq_len, visit_token_dim)
            trajectory_padding_mask: (batch_size, traj_seq_len)
            visit_padding_mask: (batch_size, visit_seq_len)
            trajectory_targets: (batch_size, traj_seq_len, trajectory_token_dim) for prediction loss
            compute_alignment: Whether to compute CLIP alignment loss
            
        Returns:
            Dictionary containing:
                - trajectory_predictions: next-token predictions
                - trajectory_embeddings: aligned embeddings
                - visit_embeddings: aligned embeddings
                - clip_loss: contrastive alignment loss (if compute_alignment=True)
        """
        outputs = {}
        
        # 1. Trajectory next-token prediction
        trajectory_predictions = self.trajectory_transformer(
            trajectory_tokens, 
            trajectory_padding_mask
        )
        outputs['trajectory_predictions'] = trajectory_predictions
        
        # 2. Encode both modalities for CLIP alignment
        if compute_alignment:
            trajectory_embeddings = self.encode_trajectory(
                trajectory_tokens, 
                trajectory_padding_mask
            )
            visit_embeddings = self.encode_visits(
                visit_tokens,
                visit_padding_mask
            )
            
            outputs['trajectory_embeddings'] = trajectory_embeddings
            outputs['visit_embeddings'] = visit_embeddings
            
            # 3. Compute CLIP loss
            clip_loss = self.compute_clip_loss(trajectory_embeddings, visit_embeddings)
            outputs['clip_loss'] = clip_loss
        
        return outputs


def compute_combined_loss(
    model_outputs,
    trajectory_targets,
    trajectory_padding_mask=None,
    prediction_loss_fn='mse',
    clip_weight=1.0,
    prediction_weight=1.0
):
    """
    Compute combined loss: trajectory prediction + CLIP alignment.
    
    Args:
        model_outputs: Dictionary from model forward pass
        trajectory_targets: Ground truth for next-token prediction
        trajectory_padding_mask: Padding mask
        prediction_loss_fn: 'mse', 'mae', or 'huber'
        clip_weight: Weight for CLIP loss
        prediction_weight: Weight for prediction loss
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual losses
    """
    # 1. Trajectory prediction loss
    predictions = model_outputs['trajectory_predictions']
    
    if prediction_loss_fn == 'mse':
        pred_loss = F.mse_loss(predictions, trajectory_targets, reduction='none')
    elif prediction_loss_fn == 'mae':
        pred_loss = F.l1_loss(predictions, trajectory_targets, reduction='none')
    elif prediction_loss_fn == 'huber':
        pred_loss = F.huber_loss(predictions, trajectory_targets, reduction='none', delta=1.0)
    else:
        raise ValueError(f"Unknown loss function: {prediction_loss_fn}")
    
    # Average over token dimensions
    pred_loss = pred_loss.mean(dim=-1)
    
    # Apply padding mask if provided
    if trajectory_padding_mask is not None:
        pred_loss = pred_loss.masked_fill(trajectory_padding_mask, 0.0)
        num_valid = (~trajectory_padding_mask).sum()
        pred_loss = pred_loss.sum() / num_valid
    else:
        pred_loss = pred_loss.mean()
    
    # 2. CLIP alignment loss
    clip_loss = model_outputs.get('clip_loss', 0.0)
    
    # 3. Combined loss
    total_loss = prediction_weight * pred_loss + clip_weight * clip_loss
    
    loss_dict = {
        'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        'prediction_loss': pred_loss.item(),
        'clip_loss': clip_loss.item() if isinstance(clip_loss, torch.Tensor) else clip_loss,
    }
    
    return total_loss, loss_dict


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 16
    TRAJECTORY_TOKEN_DIM = 4      # e.g., [lat, lon, time, speed]
    VISIT_TOKEN_DIM = 6           # e.g., [poi_id, dwell_time, visit_count, ...]
    TRAJECTORY_SEQ_LEN = 100
    VISIT_SEQ_LEN = 50
    D_MODEL = 256
    EMBEDDING_DIM = 128
    
    # Create dual-view model
    model = CLIPMobilityModel(
        trajectory_token_dim=TRAJECTORY_TOKEN_DIM,
        trajectory_max_seq_length=200,
        visit_token_dim=VISIT_TOKEN_DIM,
        visit_max_seq_length=100,
        d_model=D_MODEL,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        learned_pos_encoding=False,
        embedding_dim=EMBEDDING_DIM,
        temperature=0.07
    )
    
    # Example data (same user, two different representations)
    trajectory_tokens = torch.randn(BATCH_SIZE, TRAJECTORY_SEQ_LEN, TRAJECTORY_TOKEN_DIM)
    visit_tokens = torch.randn(BATCH_SIZE, VISIT_SEQ_LEN, VISIT_TOKEN_DIM)
    
    # Targets for next-token prediction
    trajectory_targets = torch.randn(BATCH_SIZE, TRAJECTORY_SEQ_LEN, TRAJECTORY_TOKEN_DIM)
    
    # Forward pass
    print("="*60)
    print("CLIP Mobility Model - Example Forward Pass")
    print("="*60)
    
    outputs = model(
        trajectory_tokens=trajectory_tokens,
        visit_tokens=visit_tokens,
        trajectory_targets=trajectory_targets,
        compute_alignment=True
    )
    
    print(f"\nModel outputs:")
    print(f"  Trajectory predictions shape: {outputs['trajectory_predictions'].shape}")
    print(f"  Trajectory embeddings shape: {outputs['trajectory_embeddings'].shape}")
    print(f"  Visit embeddings shape: {outputs['visit_embeddings'].shape}")
    print(f"  CLIP loss: {outputs['clip_loss'].item():.4f}")
    
    # Compute combined loss
    total_loss, loss_dict = compute_combined_loss(
        outputs,
        trajectory_targets,
        trajectory_padding_mask=None,
        prediction_loss_fn='mse',
        clip_weight=1.0,
        prediction_weight=1.0
    )
    
    print(f"\nLoss breakdown:")
    print(f"  Total loss: {loss_dict['total_loss']:.4f}")
    print(f"  Prediction loss: {loss_dict['prediction_loss']:.4f}")
    print(f"  CLIP loss: {loss_dict['clip_loss']:.4f}")
    
    # Demonstrate similarity matching
    print("\n" + "="*60)
    print("CLIP Alignment - Similarity Matrix")
    print("="*60)
    
    # Compute cosine similarity between trajectory and visit embeddings
    traj_emb = outputs['trajectory_embeddings']
    visit_emb = outputs['visit_embeddings']
    
    similarity_matrix = traj_emb @ visit_emb.t()
    print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
    print(f"Diagonal (matched pairs) mean: {similarity_matrix.diag().mean().item():.4f}")
    print(f"Off-diagonal (mismatched) mean: {(similarity_matrix.sum() - similarity_matrix.diag().sum()).item() / (BATCH_SIZE * (BATCH_SIZE - 1)):.4f}")
    
    # Training step
    print("\n" + "="*60)
    print("Example Training Step")
    print("="*60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    print(f"Training step completed!")
    print(f"Learnable temperature (logit_scale): {model.logit_scale.exp().item():.4f}")

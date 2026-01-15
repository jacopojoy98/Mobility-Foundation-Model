import torch
import torch.nn as nn

class Trajectory_transformer(nn.Module):
    def __init__(self, 
                 number_features: int, 
                 tokenizer_dimension: int, 
                 latent_space_dimension: int, 
                 nhead: int,
                 ):
        super().__init__()
        self.positional_encoder = 0 
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(tokenizer_dimension, nhead))
        self.latent_space_dimension = latent_space_dimension
        self.preliminary_linear_layer = nn.Linear(number_features, tokenizer_dimension)
    def forward(self, x):
        Batch, Lenght, Embedding_in = x.shape

        preliminary_embedding = self.preliminary_linear_layer(x)

        embedding = self.transformer_encoder(preliminary_embedding)

        
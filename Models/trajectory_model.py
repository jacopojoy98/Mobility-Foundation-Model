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

    def forward(self, x, trajectory_lenghts):
        '''
        Docstring for forward

        :param x: The data to process trough the transformer,
            different trajectories are padded up to the longer trajectories
        :param trajectory_lenghts: The actual tenght of the trajectories
            (maybe should be a function of x, probably faster like this)
        '''
        Batch, Lenght, Input_Embedding_Dimension = x.shape

        attention_mask = attention_mask(Batch, Lenght, Input_Embedding_Dimension)
        padding_mask = padding_mask(Batch,Lenght, Input_Embedding_Dimension)

        preliminary_embedding = self.preliminary_linear_layer(x)

        embedding = self.transformer_encoder(preliminary_embedding)

        
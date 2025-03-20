import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        block = [
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        ]
        
        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1800, n_residual_blocks=5, dropout=0.2):
        super(Generator, self).__init__()
        
        # Initial block - expand to hidden dimension
        model = [
            nn.Linear(input_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        ]
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(hidden_dim, dropout)]
            
        # Output layers - reduce back to output dimension
        model += [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # normalize to [-1, 1]
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(Discriminator, self).__init__()
        
        # For 900-dimensional data, use a deeper network with progressive dimension reduction
        hidden_dims = [1024, 512, 256, 128, 64]
        
        model = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout)
        ]
        
        for i in range(len(hidden_dims)-1):
            model += [
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout)
            ]
            
        # Output layer
        model += [nn.Linear(hidden_dims[-1], 1)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)
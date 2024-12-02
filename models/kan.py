import torch
import torch.nn as nn
from kan import KAN  # Import the KAN model

class KANExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_channels, grid, k, device):
        super(KANExtractor, self).__init__()
        # Initialize the KAN model
        self.kan = KAN(
            width=[[input_dim, 0], [hidden_channels, 0], [output_dim, 0]],  # KAN architecture
            grid=grid,  # Grid size for KAN
            k=k,  # Number of kernels
            seed=42,  # Random seed for reproducibility
            device=device  # Specify the computation device (CPU/GPU)
        )
        self.device = device  # Save the device for moving tensors

    def forward(self, x):
        """
        Forward pass through the KAN model.
        Extracts features by processing input data through the KAN layers.
        """
        # Ensure the input is on the correct device
        x = x.to(self.device)

        # Pass the input through the KAN model
        features = self.kan.fit(x)  # `fit` trains the KAN and extracts features

        return features
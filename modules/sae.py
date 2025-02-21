import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.gain = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.gain

class Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            RMSNorm(output_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class SAE(nn.Module):
    def __init__(self, dims: list[int]):
        """
        Initialize a skip connection autoencoder with arbitrary layer dimensions.
        
        Args:
            dims (List[int]): List of dimensions for the encoder pathway.
                            First element is input dimension, last is latent dimension.
                            Example: [128, 64, 32, 16] creates an encoder with layers:
                            128 -> 64 -> 32 -> 16 and corresponding decoder with skip connections
        """
        super().__init__()
        
        if len(dims) < 3:
            raise ValueError("Need at least 3 dimensions (input, hidden, latent)")
            
        self.dims = dims
        self.n_layers = len(dims) - 1
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.GELU()
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            Block(dims[i], dims[i+1]) 
            for i in range(self.n_layers)
        ])
        
        # Decoder layers
        # Note: Each decoder layer (except the last) takes twice the dimensions
        # due to skip connections
        self.decoder_layers = nn.ModuleList()
        
        # First decoder layer (from latent space, no skip connection yet)
        self.decoder_layers.append(
            Block(dims[-1], dims[-2])
        )
        
        # Remaining decoder layers (with skip connections)
        for i in range(self.n_layers - 2, -1, -1):
            # Input dimension is doubled due to skip connection
            in_dim = dims[i+1] * 2
            out_dim = dims[i]
            self.decoder_layers.append(Block(in_dim, out_dim))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_activations = []
        
        # Encoder
        current = x
        for i, layer in enumerate(self.encoder_layers):
            current = layer(current)
            if i < self.n_layers - 1:  # Don't apply activation to latent space
                current = self.dropout(self.activation(current))
            encoder_activations.append(current)
            
        # Decoder
        current = encoder_activations[-1]  # Start from latent space
        for i, layer in enumerate(self.decoder_layers):
            current = layer(current)
            
            # Don't apply activation to final output
            if i < len(self.decoder_layers) - 1:
                current = self.dropout(self.activation(current))
                
                # Add skip connection by concatenating with corresponding encoder activation
                # Note: we go backwards through encoder activations, skipping the latent space
                skip_connection = encoder_activations[-(i+2)]
                current = torch.cat((current, skip_connection), dim=1)
                
        return current

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get the latent space representation for input x."""
        current = x
        for i, layer in enumerate(self.encoder_layers):
            current = layer(current)
            if i < self.n_layers - 1:
                current = self.dropout(self.activation(current))
        return current

# Example usage
if __name__ == "__main__":
    # Test with different dimension configurations
    dims_configs = [
        [128, 64, 32, 16],  # Original 4-layer configuration
        # [256, 128, 64, 32, 16],  # 5-layer configuration
        # [512, 256, 128, 64],  # 4-layer with different dimensions
    ]
    
    batch_size = 10
    
    for dims in dims_configs:
        print(f"\nTesting configuration: {dims}")
        model = SAE(dims)
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, dims[0])
        
        # Forward pass
        reconstructed = model(dummy_input)
        
        # Get latent representation
        latent = model.get_latent(dummy_input)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Latent shape: {latent.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        # Verify shapes
        assert reconstructed.shape == dummy_input.shape, "Input and output shapes don't match!"
        assert latent.shape == (batch_size, dims[-1]), "Unexpected latent dimension!"

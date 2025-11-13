import yaml
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import defaultdict

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

cuda_devices = config["general"]["cuda_devices"]
device = torch.device(f"cuda:{cuda_devices}" if torch.cuda.is_available() else "cpu")


# =================== Custom Modules ===================

class JumpReLU(nn.Module):
    def __init__(self, gamma=1.0, beta=1.0):
        super(JumpReLU, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)

    def forward(self, x):
        return F.relu(x) + self.beta * (x > self.gamma).float()


def topk_mask(tensor, k):
    """Apply top-k sparsity mask to tensor"""
    if k is None or k >= tensor.shape[1]:
        return tensor
    
    batch_size, dim = tensor.shape
    
    # Get top-k indices based on absolute values
    topk_vals, topk_indices = torch.topk(tensor.abs(), k, dim=1)
    
    # Create mask
    mask = torch.zeros_like(tensor, device=tensor.device)
    mask.scatter_(1, topk_indices, 1)
    
    return tensor * mask


# =================== Matryoshka Transcoder ===================

class MatryoshkaTranscoder(nn.Module):
    def __init__(self, input_dim, output_dim, matryoshka_config, jump_params=None):
        """
        Matryoshka Transcoder with hierarchical nested structure
        
        Args:
            input_dim: Dimension of input activations (h_2)
            output_dim: Dimension of output activations (h_1)
            matryoshka_config: Dict containing:
                - levels: List of cumulative latent dimensions [64, 128, 256, 512, 1024]
                - topk_per_level: List of top-k sparsity values [8, 16, 32, 64, 128]
                - level_weights: List of loss weights [0.1, 0.15, 0.2, 0.25, 0.3]
            jump_params: Tuple of (gamma, beta) for JumpReLU
        """
        super(MatryoshkaTranscoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Matryoshka configuration
        self.levels = matryoshka_config['levels']
        self.topk_per_level = matryoshka_config['topk_per_level']
        self.level_weights = matryoshka_config['level_weights']
        self.num_levels = len(self.levels)
        self.latent_dim = self.levels[-1]  # Maximum latent dimension
        
        # Validate configuration
        assert len(self.levels) == len(self.topk_per_level), "Levels and topk must have same length"
        assert len(self.levels) == len(self.level_weights), "Levels and weights must have same length"
        assert abs(sum(self.level_weights) - 1.0) < 1e-5, f"Level weights must sum to 1.0, got {sum(self.level_weights)}"
        assert all(self.levels[i] < self.levels[i+1] for i in range(len(self.levels)-1)), "Levels must be strictly increasing"
        
        gamma, beta = jump_params if jump_params else (1.0, 1.0)
        
        # Single encoder: input -> full latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.latent_dim, bias=True),
            JumpReLU(gamma=gamma, beta=beta)
        )
        
        # Multiple decoders: one per level (cumulative latents -> output)
        self.decoders = nn.ModuleList([
            nn.Linear(level_dim, output_dim, bias=True)
            for level_dim in self.levels
        ])
        
        print(f"\n{'='*60}")
        print("Matryoshka Transcoder Architecture:")
        print(f"{'='*60}")
        print(f"  Input dim (h_2): {self.input_dim}")
        print(f"  Output dim (h_1): {self.output_dim}")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Levels: {self.levels}")
        print(f"  Top-k per level: {self.topk_per_level}")
        print(f"  Level weights: {self.level_weights}")
        print(f"{'='*60}\n")
    
    def apply_nested_topk(self, z):
        """
        Apply hierarchical top-k sparsity
        Each level gets independent top-k selection within its dimension range
        
        Key insight: Earlier levels must capture general features,
        later levels can capture more specific features
        """
        z_sparse = torch.zeros_like(z)
        
        # Level boundaries
        level_starts = [0] + self.levels[:-1]
        level_ends = self.levels
        
        for i, (start, end, k) in enumerate(zip(level_starts, level_ends, self.topk_per_level)):
            # Extract features for this level
            level_features = z[:, start:end]
            
            # Apply top-k within this level's dimension range
            level_features_sparse = topk_mask(level_features, k)
            
            # Place back into full sparse vector
            z_sparse[:, start:end] = level_features_sparse
        
        return z_sparse
    
    def forward(self, h_2, return_all_levels=False):
        """
        Forward pass with multi-level reconstruction
        
        Args:
            h_2: Input activations [batch_size, input_dim]
            return_all_levels: If True, return reconstructions at all levels
        
        Returns:
            If return_all_levels=True:
                reconstructions: List of h_1 reconstructions for each level
                z_sparse: Sparse latent representation
            Else:
                h_1_recon: Final reconstruction (from largest level)
                z_sparse: Sparse latent representation
        """
        # Encode to full latent space
        z = self.encoder(h_2)
        
        # Apply nested top-k sparsity
        z_sparse = self.apply_nested_topk(z)
        
        # Decode at each level using cumulative features
        reconstructions = []
        for i, decoder in enumerate(self.decoders):
            # Use features up to this level (cumulative)
            z_level = z_sparse[:, :self.levels[i]]
            
            # Reconstruct output activations
            h_1_recon = decoder(z_level)
            reconstructions.append(h_1_recon)
        
        if return_all_levels:
            return reconstructions, z_sparse
        else:
            # Return only the final (most complete) reconstruction
            return reconstructions[-1], z_sparse
    
    def encode(self, h_2, level=None):
        """
        Encode inputs at specific granularity level
        
        Args:
            h_2: Input activations
            level: Which level to return (0-indexed). If None, returns full encoding
        
        Returns:
            Sparse latent representation up to specified level
        """
        z = self.encoder(h_2)
        z_sparse = self.apply_nested_topk(z)
        
        if level is not None:
            # Return only up to specified level
            return z_sparse[:, :self.levels[level]]
        
        return z_sparse
    
    def get_activations(self, h_2):
        """Get sparse latent activations (for analysis/interpretation)"""
        return self.encode(h_2)


# =================== Training ===================

def train_matryoshka_transcoder(model, dataloader, num_epochs, learning_rate, 
                                 model_save_path, lambda_sparse=0.01):
    """
    Train Matryoshka Transcoder with multi-level reconstruction
    
    Args:
        model: MatryoshkaTranscoder instance
        dataloader: DataLoader with (h_2, h_1) pairs
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        model_save_path: Path to save model checkpoints
        lambda_sparse: Weight for L1 sparsity loss
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler with warmup
    warmup_epochs = min(5, num_epochs // 10)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=learning_rate * 0.01
    )
    
    best_loss = float('inf')
    
    # Track losses per level
    level_losses_history = defaultdict(list)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Accumulators
        total_loss = 0
        total_recon_loss = 0
        total_sparse_loss = 0
        level_losses = [0.0] * model.num_levels
        level_sparsities = [0.0] * model.num_levels
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for h_2, h_1 in pbar:
                h_2, h_1 = h_2.to(device), h_1.to(device)
                
                # Forward pass - get all level reconstructions
                reconstructions, z_sparse = model(h_2, return_all_levels=True)
                
                # Compute multi-level reconstruction loss
                recon_loss = 0
                for i, h_1_recon in enumerate(reconstructions):
                    loss_level = criterion(h_1_recon, h_1)
                    
                    # Weighted by level importance
                    weighted_loss = model.level_weights[i] * loss_level
                    recon_loss += weighted_loss
                    
                    # Track per-level losses
                    level_losses[i] += loss_level.item()
                
                # Sparsity loss (L1 on latent activations)
                sparse_loss = lambda_sparse * z_sparse.abs().mean()
                
                # Total loss
                total_loss_batch = recon_loss + sparse_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss_batch.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Accumulate losses
                total_loss += total_loss_batch.item()
                total_recon_loss += recon_loss.item()
                total_sparse_loss += sparse_loss.item()
                
                # Track sparsity statistics
                for i in range(model.num_levels):
                    z_level = z_sparse[:, :model.levels[i]]
                    active_ratio = (z_level != 0).float().mean().item()
                    level_sparsities[i] += active_ratio
                
                # Update progress bar
                pbar.set_postfix({
                    "Loss": f"{total_loss_batch.item():.4f}",
                    "Recon": f"{recon_loss.item():.4f}",
                    "Sparse": f"{sparse_loss.item():.6f}"
                })
        
        # Calculate epoch averages
        num_batches = len(dataloader)
        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_sparse_loss = total_sparse_loss / num_batches
        avg_level_losses = [loss / num_batches for loss in level_losses]
        avg_level_sparsities = [sparsity / num_batches for sparsity in level_sparsities]
        
        # Store history
        for i, loss in enumerate(avg_level_losses):
            level_losses_history[f"level_{i}"].append(loss)
        
        # Update learning rate (after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Print detailed statistics
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Total Loss: {avg_total_loss:.6f}")
        print(f"  Recon Loss: {avg_recon_loss:.6f}")
        print(f"  Sparse Loss: {avg_sparse_loss:.6f}")
        print(f"  Per-Level Statistics:")
        for i in range(model.num_levels):
            print(f"    Level {i} (dim={model.levels[i]}, k={model.topk_per_level[i]}):")
            print(f"      Recon Loss: {avg_level_losses[i]:.6f}")
            print(f"      Sparsity: {100 * (1 - avg_level_sparsities[i]):.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_model_path = model_save_path.replace('.pt', '_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'level_losses': avg_level_losses,
                'config': {
                    'input_dim': model.input_dim,
                    'output_dim': model.output_dim,
                    'levels': model.levels,
                    'topk_per_level': model.topk_per_level,
                    'level_weights': model.level_weights
                }
            }, best_model_path)
            print(f"  ✓ New best model saved! Loss: {best_loss:.6f}")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_total_loss,
        'level_losses': avg_level_losses,
        'level_losses_history': dict(level_losses_history),
        'config': {
            'input_dim': model.input_dim,
            'output_dim': model.output_dim,
            'levels': model.levels,
            'topk_per_level': model.topk_per_level,
            'level_weights': model.level_weights
        }
    }, model_save_path)
    
    print(f"\n✓ Final model saved to {model_save_path}")
    print(f"✓ Best model saved to {best_model_path}")
    
    return model


# =================== Main Training Loop ===================

if __name__ == "__main__":
    # Load activations
    loaded = np.load("activations_xyz.npz")
    h_2 = loaded["x"]  # Base model activations (768-dim CLIP embeddings)
    h_1 = loaded["y"]  # Hidden layer activations (256-dim)
    h_0 = loaded["z"]  # Classification outputs (1-dim) - not used for transcoder
    
    print(f"Loaded activations:")
    print(f"  h_2 shape: {h_2.shape}")
    print(f"  h_1 shape: {h_1.shape}")
    print(f"  h_0 shape: {h_0.shape}")
    
    # Convert to tensors
    h_2_tensor = torch.tensor(h_2, dtype=torch.float32)
    h_1_tensor = torch.tensor(h_1, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(h_2_tensor, h_1_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Matryoshka configuration
    matryoshka_config = {
        'levels': [128, 256, 512, 1024, 2048],  # Cumulative latent dimensions
        'topk_per_level': [16, 32, 64, 128, 256],  # Top-k sparsity at each level
        'level_weights': [0.1, 0.15, 0.2, 0.25, 0.3]  # Loss weights (must sum to 1.0)
    }
    
    # Training configuration
    num_epochs = config["general"].get("num_epochs", 100)
    learning_rate = 1e-4
    lambda_sparse = 0.01
    jump_params = (1.0, 1.0)  # (gamma, beta) for JumpReLU
    
    # Train multiple models if needed
    for model_num in range(1):  # Change range for multiple runs
        print(f"\n{'='*60}")
        print(f"Training Matryoshka Transcoder - Model {model_num}")
        print(f"{'='*60}\n")
        
        # Initialize model
        model = MatryoshkaTranscoder(
            input_dim=h_2_tensor.shape[1],
            output_dim=h_1_tensor.shape[1],
            matryoshka_config=matryoshka_config,
            jump_params=jump_params
        ).to(device)
        
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Training on: {device}\n")
        
        # Train model
        os.makedirs("./models", exist_ok=True)
        model_save_path = f"./models/matryoshka_transcoder_{model_num}.pt"
        
        model = train_matryoshka_transcoder(
            model=model,
            dataloader=dataloader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            model_save_path=model_save_path,
            lambda_sparse=lambda_sparse
        )
        
        print(f"\n{'='*60}")
        print(f"Model {model_num} Training Complete!")
        print(f"{'='*60}\n")
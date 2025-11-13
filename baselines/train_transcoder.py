import yaml
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

cuda_devices = config["general"]["cuda_devices"]
device = torch.device(f"cuda:{cuda_devices}" if torch.cuda.is_available() else "cpu")


# =================== Utilities ===================

class JumpReLU(nn.Module):
    def __init__(self, gamma=1.0, beta=1.0):
        super(JumpReLU, self).__init__()
        # keep gamma/beta as non-trainable params (same behavior as original)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)

    def forward(self, x):
        return F.relu(x) + self.beta * (x > self.gamma).float()


def topk_mask(tensor, k):
    """
    Apply top-k sparsity mask across the last dimension of tensor.
    If k is None or >= dim, no masking applied.
    tensor: [batch, dim]
    """
    if k is None:
        return tensor
    dim = tensor.shape[1]
    if k >= dim:
        return tensor

    # topk by absolute value
    _, topk_indices = torch.topk(tensor.abs(), k, dim=1)
    mask = torch.zeros_like(tensor, device=tensor.device)
    mask.scatter_(1, topk_indices, 1.0)
    return tensor * mask


# =================== Simple Transcoder (single-loss) ===================

class SimpleTranscoder(nn.Module):
    """
    Single-encoder, single-decoder transcoder that produces one reconstruction.
    """
    def __init__(self, input_dim, output_dim, latent_dim=512, topk=None, jump_params=None):
        super(SimpleTranscoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.topk = topk

        gamma, beta = jump_params if jump_params else (1.0, 1.0)

        # Encoder: input -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim, bias=True),
            JumpReLU(gamma=gamma, beta=beta)
        )

        # Decoder: latent -> output
        self.decoder = nn.Linear(latent_dim, output_dim, bias=True)

        print(f"\n{'='*60}")
        print("Simple Transcoder Architecture:")
        print(f"{'='*60}")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Output dim: {self.output_dim}")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Top-k (latent active): {self.topk}")
        print(f"{'='*60}\n")

    def forward(self, h_2):
        # encode
        z = self.encoder(h_2)                    # [batch, latent_dim]

        # apply global top-k sparsity to latent
        z_sparse = topk_mask(z, self.topk)       # [batch, latent_dim]

        # decode
        h_1_recon = self.decoder(z_sparse)       # [batch, output_dim]

        return h_1_recon, z_sparse

    def encode(self, h_2):
        z = self.encoder(h_2)
        return topk_mask(z, self.topk)
    
    def get_activations(self, h_2):
        """Get sparse latent activations (for analysis/interpretation)"""
        return self.encode(h_2)


# =================== Training (single combined loss) ===================

def train_simple_transcoder(model, dataloader, num_epochs, learning_rate,
                            model_save_path, lambda_sparse=0.01):
    """
    Train the simple transcoder using a single scalar loss:
      loss = MSE(recon, h1) + lambda_sparse * mean(abs(z_sparse))
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # optional warmup scheduling (keeps similar structure as original)
    warmup_epochs = min(5, num_epochs // 10)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=learning_rate * 0.01
    )

    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_sparse = 0.0
        num_batches = 0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for h_2, h_1 in pbar:
                h_2 = h_2.to(device)
                h_1 = h_1.to(device)

                recon, z_sparse = model(h_2)

                recon_loss = criterion(recon, h_1)
                sparse_loss = lambda_sparse * z_sparse.abs().mean()
                loss = recon_loss + sparse_loss  # single scalar loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_sparse += sparse_loss.item()
                num_batches += 1

                # progress
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "sparse": f"{sparse_loss.item():.6f}"
                })

        # scheduler step after warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = total_loss / max(1, num_batches)
        avg_recon = total_recon / max(1, num_batches)
        avg_sparse = total_sparse / max(1, num_batches)

        print(f"\nEpoch [{epoch+1}/{num_epochs}]  Avg Loss: {avg_loss:.6f}  Recon: {avg_recon:.6f}  Sparse: {avg_sparse:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = model_save_path.replace('.pt', '_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'input_dim': model.input_dim,
                    'output_dim': model.output_dim,
                    'latent_dim': model.latent_dim,
                    'topk': model.topk
                }
            }, best_model_path)
            print(f"  ✓ New best model saved! Loss: {best_loss:.6f}")

    # save final
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': {
            'input_dim': model.input_dim,
            'output_dim': model.output_dim,
            'latent_dim': model.latent_dim,
            'topk': model.topk
        }
    }, model_save_path)

    print(f"\n✓ Final model saved to {model_save_path}")
    print(f"✓ Best model saved to {best_model_path}")

    return model


# =================== Main ===================

if __name__ == "__main__":
    # Load activations
    loaded = np.load("activations_xyz.npz")
    h_2 = loaded["x"]  # Base model activations (e.g. 768-dim)
    h_1 = loaded["y"]  # Hidden layer activations (e.g. 256-dim)

    print(f"Loaded activations:")
    print(f"  h_2 shape: {h_2.shape}")
    print(f"  h_1 shape: {h_1.shape}")

    # Convert to tensors
    h_2_tensor = torch.tensor(h_2, dtype=torch.float32)
    h_1_tensor = torch.tensor(h_1, dtype=torch.float32)

    # Dataset and loader
    dataset = TensorDataset(h_2_tensor, h_1_tensor)
    dataloader = DataLoader(dataset, batch_size=config["general"].get("batch_size", 32), shuffle=True, num_workers=0)

    # Simple transcoder config (tweak as needed)
    latent_dim = 2048
    topk = 256
    num_epochs = config["general"].get("num_epochs", 100)
    learning_rate = config["general"].get("learning_rate", 1e-4)
    lambda_sparse = config["general"].get("lambda_sparse", 0.01)
    jump_params = (1.0,1.0)

    # Create model
    model = SimpleTranscoder(
        input_dim=h_2_tensor.shape[1],
        output_dim=h_1_tensor.shape[1],
        latent_dim=latent_dim,
        topk=topk,
        jump_params=jump_params
    ).to(device)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on: {device}\n")

    os.makedirs("./models", exist_ok=True)
    model_save_path = "./models/simple_transcoder.pt"

    model = train_simple_transcoder(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        model_save_path=model_save_path,
        lambda_sparse=lambda_sparse
    )

    print("\nTraining complete.")

import yaml
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------------------------
# Load configuration
# ---------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

cuda_devices = config["general"].get("cuda_devices", 0)
device = torch.device(f"cuda:{cuda_devices}" if torch.cuda.is_available() else "cpu")

# mode: "transcoder" or "sae"
mode = "sae"

# ---------------------------
# Utilities
# ---------------------------

class JumpReLU(nn.Module):
    def __init__(self, gamma=1.0, beta=1.0):
        super(JumpReLU, self).__init__()
        # preserved as non-trainable to keep original behavior
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)

    def forward(self, x):
        return F.relu(x) + self.beta * (x > self.gamma).float()


def topk_mask(tensor, k):
    """
    Apply top-k sparsity mask across last dim of tensor (per-row top-k).
    If k is None or k >= dim, returns tensor unchanged.
    tensor: [batch, dim]
    """
    if k is None:
        return tensor
    dim = tensor.shape[1]
    if k >= dim:
        return tensor

    # topk by absolute value (per-row)
    _, idx = torch.topk(tensor.abs(), k, dim=1)
    mask = torch.zeros_like(tensor, device=tensor.device)
    mask.scatter_(1, idx, 1.0)
    return tensor * mask

# ---------------------------
# Models
# ---------------------------

class SimpleTranscoder(nn.Module):
    """
    Encoder (h2 -> latent) + top-k sparse latent + decoder (latent -> h1).
    Used when mode == "transcoder".
    """
    def __init__(self, input_dim, output_dim, latent_dim=512, topk=None, jump_params=None):
        super(SimpleTranscoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.topk = topk

        gamma, beta = jump_params if jump_params else (1.0, 1.0)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim, bias=True),
            JumpReLU(gamma=gamma, beta=beta)
        )
        self.decoder = nn.Linear(latent_dim, output_dim, bias=True)

        print(f"\n{'='*60}")
        print("Simple Transcoder:")
        print(f"  input_dim(h2) = {self.input_dim}")
        print(f"  output_dim(h1) = {self.output_dim}")
        print(f"  latent_dim = {self.latent_dim}")
        print(f"  topk = {self.topk}")
        print(f"{'='*60}\n")

    def forward(self, h_2):
        z = self.encoder(h_2)
        z_sparse = topk_mask(z, self.topk)
        h_1_recon = self.decoder(z_sparse)
        return h_1_recon, z_sparse

    def encode(self, h_2):
        z = self.encoder(h_2)
        return topk_mask(z, self.topk)

    def get_activations(self, h_2):
        return self.encode(h_2)


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder: encode h2 -> latent -> top-k -> decode latent -> reconstruct h2.
    Used when mode == "sae".
    """
    def __init__(self, input_dim, latent_dim=512, topk=None, jump_params=None):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.topk = topk

        gamma, beta = jump_params if jump_params else (1.0, 1.0)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim, bias=True),
            JumpReLU(gamma=gamma, beta=beta)
        )
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

        print(f"\n{'='*60}")
        print("Sparse Autoencoder (SAE):")
        print(f"  input_dim(h2) = {self.input_dim}")
        print(f"  latent_dim = {self.latent_dim}")
        print(f"  topk = {self.topk}")
        print(f"{'='*60}\n")

    def forward(self, h_2):
        z = self.encoder(h_2)
        z_sparse = topk_mask(z, self.topk)
        h_2_recon = self.decoder(z_sparse)
        return h_2_recon, z_sparse

    def encode(self, h_2):
        z = self.encoder(h_2)
        return topk_mask(z, self.topk)

    def get_activations(self, h_2):
        return self.encode(h_2)

# ---------------------------
# Training (single combined loss)
# ---------------------------

def train_model(model, dataloader, num_epochs, learning_rate, model_save_path, lambda_sparse=0.01, mode="transcoder"):
    """
    Generalized training function for both transcoders and SAEs.
    Single scalar loss:
      loss = MSE(recon, target) + lambda_sparse * mean(|z_sparse|)
    mode: "transcoder" -> target is h1 from batch
          "sae"       -> target is h2 (reconstruction)
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

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
            for batch in pbar:
                # batch is either (h2, h1) for transcoder or (h2, h1) as well but we might ignore h1 for SAE
                if mode == "transcoder":
                    h_2, h_1 = batch
                    target = h_1.to(device)
                else:  # sae
                    h_2, _ = batch
                    target = h_2.to(device)

                h_2 = h_2.to(device)
                # forward
                recon, z_sparse = model(h_2)

                # losses
                recon_loss = criterion(recon, target)
                sparse_loss = lambda_sparse * z_sparse.abs().mean()
                loss = recon_loss + sparse_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_sparse += sparse_loss.item()
                num_batches += 1

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "sparse": f"{sparse_loss.item():.6f}"
                })

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = total_loss / max(1, num_batches)
        avg_recon = total_recon / max(1, num_batches)
        avg_sparse = total_sparse / max(1, num_batches)

        print(f"\nEpoch [{epoch+1}/{num_epochs}]  Avg Loss: {avg_loss:.6f}  Recon: {avg_recon:.6f}  Sparse: {avg_sparse:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = model_save_path.replace('.pt', '_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'mode': mode,
                    'input_dim': getattr(model, 'input_dim', None),
                    'output_dim': getattr(model, 'output_dim', None),
                    'latent_dim': getattr(model, 'latent_dim', None),
                    'topk': getattr(model, 'topk', None)
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
            'mode': mode,
            'input_dim': getattr(model, 'input_dim', None),
            'output_dim': getattr(model, 'output_dim', None),
            'latent_dim': getattr(model, 'latent_dim', None),
            'topk': getattr(model, 'topk', None)
        }
    }, model_save_path)

    print(f"\n✓ Final model saved to {model_save_path}")
    print(f"✓ Best model saved to {best_model_path}")

    return model

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # Load activations: we expect activations_xyz.npz with arrays "x", "y", "z" (z unused)
    loaded = np.load("activations_xyz.npz")
    h_2 = loaded["x"]  # base model activations (e.g. CLIP 768-d)
    h_1 = loaded["y"]  # target hidden activations (e.g. 256-d)
    # h_0 = loaded.get("z")  # unused

    print(f"Loaded activations:")
    print(f"  h_2 shape: {h_2.shape}")
    print(f"  h_1 shape: {h_1.shape}")

    # Convert to tensors
    h_2_tensor = torch.tensor(h_2, dtype=torch.float32)
    h_1_tensor = torch.tensor(h_1, dtype=torch.float32)

    # Dataset and loader
    # We pack both into dataset so code is uniform: (h2, h1)
    dataset = TensorDataset(h_2_tensor, h_1_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=config["general"].get("batch_size", 32),
        shuffle=True,
        num_workers=config["general"].get("num_workers", 0)
    )

    # Model hyperparams (config overrides)
    latent_dim = 2048
    topk = 256             # None to disable top-k
    num_epochs = config["general"].get("num_epochs", 100)
    learning_rate = config["general"].get("learning_rate", 1e-4)
    lambda_sparse = config["general"].get("lambda_sparse", 0.01)
    jump_params = (1.0,1.0)

    # Create model according to mode
    os.makedirs("./models", exist_ok=True)
    if mode == "transcoder":
        model = SimpleTranscoder(
            input_dim=h_2_tensor.shape[1],
            output_dim=h_1_tensor.shape[1],
            latent_dim=latent_dim,
            topk=topk,
            jump_params=jump_params
        ).to(device)
        model_save_path = "./models/simple_transcoder.pt"
    elif mode == "sae":
        model = SparseAutoencoder(
            input_dim=h_2_tensor.shape[1],
            latent_dim=latent_dim,
            topk=topk,
            jump_params=jump_params
        ).to(device)
        model_save_path = "./models/sparse_autoencoder.pt"
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'transcoder' or 'sae' in config['model']['mode'].")

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on: {device}\n")

    model = train_model(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        model_save_path=model_save_path,
        lambda_sparse=lambda_sparse,
        mode=mode
    )

    print("\nTraining complete.")

"""
Requirements:
    pip install torch torchvision matplotlib numpy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

# ─────────────────────────────────────────────
#  HYPERPARAMETERS
# ─────────────────────────────────────────────
LATENT_DIM   = 100      
IMAGE_SIZE   = 28        
CHANNELS     = 1        
BATCH_SIZE   = 128
NUM_EPOCHS   = 30        
LR           = 0.0002 
BETA1        = 0.5     
BETA2        = 0.999  
NGF          = 64        
NDF          = 64        
SAMPLE_EVERY = 5         
OUTPUT_DIR   = "dcgan_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Project and reshape: z → (NGF*4, 7, 7)
        self.project = nn.Sequential(
            nn.Linear(LATENT_DIM, NGF * 4 * 7 * 7, bias=False),
            nn.BatchNorm1d(NGF * 4 * 7 * 7),
            nn.ReLU(True),
        )

        # Upsample: 7×7 → 14×14 → 28×28
        self.main = nn.Sequential(
            # Block 1: (NGF*4, 7, 7) → (NGF*2, 14, 14)
            nn.ConvTranspose2d(NGF * 4, NGF * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),

            # Block 2: (NGF*2, 14, 14) → (NGF, 28, 28)
            nn.ConvTranspose2d(NGF * 2, NGF,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),

            # Output: (NGF, 28, 28) → (1, 28, 28)
            nn.ConvTranspose2d(NGF, CHANNELS,
                               kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),   # Output in [-1, 1]
        )

    def forward(self, z):
        x = self.project(z)
        x = x.view(x.size(0), NGF * 4, 7, 7)
        return self.main(x)



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # Block 1: (1, 28, 28) → (NDF, 14, 14)
            # No BatchNorm on first layer (per DCGAN paper)
            nn.Conv2d(CHANNELS, NDF,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: (NDF, 14, 14) → (NDF*2, 7, 7)
            nn.Conv2d(NDF, NDF * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: (NDF*2, 7, 7) → (NDF*4, 3, 3)
            nn.Conv2d(NDF * 2, NDF * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: (NDF*4, 3, 3) → (1, 1, 1)
            nn.Conv2d(NDF * 4, 1,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, img):
        return self.main(img).view(-1)


# ─────────────────────────────────────────────
#  DATASET & DATALOADER
# ─────────────────────────────────────────────
def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalise to [-1, 1]
    ])
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=2, pin_memory=True)


# ─────────────────────────────────────────────
#  SAVE SAMPLE GRID
# ─────────────────────────────────────────────
def save_samples(generator, fixed_noise, epoch, path):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise).cpu()
    generator.train()

    grid = make_grid(fake, nrow=5, normalize=True, value_range=(-1, 1))
    img_np = grid.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(img_np.squeeze(), cmap="gray")
    ax.axis("off")
    ax.set_title(f"Generated Digits — Epoch {epoch}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved sample → {path}")


# ─────────────────────────────────────────────
#  PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_losses(g_losses, d_losses):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(g_losses, label="Generator Loss", linewidth=1.5)
    ax.plot(d_losses, label="Discriminator Loss", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("DCGAN Training Losses")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved → {path}")


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def train():
    dataloader = get_dataloader()

    # Build models
    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    print("\n── Generator ──")
    print(G)
    print("\n── Discriminator ──")
    print(D)

    # Loss & Optimizers
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    # Fixed noise for consistent sample visualisation across epochs
    fixed_noise = torch.randn(25, LATENT_DIM, device=device)

    g_losses, d_losses = [], []

    print(f"\nStarting training for {NUM_EPOCHS} epochs …\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        G.train(); D.train()

        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch = real_imgs.size(0)

            real_labels = torch.ones(batch, device=device)
            fake_labels = torch.zeros(batch, device=device)

            # ── Train Discriminator ──────────────────────────────────────
            # Goal: maximise log(D(x)) + log(1 - D(G(z)))
            opt_D.zero_grad()

            # Real images → D should output ~1
            out_real = D(real_imgs)
            loss_D_real = criterion(out_real, real_labels)

            # Fake images → D should output ~0
            noise = torch.randn(batch, LATENT_DIM, device=device)
            fake_imgs = G(noise)
            out_fake = D(fake_imgs.detach())   # detach so G grads aren't computed
            loss_D_fake = criterion(out_fake, fake_labels)

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            opt_D.step()

            # ── Train Generator ──────────────────────────────────────────
            # Goal: maximise log(D(G(z)))  ←→  minimise log(1 - D(G(z)))
            opt_G.zero_grad()

            out_fake_for_G = D(fake_imgs)
            # Generator wants D to think fakes are real
            loss_G = criterion(out_fake_for_G, real_labels)

            loss_G.backward()
            opt_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

        # ── Epoch summary ────────────────────────────────────────────────
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
              f"Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}")

        if epoch % SAMPLE_EVERY == 0 or epoch == NUM_EPOCHS:
            path = os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch:03d}.png")
            save_samples(G, fixed_noise, epoch, path)

    # ── Save final model weights ─────────────────────────────────────────
    torch.save(G.state_dict(), os.path.join(OUTPUT_DIR, "generator.pth"))
    torch.save(D.state_dict(), os.path.join(OUTPUT_DIR, "discriminator.pth"))
    print("\nModel weights saved.")

    # ── Final samples (≥10 digits) ───────────────────────────────────────
    G.eval()
    with torch.no_grad():
        noise = torch.randn(25, LATENT_DIM, device=device)
        final_imgs = G(noise).cpu()

    grid = make_grid(final_imgs, nrow=5, normalize=True, value_range=(-1, 1))
    save_image(grid, os.path.join(OUTPUT_DIR, "final_generated_digits.png"))
    print("Final 25-digit grid saved → dcgan_output/final_generated_digits.png")

    # ── Plot loss curves ─────────────────────────────────────────────────
    plot_losses(g_losses, d_losses)

    return G, D


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    G, D = train()
    print("\nTraining complete! Check the 'dcgan_output/' folder for results.")

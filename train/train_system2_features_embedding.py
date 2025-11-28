import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_split import load_or_create_split
from data.datasets import TripletDataset
from utils.triplet_loss import TripletLoss
from train.system2_patch_training import train_triplet_epoch, eval_triplet_epoch
from feature_extractor.patch_models import FeatureEmbeddingNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES_NPZ = "/fhome/maed02/proj_repte3/results/system2_resnet_features/patch_features_resnet152.npz"
SAVE_DIR = "/fhome/maed02/proj_repte3/results/system2_features_embedding"
os.makedirs(SAVE_DIR, exist_ok=True)


def train_features_embedding():
    print("Carregant features pre-computats...")
    data = np.load(FEATURES_NPZ)
    X_feat = data["X_feat"]    # (N,2048)
    y = data["y"]

    # ================================================================
    # 1) LOAD OR CREATE GLOBAL SPLIT (TRAIN/VAL)
    # ================================================================
    train_idx, val_idx, _ = load_or_create_split(len(X_feat), y)

    train_ds = TripletDataset(X_feat[train_idx], y[train_idx])
    val_ds   = TripletDataset(X_feat[val_idx],   y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)

    # ================================================================
    # 2) MODEL
    # ================================================================
    model = FeatureEmbeddingNet(
        in_dim=2048,
        proj_hidden_dim=256,
        proj_out_dim=128,
        normalize=False
    ).to(DEVICE)

    loss_fn = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Early stopping
    best_val_loss = float("inf")
    patience = 5
    patience_count = 0

    n_epochs = 50
    print("\n===== ENTRENANT EMBEDDING (FEATURES) =====")

    for epoch in range(n_epochs):

        train_loss = train_triplet_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss   = eval_triplet_epoch(model, val_loader, loss_fn, DEVICE)

        print(f"[Embedding] Epoch {epoch+1}/{n_epochs} | "
              f"Train={train_loss:.4f}  Val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_embedding_features.pth"))
            print("   ✔ Nou millor model guardat!")
        else:
            patience_count += 1
            print(f"   ⏳ Patience {patience_count}/{patience}")

        if patience_count >= patience:
            print("🛑 Early stopping activat.")
            break

    print(f"Millor Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_features_embedding()

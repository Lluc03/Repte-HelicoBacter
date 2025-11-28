import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_split import load_or_create_split
from data.datasets import Standard_Dataset
from train.system2_patch_training import (
    train_classifier_epoch,
    eval_classifier_epoch,
    compute_class_weights
)
from feature_extractor.patch_models import (
    FeatureEmbeddingNet,
    PatchClassifierHead,
    PatchEmbeddingAndClassifier
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES_NPZ = "/fhome/maed02/proj_repte3/results/system2_resnet_features/patch_features_resnet152.npz"
EMB_PTH = "/fhome/maed02/proj_repte3/results/system2_features_embedding/best_embedding_features.pth"
SAVE_DIR = "/fhome/maed02/proj_repte3/results/system2_features_classifier"
os.makedirs(SAVE_DIR, exist_ok=True)


def train_features_classifier():

    print("Carregant features pre-computats...")
    data = np.load(FEATURES_NPZ)
    X_feat = data["X_feat"]
    y = data["y"]

    # ================================================================
    # 1) MATEIX SPLIT QUE L’EMBEDDING
    # ================================================================
    train_idx, val_idx, _ = load_or_create_split(len(X_feat), y)

    train_ds = Standard_Dataset(X_feat[train_idx], y[train_idx])
    val_ds   = Standard_Dataset(X_feat[val_idx],   y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)

    # ================================================================
    # 2) CLASS WEIGHTS
    # ================================================================
    class_weights = compute_class_weights(torch.tensor(y[train_idx])).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ================================================================
    # 3) LOAD EMBEDDING NETWORK (FROZEN)
    # ================================================================
    emb_net = FeatureEmbeddingNet(
        in_dim=2048,
        proj_hidden_dim=256,
        proj_out_dim=128,
        normalize=False
    ).to(DEVICE)

    emb_net.load_state_dict(torch.load(EMB_PTH, map_location=DEVICE))
    for p in emb_net.parameters():
        p.requires_grad = False

    # ================================================================
    # 4) CLASSIFIER HEAD (trainable)
    # ================================================================
    cls_head = PatchClassifierHead(emb_dim=128, hidden_dim=64, n_classes=2)
    model = PatchEmbeddingAndClassifier(emb_net, cls_head).to(DEVICE)

    optimizer = torch.optim.Adam(cls_head.parameters(), lr=1e-3)

    # ================================================================
    # 5) TRAINING + EARLY STOPPING
    # ================================================================
    best_val_loss = float("inf")
    patience = 5
    patience_count = 0
    n_epochs = 50

    print("\n===== ENTRENANT CLASSIFIER (FEATURES) =====")

    for epoch in range(n_epochs):

        train_loss, train_acc = train_classifier_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_acc     = eval_classifier_epoch(model, val_loader, loss_fn, DEVICE)

        print(f"[Classifier] Epoch {epoch+1}/{n_epochs} | "
              f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc*100:.2f}%  "
              f"ValLoss={val_loss:.4f}, ValAcc={val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, "best_classifier_features.pth"))
            print("   ✔ Nou millor model guardat!")
        else:
            patience_count += 1
            print(f"   ⏳ Patience {patience_count}/{patience}")

        if patience_count >= patience:
            print("🛑 Early stopping activat.")
            break

    print(f"Millor Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_features_classifier()

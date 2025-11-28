# ================================================================
# SYSTEM2 - FEATURES-BASED K-FOLD ROC CURVE (PATCH-LEVEL)
# ================================================================
# Autor: Lluc Verdaguer Macias
#
# Pipeline:
#   1. Carrega features precomputats (ResNet152 → 2048D)
#   2. Carrega embeddings (FeatureEmbeddingNet)
#   3. Carrega classifier (PatchClassifierHead)
#   4. K-Fold ROC estratificat per pacient (StratifiedGroupKFold)
#   5. Guarda totes les ROC per fold + ROC mitjana
#
# Notes:
#   - NO es fan servir imatges; només features (.npz)
#   - Totalment coherent amb System2 (features)
# ================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc

import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_extractor.patch_models import (
    FeatureEmbeddingNet,
    PatchClassifierHead,
    PatchEmbeddingAndClassifier
)

# ================================================================
# PATHS
# ================================================================
FEATURES_NPZ = "/fhome/maed02/proj_repte3/results/system2_resnet_features/patch_features_resnet152.npz"
EMB_PTH = "/fhome/maed02/proj_repte3/results/system2_features_embedding/best_embedding_features.pth"
CLS_PTH = "/fhome/maed02/proj_repte3/results/system2_features_classifier/best_classifier_features.pth"

OUTPUT_DIR = "/fhome/maed02/proj_repte3/results/system2_features_kfold"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# LOAD FEATURE NPZ (X_feat, y, groups)
# ================================================================
def load_npz_features():
    data = np.load(FEATURES_NPZ)
    X_feat = data["X_feat"]   # (N,2048)
    y = data["y"]             # (N,)
    groups = data["groups"]   # (N,)
    print("Carregat NPZ:", FEATURES_NPZ)
    return X_feat, y, groups


# ================================================================
# LOAD SYSTEM2 MODEL (FEATURES-BASED)
# ================================================================
def load_system2_model():

    # Feature Embedding
    emb_net = FeatureEmbeddingNet(
        in_dim=2048,
        proj_hidden_dim=256,
        proj_out_dim=128,
        normalize=False
    ).to(DEVICE)

    emb_net.load_state_dict(torch.load(EMB_PTH, map_location=DEVICE))
    for p in emb_net.parameters():
        p.requires_grad = False

    # Classifier
    cls_head = PatchClassifierHead(
        emb_dim=128,
        hidden_dim=64,
        n_classes=2
    )

    model = PatchEmbeddingAndClassifier(emb_net, cls_head).to(DEVICE)
    model.load_state_dict(torch.load(CLS_PTH, map_location=DEVICE))

    model.eval()
    print("Model System2 carregat correctament.")
    return model


# ================================================================
# INFERÈNCIA SOBRE FEATURES
# ================================================================
def predict_scores(model, X_test, batch_size=256):
    ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    scores = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(probs.cpu().numpy())

    return np.array(scores)


# ================================================================
# ROC PER FOLD
# ================================================================
def evaluate_fold(model, X_test, y_test, fold_id):

    scores = predict_scores(model, X_test)

    fpr, tpr, thr = roc_curve(y_test, scores)
    auc_val = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Fold {fold_id}")
    plt.legend()
    plt.grid(True)

    fold_path = os.path.join(OUTPUT_DIR, f"roc_fold_{fold_id}.png")
    plt.savefig(fold_path)
    plt.close()

    print(f"✔ ROC guardada: {fold_path}")

    return auc_val, fpr, tpr


# ================================================================
# ROC MITJANA
# ================================================================
def plot_mean_roc(all_fprs, all_tprs, aucs):

    mean_fpr = np.linspace(0, 1, 200)
    interp_tprs = []

    plt.figure(figsize=(7,6))

    # Corbes difuminades
    for fpr, tpr in zip(all_fprs, all_tprs):
        plt.plot(fpr, tpr, color="gray", alpha=0.3)
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        interp_tprs.append(interp)

    # Mean
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    plt.plot(mean_fpr, mean_tpr, color="blue", lw=2,
             label=f"Mean ROC (AUC = {mean_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("System2 FEATURES — Patch-Level Mean ROC")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, "roc_mean_system2.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✔ ROC mitjana guardada: {save_path}")


# ================================================================
# K-FOLD ROC
# ================================================================
def kfold_patch_evaluation(model, X, y, groups, k=5):

    cv = StratifiedGroupKFold(n_splits=k, shuffle=False)

    aucs = []
    all_fprs = []
    all_tprs = []

    fold = 1
    for train_idx, test_idx in cv.split(X, y, groups):

        print(f"\n=========== FOLD {fold}/{k} ===========")

        X_test = X[test_idx]
        y_test = y[test_idx]

        auc_val, fpr, tpr = evaluate_fold(model, X_test, y_test, fold)
        aucs.append(auc_val)
        all_fprs.append(fpr)
        all_tprs.append(tpr)

        fold += 1

    print("\n========= RESULTATS FINALS =========")
    print("AUCs per fold:", aucs)
    print("AUC mitjana:", np.mean(aucs))

    plot_mean_roc(all_fprs, all_tprs, aucs)


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":

    print("Carregant NPZ...")
    X, y, groups = load_npz_features()

    print(f"Total patches: {len(X)}")
    print(f"Classe 0 = {(y==0).sum()} | Classe 1 = {(y==1).sum()}")
    print(f"Pacients únics: {len(np.unique(groups))}")

    print("Carregant model...")
    model = load_system2_model()

    print("Iniciant ROC K-Fold...")
    kfold_patch_evaluation(model, X, y, groups, k=5)

    print("\n🎉 ROC COMPLETADA CORRECTAMENT")

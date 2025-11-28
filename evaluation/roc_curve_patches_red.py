# ================================================================
# K-FOLD ROC CURVE for PATCHES using a pre-trained AutoEncoder
# ================================================================
# Autor: Lluc Verdaguer Macias
# Explicació:
#   - Es carreguen tots els patches Annotated (presence -1 = sa, 1 = bacteria)
#   - Es calcula l'error de reconstrucció amb l'AutoEncoder entrenat
#   - L'error es defineix sobre el canal "vermell" fent servir HSV (Hue)
#   - Es fa un K-Fold cross-validation PER PATCHES, estratificat per pacient
#     (StratifiedGroupKFold, balanced folds, shuffle=False)
#   - Es genera un AUC i un threshold per cada fold
#   - Es calcula AUC mitjà + Interval de Confiança 95%
#   - Es mostra la mitjana final i es ploteja la ROC
# ================================================================

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ================================================================
# PATHS
# ================================================================
DATASET_DIR = "/fhome/maed/HelicoDataSet/CrossValidation"
ANNOTATED_PATH = os.path.join(DATASET_DIR, "Annotated")
EXCEL = "/fhome/maed/HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"

AUTOENCODER_WEIGHTS = "/fhome/maed02/proj_repte3/results/autoencoder_config1.pth"

# Output directory + file for ROC plots
OUTPUT_DIR = "/fhome/maed02/proj_repte3/results/kfold"
ROC_FILENAME = "roc_curve_patches_S_channel_HSV.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# LOAD ANNOTATED PATCHES
# ================================================================
def LoadAnnotated(base_path, annot_df, size=(256, 256)):
    """
    Carrega patches Annotated. Mapeig correcte:
      Presence = -1 → 0 (sa)
      Presence =  1 → 1 (bacteria)
      Presence =  0 → ignorar

    Retorna:
      X: imatges (N, 3, H, W)
      y: etiquetes (0/1)
      groups: ID de pacient per a StratifiedGroupKFold
    """
    images = []
    labels = []
    groups = []

    folders = glob.glob(os.path.join(base_path, "*"))

    for folder in folders:
        folder_name = os.path.basename(folder)
        parts = folder_name.split("_")
        pat_id = parts[0]
        section_id = int(parts[1])

        image_paths = sorted(glob.glob(os.path.join(folder, "*.png")))

        for img_path in image_paths:
            win_str = os.path.splitext(os.path.basename(img_path))[0]
            if not win_str.isdigit():
                continue

            win_id = int(win_str)

            row = annot_df[
                (annot_df["Pat_ID"] == pat_id) &
                (annot_df["Section_ID"] == section_id) &
                (annot_df["Window_ID"] == win_id)
            ]

            if len(row) == 0:
                continue

            presence = int(row["Presence"].values[0])

            if presence == 0:
                continue  # ignorem desconeguts

            label = 0 if presence == -1 else 1

            img = Image.open(img_path).convert("RGB").resize(size)
            arr = np.array(img).transpose(2, 0, 1) / 255.0

            images.append(arr)
            labels.append(label)
            groups.append(pat_id)  # agrupació per pacient

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(groups)
    )

# ================================================================
# FOLD EVALUATION: reconstruction error + ROC + PLOT
# ================================================================
def evaluate_fold(model, X_test, y_test, fold_id, batch_size=16):
    """
    Calcula l'error de reconstrucció en el canal S de HSV i retorna:
      - AUC del fold
      - threshold òptim (Youden J)
      - scores (error per patch)
      - fpr, tpr per la ROC
    """
    loader = DataLoader(torch.tensor(X_test), batch_size=batch_size, shuffle=False)

    scores = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)  # (B,3,H,W) RGB [0,1]
            recon = model(batch)

            # Convertim a HSV
            batch_hsv = rgb_to_hsv_torch(batch)       # (B,3,H,W)
            recon_hsv = rgb_to_hsv_torch(recon)

            # ---------- ERROR EN S (SATURACIÓN) ----------
            diff_s = (recon_hsv[:, 1] - batch_hsv[:, 1]) ** 2  # canal S
            err = diff_s.mean(dim=(1, 2))
            # ----------------------------------------------

            scores.extend(err.cpu().numpy())

    scores = np.array(scores)

    # ROC curve
    fpr, tpr, thr = roc_curve(y_test, scores)
    auc_val = auc(fpr, tpr)

    # Plot ROC per fold
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Fold {fold_id} (Canal S)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    fold_path = os.path.join(OUTPUT_DIR, f"roc_fold_s_{fold_id}.png")
    plt.savefig(fold_path)
    plt.close()

    print(f"✔ ROC guardada: {fold_path}")

    # Threshold òptim
    J = tpr - fpr
    thr_opt = thr[np.argmax(J)]

    return auc_val, thr_opt, scores, fpr, tpr


# ================================================================
# PLOT ROC MITJANA (AGREGADA)
# ================================================================
def plot_mean_roc(all_fprs, all_tprs, aucs):
    plt.figure(figsize=(6, 5))

    for fpr, tpr in zip(all_fprs, all_tprs):
        plt.plot(fpr, tpr, color="gray", alpha=0.3)

    mean_fpr = np.linspace(0, 1, 200)
    mean_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fprs, all_tprs)]
    mean_tpr = np.mean(mean_tprs, axis=0)
    mean_auc = np.mean(aucs)

    plt.plot(mean_fpr, mean_tpr, color="blue", linewidth=2,
             label=f"Mean AUC = {mean_auc:.3f}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curve - Patch KFold (S channel HSV)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, ROC_FILENAME)
    plt.savefig(save_path)
    plt.close()

    print(f"✔ ROC mitjana guardada a: {save_path}")

# ================================================================
# K-FOLD CROSS-VALIDATION per PATCHES (StratifiedGroupKFold)
# ================================================================
def kfold_patch_evaluation(model, X, y, groups, k=5):
    """
    K-Fold estratificat per pacient:
      - StratifiedGroupKFold(n_splits=k, shuffle=False)
      - Balanced folds (estratificació per y)
      - groups = Pat_ID
    """
    cv = StratifiedGroupKFold(n_splits=k, shuffle=False)

    aucs = []
    thresholds = []
    all_fprs = []
    all_tprs = []

    fold = 1

    for train_idx, test_idx in cv.split(X, y, groups):
        print(f"\n================ FOLD {fold}/{k} ================")

        X_test = X[test_idx]
        y_test = y[test_idx]

        auc_val, thr_opt, _, fpr, tpr = evaluate_fold(
            model, X_test, y_test, fold_id=fold
        )

        aucs.append(auc_val)
        thresholds.append(thr_opt)
        all_fprs.append(fpr)
        all_tprs.append(tpr)

        print(f"AUC Fold {fold}: {auc_val:.4f}")
        print(f"Threshold Fold {fold}: {thr_opt:.6f}")

        fold += 1

    # Estadística bàsica + IC 95%
    aucs = np.array(aucs)
    thr_arr = np.array(thresholds)
    mean_auc = aucs.mean()
    std_auc = aucs.std(ddof=1) if len(aucs) > 1 else 0.0
    ci95_low = mean_auc - 1.96 * std_auc / np.sqrt(len(aucs)) if len(aucs) > 1 else mean_auc
    ci95_high = mean_auc + 1.96 * std_auc / np.sqrt(len(aucs)) if len(aucs) > 1 else mean_auc

    print("\n============== RESULTATS FINALS ==============")
    print(f"AUC mitjà        = {mean_auc:.4f}")
    print(f"Std(AUC)         = {std_auc:.4f}")
    print(f"IC 95% AUC       = [{ci95_low:.4f}, {ci95_high:.4f}]")
    print(f"Threshold mitjà  = {thr_arr.mean():.6f}")

    # ROC mitjana
    plot_mean_roc(all_fprs, all_tprs, aucs)

    return aucs, thresholds, (ci95_low, ci95_high)
    
    
# ================================================================
# Funció auxiliar
# ================================================================
    
def rgb_to_hsv_torch(img):
    """
    img: tensor (B,3,H,W) en rang [0,1]
    Retorna HSV tensor (B,3,H,W)
    """
    r, g, b = img[:, 0], img[:, 1], img[:, 2]

    maxc, _ = torch.max(img, dim=1)
    minc, _ = torch.min(img, dim=1)
    v = maxc
    deltac = maxc - minc

    # Saturation
    s = deltac / (maxc + 1e-6)

    # Hue
    rc = (maxc - r) / (deltac + 1e-6)
    gc = (maxc - g) / (deltac + 1e-6)
    bc = (maxc - b) / (deltac + 1e-6)

    h = torch.zeros_like(maxc)

    mask = (maxc == r)
    h[mask] = (bc - gc)[mask]

    mask = (maxc == g)
    h[mask] = 2.0 + (rc - bc)[mask]

    mask = (maxc == b)
    h[mask] = 4.0 + (gc - rc)[mask]

    h = (h / 6.0) % 1.0  # normalització 0–1

    return torch.stack([h, s, v], dim=1)

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("Carregant Excel d'anotacions...")
    annot_df = pd.read_excel(EXCEL)

    print("Carregant patches Annotated...")
    X, y, groups = LoadAnnotated(ANNOTATED_PATH, annot_df)

    print(f"Total patches carregats: {len(X)}")
    print(f"Sanos: {(y == 0).sum()}, Bacteria: {(y == 1).sum()}")
    print(f"N pacients (únics) en dades: {len(np.unique(groups))}")

    # ============================================================
    # CARREGAR AUTOENCODER ENTRENAT
    # ============================================================
    print("Carregant AutoEncoder...")

    from models.AEmodels import AutoEncoderCNN

    inputmodule_paramsEnc = {"num_input_channels": 3}

    # Configuració igual al training
    def AEConfigs(Config):
        # Configuració de la xarxa de l'AutoEncoder
        net_paramsEnc = {}
        net_paramsDec = {}
        inputmodule_paramsDec = {}
    
        if Config == '1':
            net_paramsEnc['block_configs'] = [[32, 32], [64, 64]]
            net_paramsEnc['stride'] = [[1, 2], [1, 2]]
            net_paramsDec['block_configs'] = [[64, 32], [32, inputmodule_paramsEnc['num_input_channels']]]
            net_paramsDec['stride'] = net_paramsEnc['stride']
            inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]
    
        elif Config == '2':
            net_paramsEnc['block_configs'] = [[32], [64], [128], [256]]
            net_paramsEnc['stride'] = [[2], [2], [2], [2]]
            net_paramsDec['block_configs'] = [[128], [64], [32], [inputmodule_paramsEnc['num_input_channels']]]
            net_paramsDec['stride'] = net_paramsEnc['stride']
            inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]
    
        elif Config == '3':
            net_paramsEnc['block_configs'] = [[32], [64], [64]]
            net_paramsEnc['stride'] = [[1], [2], [2]]
            net_paramsDec['block_configs'] = [[64], [32], [inputmodule_paramsEnc['num_input_channels']]]
            net_paramsDec['stride'] = net_paramsEnc['stride']
            inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]
    
        return net_paramsEnc, net_paramsDec, inputmodule_paramsDec

    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs('1')

    model = AutoEncoderCNN(
        inputmodule_paramsEnc,
        net_paramsEnc,
        inputmodule_paramsDec,
        net_paramsDec
    )

    model.load_state_dict(torch.load(AUTOENCODER_WEIGHTS, map_location=DEVICE))
    model = model.to(DEVICE)

    print("\nIniciant K-Fold ROC per Patches (StratifiedGroupKFold)...")
    aucs, thresholds, ci95 = kfold_patch_evaluation(model, X, y, groups, k=5)

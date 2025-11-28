# =====================================================================
# ROC A NIVELL DE PACIENT utilitzant AutoEncoder + HSV-H (vermell)
# Dataset: NOMÉS CROPPED, EN MODE STREAMING (sense OOM)
# =====================================================================

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

# Afegim la carpeta pare al path per poder importar models.AEmodels
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.AEmodels import AutoEncoderCNN  # :contentReference[oaicite:0]{index=0}

# -------------------------------------------------------------
# 0. CONFIG
# -------------------------------------------------------------
CROPPED_DIR = "/fhome/maed/HelicoDataSet/CrossValidation/Cropped"
PATIENT_CSV = "/fhome/maed/HelicoDataSet/PatientDiagnosis.csv"

MODEL_PATH = "/fhome/maed02/proj_repte3/results/autoencoder_config1.pth"

OUTPUT_DIR = "/fhome/maed02/proj_repte3/results/patient_level"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# nombre màxim de patches processats a la vegada a GPU
BATCH_GPU = 32
IMG_SIZE = (256, 256)

# -------------------------------------------------------------
# 1. RGB -> HSV (PyTorch, versió compatible)
# -------------------------------------------------------------
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

    s = deltac / (maxc + 1e-6)

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

    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v], dim=1)

# -------------------------------------------------------------
# 2. LOAD MODEL (Autoencoder) – mateixes configs que training
# -------------------------------------------------------------
def AEConfigs(Config):
    net_paramsEnc = {}
    net_paramsDec = {}
    inputmodule_paramsDec = {}
    inputmodule_paramsEnc = {"num_input_channels": 3}

    if Config == '1':
        net_paramsEnc['block_configs'] = [[32, 32], [64, 64]]
        net_paramsEnc['stride'] = [[1, 2], [1, 2]]
        net_paramsDec['block_configs'] = [[64, 32], [32, inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    return inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec

inputmodule_paramsEnc, netE, inputmodule_paramsDec, netD = AEConfigs("1")

model = AutoEncoderCNN(inputmodule_paramsEnc, netE, inputmodule_paramsDec, netD).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------------------------------------
# 3. LOAD PATIENT DIAGNOSIS CSV
# -------------------------------------------------------------
df_diag = pd.read_csv(PATIENT_CSV)
df_diag["CODI"] = df_diag["CODI"].str.strip()
df_diag["DENSITAT"] = df_diag["DENSITAT"].str.upper().str.strip()

df_diag["label"] = df_diag["DENSITAT"].replace({
    "NEGATIVA": 0,
    "BAIXA": 1,
    "ALTA": 1
}).astype(int)


patient_label_map = dict(zip(df_diag["CODI"], df_diag["label"]))

# -------------------------------------------------------------
# 4. MAPEAR FOLDERS DE CROPPED A PACIENTS
# -------------------------------------------------------------
# Cada carpeta dins de CROPPED és d'una WSI/section, el PatID és la part abans del "_"
all_folders = [
    f for f in sorted(os.listdir(CROPPED_DIR))
    if os.path.isdir(os.path.join(CROPPED_DIR, f))
]

pat_to_folders = {}
for folder in all_folders:
    pat_id = folder.split("_")[0]
    pat_to_folders.setdefault(pat_id, []).append(os.path.join(CROPPED_DIR, folder))

print(f"Total carpetes (WSI/sections): {len(all_folders)}")
print(f"Total pacients trobats al CROPPED: {len(pat_to_folders)}")

# -------------------------------------------------------------
# 5. FUNCIONS AUXILIARS: PROCESSAR 1 PACIENT EN STREAMING
# -------------------------------------------------------------
def process_patient(pat_id, folders, model):
    """
    Calcula tots els errors HSV-H dels patches d'un pacient, en streaming.
    Retorna: llista d'errors (un per patch).
    """
    patch_errors = []

    with torch.no_grad():
        for folder in folders:
            img_paths = sorted(glob.glob(os.path.join(folder, "*.png")))
            if len(img_paths) == 0:
                continue

            # Bucle per batches per no carregar massa a RAM/GPU
            for i in range(0, len(img_paths), BATCH_GPU):
                batch_paths = img_paths[i : i + BATCH_GPU]
                imgs_np = []

                for pth in batch_paths:
                    img = Image.open(pth).convert("RGB").resize(IMG_SIZE)
                    arr = np.array(img).transpose(2, 0, 1) / 255.0
                    imgs_np.append(arr)

                batch = torch.from_numpy(np.stack(imgs_np)).float().to(DEVICE)

                recon = model(batch)
                batch_hsv = rgb_to_hsv_torch(batch)
                recon_hsv = rgb_to_hsv_torch(recon)

                diff_h = (recon_hsv[:, 0] - batch_hsv[:, 0]) ** 2
                err = diff_h.mean(dim=(1, 2))

                patch_errors.extend(err.cpu().numpy())

                # alliberem memòria temporal
                del batch, recon, batch_hsv, recon_hsv, diff_h, err
                torch.cuda.empty_cache()

    return patch_errors

# -------------------------------------------------------------
# 6. RECÓRRER TOTS ELS PACIENTS I CALCULAR SCORE (p95)
# -------------------------------------------------------------
patient_ids = []
patient_scores = []
patient_labels = []

for pat_id, folders in pat_to_folders.items():
    print(f"Processant pacient {pat_id} amb {len(folders)} carpetes...")

    errs = process_patient(pat_id, folders, model)

    if len(errs) == 0:
        print(f"  [AVÍS] Pacient {pat_id} sense imatges, s'ignora.")
        continue

    score_p95 = np.percentile(errs, 95)

    # label real del pacient (si no és al CSV, assumim 0 = NEGATIU)
    label = patient_label_map.get(pat_id, 0)

    patient_ids.append(pat_id)
    patient_scores.append(score_p95)
    patient_labels.append(label)

patient_scores = np.array(patient_scores)
patient_labels = np.array(patient_labels)
patient_ids = np.array(patient_ids)

print(f"\nPacients amb dades vàlides: {len(patient_ids)}")

# -------------------------------------------------------------
# 7. STRATIFIED-GROUP K-FOLD (nivell pacient)
# -------------------------------------------------------------
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc

groups = patient_ids  # cada pacient és un grup

cv = StratifiedGroupKFold(n_splits=5, shuffle=False)

all_fprs, all_tprs, aucs = [], [], []
fold = 1

for train_idx, test_idx in cv.split(patient_scores, patient_labels, groups):
    y_test = patient_labels[test_idx]
    s_test = patient_scores[test_idx]

    fpr, tpr, thr = roc_curve(y_test, s_test)
    auc_val = auc(fpr, tpr)

    all_fprs.append(fpr)
    all_tprs.append(tpr)
    aucs.append(auc_val)

    print(f"Fold {fold} - AUC = {auc_val:.4f}")
    fold += 1

aucs = np.array(aucs)

# -------------------------------------------------------------
# 8. AUC MITJÀ + INTERVAL DE CONFIANÇA 95%
# -------------------------------------------------------------
mean_auc = aucs.mean()
std_auc = aucs.std(ddof=1) if len(aucs) > 1 else 0.0
if len(aucs) > 1:
    ci_low = mean_auc - 1.96 * std_auc / np.sqrt(len(aucs))
    ci_high = mean_auc + 1.96 * std_auc / np.sqrt(len(aucs))
else:
    ci_low = ci_high = mean_auc

print("\n=======================================")
print("RESULTATS ROC A NIVELL PACIENT (HSV-H)")
print("=======================================")
print(f"AUC mitjà = {mean_auc:.4f}")
print(f"IC95%     = [{ci_low:.4f}, {ci_high:.4f}]")

# -------------------------------------------------------------
# 9. PLOT ROC MITJANA
# -------------------------------------------------------------
mean_fpr = np.linspace(0, 1, 200)
interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fprs, all_tprs)]
mean_tpr = np.mean(interp_tprs, axis=0)

plt.figure(figsize=(6, 5))
for fpr, tpr in zip(all_fprs, all_tprs):
    plt.plot(fpr, tpr, color="gray", alpha=0.3)

plt.plot(mean_fpr, mean_tpr, color="red",
         label=f"AUC mitjà = {mean_auc:.3f}", linewidth=2)

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Pacient-Level (HSV-H anomaly score)")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "ROC_patient_level_HSVH_streaming.png"))
plt.close()

from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 10. CONFUSION MATRIX A NIVELL PACIENT
# ============================================================

# Threshold òptim Youden J (del ROC global)
fpr, tpr, thr = roc_curve(patient_labels, patient_scores)
J = tpr - fpr
best_thr = thr[np.argmax(J)]

print(f"\nThreshold òptim (Youden J): {best_thr:.6f}")

# Prediccions binàries
y_pred = (patient_scores >= best_thr).astype(int)
y_true = patient_labels

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix (nivell pacient):")
print(cm)

# Informe complet
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["NEGATIU", "POSITIU"]))

# També pots calcular recalls manualment:
recall_neg = cm[0,0] / (cm[0,0] + cm[0,1])
recall_pos = cm[1,1] / (cm[1,1] + cm[1,0])

print(f"\nRecall NEGATIU (HP-): {recall_neg:.4f}")
print(f"Recall POSITIU (HP+): {recall_pos:.4f}")

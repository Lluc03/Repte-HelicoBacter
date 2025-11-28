#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline per diagnosticar TOTS els pacients del dataset Cropped
1. Genera embeddings per TOTS els pacients
2. Entrena model amb CV
3. Prediu CADA pacient amb ensemble dels 5 models
"""

import os
import glob
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# ============================================================
# CONFIGURACIO
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
sys.path.append(MODELS_DIR)

from feature_extractor.feature_extractors import ResNetFeatureExtractor
from feature_extractor.patch_models import FeatureEmbeddingNet
from models.AttentionUnits import GatedAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32

# Paths
CROPPED_DIR = "/fhome/maed/HelicoDataSet/CrossValidation/Cropped"
PATIENT_CSV = "/fhome/maed/HelicoDataSet/PatientDiagnosis.csv"
MLP_MODEL_PATH = "/fhome/maed02/proj_repte3/results/system2_features_embedding/best_embedding_features.pth"

OUTPUT_DIR = "/fhome/maed02/proj_repte3/results/pipeline_ALL_patients"
EMB_DIR = os.path.join(OUTPUT_DIR, "embeddings")
MODELS_DIR_OUT = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

for d in [OUTPUT_DIR, EMB_DIR, MODELS_DIR_OUT, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Processant TOTS els pacients del dataset\n")

# ============================================================
# STEP 1: CARREGAR TOTS ELS PACIENTS
# ============================================================
print("="*70)
print("STEP 1: Carregant informacio de TOTS els pacients")
print("="*70)

# Carregar CSV amb diagnostics
df = pd.read_csv(PATIENT_CSV)
df["CODI"] = df["CODI"].astype(str).str.strip()
df["DENSITAT"] = df["DENSITAT"].astype(str).str.upper().str.strip()

df["label"] = df["DENSITAT"].replace({
    "NEGATIVA": 0,
    "BAIXA": 1,
    "ALTA": 1
}).astype(int)

df = df[df["label"].isin([0, 1])].reset_index(drop=True)

# Detectar TOTS els pacients que tenen carpetes a Cropped
all_folders = sorted([f for f in os.listdir(CROPPED_DIR) 
                      if os.path.isdir(os.path.join(CROPPED_DIR, f))])

# Extreure IDs unics de pacients (B22-129_0, B22-129_1 -> B22-129)
patient_ids_in_cropped = set()
for folder in all_folders:
    # Agafar ID base (abans del ultim "_")
    if "_" in folder:
        patient_id = "_".join(folder.split("_")[:-1])
    else:
        patient_id = folder
    patient_ids_in_cropped.add(patient_id)

patient_ids_in_cropped = sorted(list(patient_ids_in_cropped))

print(f"Pacients trobats a Cropped: {len(patient_ids_in_cropped)}")

# Filtrar CSV per tenir nomes pacients que estan a Cropped
df_all = df[df["CODI"].isin(patient_ids_in_cropped)].reset_index(drop=True)

print(f"Pacients amb diagnostic: {len(df_all)}")
print(f"  Positius (BAIXA/ALTA): {sum(df_all['label'])}")
print(f"  Negatius (NEGATIVA): {len(df_all) - sum(df_all['label'])}\n")

# ============================================================
# STEP 2: GENERAR EMBEDDINGS PER TOTS ELS PACIENTS
# ============================================================
print("="*70)
print("STEP 2: Generant embeddings per TOTS els pacients")
print("="*70)

resnet = ResNetFeatureExtractor(pretrained=True).to(DEVICE)
resnet.eval()

mlp = FeatureEmbeddingNet(
    in_dim=2048,
    proj_hidden_dim=256,
    proj_out_dim=128
).to(DEVICE)
mlp.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=DEVICE))
mlp.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CroppedDataset(Dataset):
    def __init__(self, img_paths):
        self.paths = img_paths
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        x = transform(img)
        return x, path

# Recopilar TOTES les imatges
all_img_paths = []
for folder in tqdm(all_folders, desc="Escanejant carpetes"):
    folder_path = os.path.join(CROPPED_DIR, folder)
    imgs = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    all_img_paths.extend(imgs)

print(f"\nTotal d'imatges trobades: {len(all_img_paths)}")

dataset = CroppedDataset(all_img_paths)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Generar embeddings
print("\nGenerant embeddings amb ResNet152 + MLP...")
with torch.no_grad():
    for batch_imgs, batch_paths in tqdm(loader, desc="Processant batches"):
        batch_imgs = batch_imgs.to(DEVICE)
        feats2048 = resnet(batch_imgs)
        emb128 = mlp(feats2048).cpu()
        
        for emb, path in zip(emb128, batch_paths):
            folder_name = os.path.basename(os.path.dirname(path))
            out_folder = os.path.join(EMB_DIR, folder_name)
            os.makedirs(out_folder, exist_ok=True)
            
            filename = os.path.splitext(os.path.basename(path))[0]
            torch.save(emb, os.path.join(out_folder, f"{filename}.pt"))

print(f"\nEmbeddings guardats a: {EMB_DIR}\n")

# ============================================================
# STEP 3: CREAR DATASET DE BAGS PER TOTS ELS PACIENTS
# ============================================================
print("="*70)
print("STEP 3: Creant bags per TOTS els pacients")
print("="*70)

class PatientBagDataset:
    def __init__(self, emb_dir, df_patients):
        self.emb_dir = emb_dir
        self.df = df_patients
        self.patients = []
        self.labels = []
        
        # Validar que cada pacient te embeddings
        for _, row in tqdm(df_patients.iterrows(), total=len(df_patients), desc="Validant pacients"):
            patient_id = row["CODI"]
            folders = sorted(glob.glob(os.path.join(emb_dir, f"{patient_id}_*")))
            
            emb_paths = []
            for folder in folders:
                emb_paths.extend(sorted(glob.glob(os.path.join(folder, "*.pt"))))
            
            if len(emb_paths) > 0:
                self.patients.append(patient_id)
                self.labels.append(int(row["label"]))
        
        print(f"\nPacients valids amb embeddings: {len(self.patients)}")
        print(f"  Positius: {sum(self.labels)}")
        print(f"  Negatius: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        label = self.labels[idx]
        
        folders = sorted(glob.glob(os.path.join(self.emb_dir, f"{patient_id}_*")))
        emb_paths = []
        for folder in folders:
            emb_paths.extend(sorted(glob.glob(os.path.join(folder, "*.pt"))))
        
        embeddings = [torch.load(p) for p in emb_paths]
        bag = torch.stack(embeddings, dim=0)  # [Np, 128]
        
        return patient_id, bag, label

bag_dataset = PatientBagDataset(EMB_DIR, df_all)
print()

# ============================================================
# STEP 4: MODEL D'ATENCIO
# ============================================================
print("="*70)
print("STEP 4: Definint model d'atencio")
print("="*70)

class ImprovedAttentionModel(nn.Module):
    def __init__(self, emb_dim=128, L=128, branches=1):
        super().__init__()
        
        att_params = {
            "in_features": emb_dim,
            "decom_space": L,
            "ATTENTION_BRANCHES": branches
        }
        self.attention = GatedAttention(att_params)
        
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * branches, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, bag):
        if bag.dim() == 1:
            bag = bag.unsqueeze(0)
        
        Z, A = self.attention(bag)
        Z = Z.view(1, -1)
        logit = self.classifier(Z)
        prob = torch.sigmoid(logit)
        
        return prob.squeeze(), A

print("Model definit\n")

# ============================================================
# STEP 5: ENTRENAMENT AMB CROSS-VALIDATION
# ============================================================
print("="*70)
print("STEP 5: Entrenament amb StratifiedGroupKFold (K=5)")
print("="*70)

patients = bag_dataset.patients
labels = bag_dataset.labels
groups = patients

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

EPOCHS = 50
LR = 5e-5
fold_results = []

for fold, (train_idx, val_idx) in enumerate(sgkf.split(patients, labels, groups)):
    print(f"\n{'='*70}")
    print(f"FOLD {fold+1}/5")
    print(f"{'='*70}")
    
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    
    print(f"Train: {len(train_idx)} | Pos: {sum(train_labels)}, Neg: {len(train_labels)-sum(train_labels)}")
    print(f"Val:   {len(val_idx)} | Pos: {sum(val_labels)}, Neg: {len(val_labels)-sum(val_labels)}")
    
    # Model
    model = ImprovedAttentionModel().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(MODELS_DIR_OUT, f"fold_{fold+1}_best.pth")
    
    for ep in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        
        indices = list(train_idx)
        random.shuffle(indices)
        
        for idx in indices:
            pid, bag, label = bag_dataset[idx]
            bag = bag.to(DEVICE)
            y = torch.tensor(float(label), device=DEVICE)
            
            optimizer.zero_grad()
            prob, A = model(bag)
            loss = criterion(prob, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for idx in val_idx:
                pid, bag, label = bag_dataset[idx]
                bag = bag.to(DEVICE)
                y = torch.tensor(float(label), device=DEVICE)
                
                prob, A = model(bag)
                loss = criterion(prob, y)
                
                val_loss += loss.item()
                val_preds.append(float(prob))
                val_true.append(label)
        
        avg_train = train_loss / len(train_idx)
        avg_val = val_loss / len(val_idx)
        val_acc = sum([1 for p, t in zip(val_preds, val_true) if (p>0.5)==t]) / len(val_true)
        
        scheduler.step(avg_val)
        
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} (Acc: {val_acc:.3f})")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {ep+1}")
                break
    
    print(f"Best val loss: {best_val_loss:.4f}")
    
    fold_results.append({
        "fold": fold + 1,
        "best_val_loss": best_val_loss,
        "model_path": best_model_path
    })

# ============================================================
# STEP 6: PREDIR TOTS ELS PACIENTS AMB ENSEMBLE
# ============================================================
print("\n" + "="*70)
print("STEP 6: PREDICCIO DE TOTS ELS PACIENTS (ENSEMBLE)")
print("="*70)

# Carregar els 5 models entrenats
models = []
for fold_info in fold_results:
    model = ImprovedAttentionModel().to(DEVICE)
    model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
    model.eval()
    models.append(model)

print(f"Models carregats: {len(models)}")

# Predir cada pacient amb tots els models
all_predictions = []

print(f"\nPredint {len(bag_dataset)} pacients...")
for idx in tqdm(range(len(bag_dataset)), desc="Prediccions"):
    pid, bag, true_label = bag_dataset[idx]
    bag = bag.to(DEVICE)
    
    # Obtenir prediccions dels 5 models
    probs = []
    with torch.no_grad():
        for model in models:
            prob, A = model(bag)
            probs.append(float(prob))
    
    # Ensemble: mitjana de probabilitats
    ensemble_prob = np.mean(probs)
    ensemble_pred = 1 if ensemble_prob > 0.5 else 0
    
    all_predictions.append({
        "patient_id": pid,
        "true_label": true_label,
        "true_diagnosis": "POSITIU (HP+)" if true_label == 1 else "NEGATIU (HP-)",
        "ensemble_probability": ensemble_prob,
        "predicted_label": ensemble_pred,
        "predicted_diagnosis": "POSITIU (HP+)" if ensemble_pred == 1 else "NEGATIU (HP-)",
        "correct": ensemble_pred == true_label,
        "prob_model_1": probs[0],
        "prob_model_2": probs[1],
        "prob_model_3": probs[2],
        "prob_model_4": probs[3],
        "prob_model_5": probs[4]
    })

# Guardar totes les prediccions
pred_df = pd.DataFrame(all_predictions)
pred_csv = os.path.join(RESULTS_DIR, "ALL_patient_predictions.csv")
pred_df.to_csv(pred_csv, index=False)

print(f"\nPrediccions guardades a: {pred_csv}")

# ============================================================
# STEP 7: METRIQUES GLOBALS
# ============================================================
print("\n" + "="*70)
print("STEP 7: METRIQUES GLOBALS")
print("="*70)

y_true = pred_df["true_label"].values
y_pred = pred_df["predicted_label"].values
y_prob = pred_df["ensemble_probability"].values

# Accuracy global
accuracy = sum(pred_df["correct"]) / len(pred_df)
print(f"\nAccuracy global: {accuracy:.4f} ({sum(pred_df['correct'])}/{len(pred_df)})")

# Matriu de confusio
cm = confusion_matrix(y_true, y_pred)
print("\nMatriu de Confusio:")
print("                Pred NEG  Pred POS")
print(f"True NEG           {cm[0,0]:3d}       {cm[0,1]:3d}")
print(f"True POS           {cm[1,0]:3d}       {cm[1,1]:3d}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["NEGATIU", "POSITIU"]))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
auc_score = auc(fpr, tpr)

print(f"\nAUC: {auc_score:.4f}")

# Plot ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", linewidth=2.5, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title(f"ROC Curve - TOTS ELS PACIENTS (N={len(patients)})", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()

roc_path = os.path.join(RESULTS_DIR, "ROC_ALL_patients.png")
plt.savefig(roc_path, dpi=300)
plt.close()

print(f"\nFigura ROC guardada: {roc_path}")

# Guardar resum
summary = {
    "total_patients": len(patients),
    "positive_patients": int(sum(y_true)),
    "negative_patients": int(len(y_true) - sum(y_true)),
    "accuracy": float(accuracy),
    "auc": float(auc_score),
    "true_negatives": int(cm[0,0]),
    "false_positives": int(cm[0,1]),
    "false_negatives": int(cm[1,0]),
    "true_positives": int(cm[1,1])
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(RESULTS_DIR, "summary_metrics.csv"), index=False)

print("\n" + "="*70)
print("PIPELINE COMPLETAT!")
print("="*70)
print(f"\nResultats guardats a: {RESULTS_DIR}")
print(f"  - ALL_patient_predictions.csv: Prediccio de cada pacient")
print(f"  - ROC_ALL_patients.png: Corba ROC")
print(f"  - summary_metrics.csv: Resum de metriques")
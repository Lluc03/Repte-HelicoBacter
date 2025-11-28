# -*- coding: utf-8 -*-
"""
Pipeline complet amb:
- Missatges de consola en anglès
- Comentaris interns en català
"""

# -------------------- IO Libraries --------------------
import os
import pickle
import glob
import gc
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -------------------- Standard Libraries --------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------- Torch Libraries --------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# -------------------- Own Functions --------------------
from models.AEmodels import AutoEncoderCNN

# -------------------- Time --------------------
import time

start_total = time.time()
print("Starting the full pipeline...")

# ================================================================
# 0. AUXILIARY DATASET FUNCTIONS
# ================================================================

def LoadCropped(folders_list, n_images_per_folder, size=(256, 256)):
    """
    Load cropped healthy patches.
    """
    Ims = []
    metadata = []

    for folder in folders_list:
        # carregar llistat d'imatges dins la carpeta
        image_files = glob.glob(os.path.join(folder, '*.png'))
        image_files = image_files[:n_images_per_folder]

        for img_path in image_files:
            # lectura i preprocessat d'imatges
            img = Image.open(img_path).convert('RGB')
            img = img.resize(size)  # redimensionament
            arr = np.array(img).transpose(2, 0, 1) / 255.0
            Ims.append(arr)

            metadata.append({
                'PatID': os.path.basename(folder),
                'imfilename': os.path.basename(img_path)
            })

    return np.array(Ims, dtype=np.float32), pd.DataFrame(metadata)


def LoadAnnotated(folders_list, n_images_per_folder, patient_csv, size=(256,256)):
    """
    Load annotated patches (healthy + infected).
    """
    Ims = []
    metadata = []

    df_labels = pd.read_csv(patient_csv)

    for folder in folders_list:
        image_files = glob.glob(os.path.join(folder, '*.png'))
        image_files = image_files[:n_images_per_folder]

        # extreure id pacient de la carpeta
        pat_id = os.path.basename(folder)
        label = int(pat_id.split('_')[-1])  # 0 o 1

        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(size)
            arr = np.array(img).transpose(2, 0, 1) / 255.0
            Ims.append(arr)

            metadata.append({
                'PatID': pat_id,
                'imfilename': os.path.basename(img_path),
                'presenceHelico': label
            })

    return np.array(Ims, dtype=np.float32), pd.DataFrame(metadata)


# ================================================================
# 1. PARAMETERS
# ================================================================
inputmodule_paramsEnc = {'num_input_channels': 3}

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

# Obtenir la GPU assignada pel cluster
# Normalment Slurm defineix CUDA_VISIBLE_DEVICES
gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
if gpu_id is None:
    raise RuntimeError("No hi ha cap GPU assignada pel cluster. Assegura't de reservar-la correctament!")

DEVICE = torch.device(f"cuda:{gpu_id}")

print(f"Usant GPU: {DEVICE}")
print(f"Running on device: {DEVICE}")

DATASET_DIR = '/fhome/maed/HelicoDataSet/CrossValidation'
RESULTS_DIR = '/fhome/maed02/proj_repte3/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("AUTOENCODER TRAINING FOR H. PYLORI DETECTION")
print("="*70)

# ================================================================
# 2. LOAD DATA
# ================================================================
print("1. Loading datasets...")
start_load = time.time()

cropped_path = os.path.join(DATASET_DIR, 'Cropped')
annotated_path = os.path.join(DATASET_DIR, 'Annotated')
patient_csv = '/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'

# llegir CSV i netejar
df_diag = pd.read_csv(patient_csv)
df_diag['CODI'] = df_diag['CODI'].str.strip()

# seleccionar només pacients negatius
negatives = set(df_diag[df_diag['DENSITAT'].str.upper() == 'NEGATIVA']['CODI'])
print("Negative patients found:", negatives)

# filtrar carpetes
cropped_folders_all = sorted(os.listdir(cropped_path))
healthy_folders = [os.path.join(cropped_path, f) for f in cropped_folders_all if f.split('_')[0] in negatives]

print(f"Total healthy folders available: {len(healthy_folders)}")

# SPLIT 80/20 PER PACIENT
train_folders, val_folders = train_test_split(healthy_folders, test_size=0.2, random_state=42)

print(f"Training folders: {len(train_folders)}")
print(f"Validation folders: {len(val_folders)}")

# carregar dades
train_imgs, _ = LoadCropped(train_folders, n_images_per_folder=50)
val_imgs, _ = LoadCropped(val_folders, n_images_per_folder=50)
Ims_annotated, meta_annotated = LoadAnnotated(glob.glob(os.path.join(annotated_path, '*')), 30, patient_csv)

print(f"Loaded train: {len(train_imgs)} images")
print(f"Loaded validation: {len(val_imgs)} images")
print(f"Loaded test annotated: {len(Ims_annotated)} images")

print(f"Data loading time: {time.time() - start_load:.2f} seconds")


# ================================================================
# 3. DATA LOADERS
# ================================================================
train_loader = DataLoader(torch.tensor(train_imgs), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(torch.tensor(val_imgs), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(torch.tensor(Ims_annotated), batch_size=BATCH_SIZE, shuffle=False)

# ================================================================
# 4. MODEL CONFIGURATION
# ================================================================

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

Config = '1'
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec).to(DEVICE)

# ================================================================
# 5. TRAINING
# ================================================================
print("2. Training AutoEncoder...")
start_train = time.time()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        images = batch.to(DEVICE)
        recon = model(images)
        loss = criterion(recon, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # fase validació
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch.to(DEVICE)
            recon = model(images)
            val_loss += criterion(recon, images).item()

    avg_train = epoch_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    losses.append((avg_train, avg_val))

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

print(f"Training time: {time.time() - start_train:.2f} seconds")

# ================================================================
# 6. SAVE RESULTS
# ================================================================
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f'autoencoder_config{Config}.pth'))
with open(os.path.join(RESULTS_DIR, 'training_losses.pkl'), 'wb') as f:
    pickle.dump(losses, f)

print("Model and losses successfully saved.")

print("Total pipeline time: {:.2f} seconds".format(time.time() - start_total))
print("Pipeline finished successfully.")

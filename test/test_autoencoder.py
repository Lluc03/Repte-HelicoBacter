import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.AEmodels import AutoEncoderCNN


# ================================================================
# PATHS
# ================================================================
DATASET_DIR = "/fhome/maed/HelicoDataSet/CrossValidation"
CROPPED_PATH = os.path.join(DATASET_DIR, "Cropped")
PATIENT_CSV = "/fhome/maed/HelicoDataSet/PatientDiagnosis.csv"

AUTOENCODER_WEIGHTS = "/fhome/maed02/proj_repte3/results/autoencoder_config1.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# LOAD TRAINING HEALTHY PATCHES (same as training)
# ================================================================
def LoadHealthyForTest(cropped_path, diagnosis_csv, n_images=50, size=(256,256)):
    df = pd.read_csv(diagnosis_csv)
    df["CODI"] = df["CODI"].str.strip()

    healthy_ids = set(df[df["DENSITAT"].str.upper() == "NEGATIVA"]["CODI"])

    folders = [
        os.path.join(cropped_path, f)
        for f in sorted(os.listdir(cropped_path))
        if f.split("_")[0] in healthy_ids
    ]

    print(f"Carpetes de pacients sans trobades: {len(folders)}")

    imgs = []

    for folder in folders:
        files = sorted(glob.glob(os.path.join(folder, "*.png")))[:n_images]

        for img_path in files:
            img = Image.open(img_path).convert("RGB").resize(size)
            arr = np.array(img).transpose(2, 0, 1) / 255.0
            imgs.append(arr)

    imgs = np.array(imgs, dtype=np.float32)
    return imgs


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    import pandas as pd

    print("Carregant patches sans utilitzats en training...")
    train_imgs = LoadHealthyForTest(CROPPED_PATH, PATIENT_CSV, n_images=50)

    print(f"Patches carregats: {len(train_imgs)}")

    # DataLoader
    loader = DataLoader(torch.tensor(train_imgs), batch_size=32, shuffle=False)

    # ============================================================
    # LOAD AUTOENCODER (CONFIG 1)
    # ============================================================
    print("Carregant AutoEncoder...")

    inputmodule_paramsEnc = {'num_input_channels': 3}

    def AEConfigs(Config):
        net_paramsEnc = {}
        net_paramsDec = {}
        inputmodule_paramsDec = {}

        if Config == '1':
            net_paramsEnc['block_configs'] = [[32, 32], [64, 64]]
            net_paramsEnc['stride'] = [[1, 2], [1, 2]]
            net_paramsDec['block_configs'] = [[64, 32], [32, inputmodule_paramsEnc['num_input_channels']]]
            net_paramsDec['stride'] = net_paramsEnc['stride']
            inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

        return net_paramsEnc, net_paramsDec, inputmodule_paramsDec

    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs('1')

    model = AutoEncoderCNN(
        inputmodule_paramsEnc,
        net_paramsEnc,
        inputmodule_paramsDec,
        net_paramsDec
    ).to(DEVICE)

    model.load_state_dict(torch.load(AUTOENCODER_WEIGHTS))
    model.eval()

    # ============================================================
    # COMPUTE RECONSTRUCTION ERROR
    # ============================================================
    print("Calculant error de reconstrucció...")
    criterion = nn.MSELoss(reduction="none")

    all_errors = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)

            err = criterion(recon, batch).mean(dim=(1,2,3))
            all_errors.extend(err.cpu().numpy())

    all_errors = np.array(all_errors)

    print("\n========== RESULTATS ==========")
    print(f"Mitjana error reconstrucció: {all_errors.mean():.6f}")
    print(f"Desviació estàndard:          {all_errors.std():.6f}")
    print(f"Error mínim:                  {all_errors.min():.6f}")
    print(f"Error màxim:                  {all_errors.max():.6f}")
# ================================================================
# EXTRACT RESNET FEATURES (System2 - FEATURES)
# ================================================================
# Genera un NPZ amb:
#   - X_feat: features 2048D de ResNet152
#   - y: labels patch-level
#   - groups: pacient de cada patch (per ROC StratifiedGroupKFold)
#
# IMPORTANT:
#   Aquest script garanteix que el .npz tingui EXACTAMENT N entrades,
#   i les mateixes en el mateix ordre que necessitarà la ROC.
# ================================================================

import os, glob, sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_extractor.feature_extractors import ResNetFeatureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "/fhome/maed/HelicoDataSet/CrossValidation"
ANNOTATED_PATH = os.path.join(DATASET_DIR, "Annotated")
EXCEL = "/fhome/maed/HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"

OUT_DIR = "/fhome/maed02/proj_repte3/results/system2_resnet_features"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "patch_features_resnet152.npz")


# ================================================================
# LOAD PATCHES + LABELS + GROUPS
# ================================================================
def load_annotated_patches(base_path, annot_df, size=(224,224)):
    X = []
    y = []
    groups = []

    folders = sorted(glob.glob(os.path.join(base_path, "*")))

    for folder in folders:
        folder_name = os.path.basename(folder)
        pat_id = folder_name.split("_")[0]
        section_id = int(folder_name.split("_")[1])

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
            arr = np.array(img).transpose(2,0,1) / 255.0

            X.append(arr)
            y.append(label)
            groups.append(pat_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups)

    print(f"Total patches carregats: {len(X)}")
    print(f"Classe 0: {(y==0).sum()}, Classe 1: {(y==1).sum()}")
    print(f"N pacients únics: {len(np.unique(groups))}")

    return X, y, groups


# ================================================================
# DATASET PER DATALOADER
# ================================================================
class PatchTensorDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx]


# ================================================================
# MAIN
# ================================================================
def main():

    print("Carregant Excel d'anotacions...")
    annot_df = pd.read_excel(EXCEL)

    print("Carregant patches Annotated...")
    X, y, groups = load_annotated_patches(ANNOTATED_PATH, annot_df)

    ds = PatchTensorDataset(X)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    print("Carregant ResNet152...")
    backbone = ResNetFeatureExtractor(pretrained=True).to(DEVICE)
    backbone.eval()

    all_feats = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            feats = backbone(batch)    # (B,2048)
            all_feats.append(feats.cpu().numpy())

    X_feat = np.concatenate(all_feats, axis=0)

    print("Guardant NPZ amb X_feat, y, groups...")
    np.savez(OUT_FILE, X_feat=X_feat, y=y, groups=groups)

    print("\n✔ Features guardades correctament a:")
    print(OUT_FILE)
    print("✔ Shape:", X_feat.shape)


if __name__ == "__main__":
    main()

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Per poder importar AutoEncoderCNN
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.AEmodels import AutoEncoderCNN


# =============================================================
# CONFIG
# =============================================================
MODEL_PATH = "/fhome/maed02/proj_repte3/results/autoencoder_config1.pth"

IMAGE_FILES = [
    "/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-129_0/00659.png",
    "/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-46_0/01340.png"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (256, 256)

OUTPUT_DIR = "/export/fhome/maed02/proj_repte3/test/reconstructions_h_channel"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# RGB → HSV (incloent només el canal H)
# =============================================================
def rgb_to_h_torch(img):
    """
    img: tensor (B,3,H,W) en [0,1]
    return: canal H només, tensor (B,H,W)
    """
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    maxc, _ = torch.max(img, dim=1)
    minc, _ = torch.min(img, dim=1)
    deltac = maxc - minc

    # Hue (H)
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
    return h  # només canal H


# =============================================================
# LOAD IMAGE
# =============================================================
def load_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).transpose(2,0,1) / 255.0
    tensor = torch.from_numpy(arr).float().unsqueeze(0)  # (1,3,H,W)
    return img, tensor


# =============================================================
# LOAD MODEL
# =============================================================
def AEConfigs(Config):
    net_paramsEnc = {}
    net_paramsDec = {}
    inputmodule_paramsDec = {}
    inputmodule_paramsEnc = {"num_input_channels": 3}

    if Config == '1':
        net_paramsEnc['block_configs'] = [[32,32],[64,64]]
        net_paramsEnc['stride'] = [[1,2],[1,2]]
        net_paramsDec['block_configs'] = [[64,32],[32,inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    return inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec


inputmodule_paramsEnc, netE, inputmodule_paramsDec, netD = AEConfigs("1")

model = AutoEncoderCNN(inputmodule_paramsEnc, netE, inputmodule_paramsDec, netD).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model carregat correctament.\n")


# =============================================================
# EXECUCIÓ PEL CANAL H
# =============================================================
for fname in IMAGE_FILES:
    print(f"Processant {fname}...")

    img_rgb, x = load_image(fname)
    x = x.to(DEVICE)

    with torch.no_grad():
        rec = model(x).clamp(0,1)

    # Canal H original i reconstruït
    H_orig  = rgb_to_h_torch(x)[0].cpu().numpy()
    H_recon = rgb_to_h_torch(rec)[0].cpu().numpy()

    # Errors
    H_diff  = H_recon - H_orig
    H_diff2 = (H_recon - H_orig)**2

    # =========================================================
    # PLOT FINAL
    # =========================================================
    fig, ax = plt.subplots(1,4,figsize=(14,4))

    ax[0].imshow(img_rgb)
    ax[0].set_title("Original RGB")
    ax[0].axis("off")

    ax[1].imshow(H_orig, cmap="hsv")
    ax[1].set_title("H original")
    ax[1].axis("off")

    ax[2].imshow(H_recon, cmap="hsv")
    ax[2].set_title("H reconstruït")
    ax[2].axis("off")

    im = ax[3].imshow(H_diff2, cmap="hot")
    ax[3].set_title("Error (H - H')²")
    ax[3].axis("off")
    fig.colorbar(im, ax=ax[3])

    base = os.path.basename(fname)
    out_name = os.path.join(OUTPUT_DIR, f"H_channel_{base}")
    plt.savefig(out_name, bbox_inches='tight')
    plt.close()

    print(f"✔ Guardat: {out_name}")

print("\nFi del test del canal H.")

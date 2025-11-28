"""
data_split.py
--------------------------------------------
Utilitat central per garantir que TOT System2
utilitza el mateix split Train / Validation / Test.

Un cop es crea, queda guardat en un fitxer NPZ.
Qualsevol part del pipeline pot carregar-lo
i utilitzar exactament els mateixos indexs.
--------------------------------------------
Autor: Lluc Verdaguer
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

# On es guardarà el split
SPLIT_FILE = "/fhome/maed02/proj_repte3/data/system2_split_indices.npz"


def load_or_create_split(num_samples, labels, val_size=0.15, test_size=0.0, seed=42):
    """
    Crea o carrega un split consistent per a tot System2.
    
    Retorna:
        train_idx, val_idx, test_idx
    """

    # ----------------------------------------------------
    # 1. SI JA EXISTEIX, ES CARREGA DIRECTAMENT
    # ----------------------------------------------------
    if os.path.exists(SPLIT_FILE):
        print(f"[System2 Split] Carregant split existent: {SPLIT_FILE}")
        data = np.load(SPLIT_FILE)
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        test_idx = data["test_idx"]
        return train_idx, val_idx, test_idx

    # ----------------------------------------------------
    # 2. SI NO EXISTEIX, ES CREA
    # ----------------------------------------------------
    print("[System2 Split] No existeix split. Creant-ne un de nou...")

    idx = np.arange(num_samples)
    labels = np.array(labels)

    # Primer fem split train+val / test
    if test_size > 0:
        X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
            np.zeros(len(idx)), labels, idx,
            test_size=test_size,
            stratify=labels,
            random_state=seed
        )
    else:
        idx_temp = idx
        idx_test = np.array([], dtype=int)

    # Després fem train / val
    val_ratio = val_size / (1 - test_size) if test_size > 0 else val_size

    _, _, _, _, train_idx, val_idx = train_test_split(
        np.zeros(len(idx_temp)), labels[idx_temp], idx_temp,
        test_size=val_ratio,
        stratify=labels[idx_temp],
        random_state=seed
    )

    # ----------------------------------------------------
    # 3. GUARDEM EL SPLIT
    # ----------------------------------------------------
    os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok=True)
    np.savez(
        SPLIT_FILE,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=idx_test
    )

    print(f"[System2 Split] Split guardat a: {SPLIT_FILE}")

    return train_idx, val_idx, idx_test

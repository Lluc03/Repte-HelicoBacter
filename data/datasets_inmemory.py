# -*- coding: utf-8 -*-
"""
Dataset in-memory optimitzat per evitar el bottleneck d'I/O.
Carrega totes les imatges a RAM una sola vegada i usa DataLoader eficient.

Basat en les millors pràctiques de:
- GPU Profiling & Monitoring for PyTorch
- PyTorch DataLoader optimization patterns
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ================================================================
# IN-MEMORY DATASET CLASSES
# ================================================================

class InMemoryHealthyDataset(Dataset):
    """
    Dataset que carrega tots els patches sans a RAM al inicialitzar-se.
    Evita accés a disc durant l'entrenament.
    """
    
    def __init__(
        self,
        folders_list: List[str],
        n_images_per_folder: int,
        size: Tuple[int, int] = (256, 256),
        verbose: bool = True
    ):
        """
        Args:
            folders_list: Llista de paths a carpetes amb imatges
            n_images_per_folder: Nombre màxim d'imatges per carpeta
            size: Mida de redimensionament (H, W)
            verbose: Mostrar progrés de càrrega
        """
        self.size = size
        self.images = []  # Totes les imatges en RAM com tensors
        self.metadata = []
        
        if verbose:
            print(f"\n?? Loading {len(folders_list)} folders into RAM...")
        
        for idx, folder in enumerate(folders_list):
            # Llistar imatges de la carpeta
            image_files = glob.glob(os.path.join(folder, '*.png'))
            image_files = image_files[:n_images_per_folder]
            
            for img_path in image_files:
                # Carregar i preprocessar imatge
                img = Image.open(img_path).convert('RGB')
                img = img.resize(size)
                
                # Convertir a tensor directament (C, H, W) i normalitzar
                arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr)
                
                self.images.append(tensor)
                self.metadata.append({
                    'PatID': os.path.basename(folder),
                    'imfilename': os.path.basename(img_path)
                })
            
            if verbose and (idx + 1) % 10 == 0:
                print(f"  Loaded {idx + 1}/{len(folders_list)} folders ({len(self.images)} images)")
        
        if verbose:
            total_mb = sum(t.element_size() * t.nelement() for t in self.images) / (1024**2)
            print(f"? Loaded {len(self.images)} images ({total_mb:.1f} MB in RAM)\n")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Retorna directament el tensor des de RAM.
        No hi ha càrrega de disc ? molt ràpid.
        """
        return self.images[idx]


class InMemoryAnnotatedDataset(Dataset):
    """
    Dataset para patches anotados (bacteria o no) cargados en RAM.
    Etiquetas: 1=bacteria, -1=no bacteria, 0=desconocido
    """
    
    def __init__(
        self,
        folders_list: List[str],
        n_images_per_folder: int,
        patient_csv: str,
        size: Tuple[int, int] = (256, 256),
        verbose: bool = True
    ):
        self.size = size
        self.images = []
        self.labels = []
        self.metadata = []
        
        # Cargar CSV de diagnósticos
        df_labels = pd.read_csv(patient_csv)
        
        if verbose:
            print(f"\n?? Loading {len(folders_list)} annotated folders into RAM...")
        
        for idx, folder in enumerate(folders_list):
            image_files = glob.glob(os.path.join(folder, '*.png'))
            image_files = image_files[:n_images_per_folder]
            
            # Extraer etiqueta del nombre de carpeta o CSV
            pat_id = os.path.basename(folder)
            raw_label = int(pat_id.split('_')[-1])  # Valor original
            # Mapear a tus nuevas etiquetas
            if raw_label == 1:
                label = 1        # bacteria
            elif raw_label == 0:
                label = 0        # desconocido
            else:
                label = -1       # no bacteria
            
            for img_path in image_files:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(size)
                
                arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr)
                
                self.images.append(tensor)
                self.labels.append(label)
                self.metadata.append({
                    'PatID': pat_id,
                    'imfilename': os.path.basename(img_path),
                    'presenceHelico': label
                })
            
            if verbose and (idx + 1) % 10 == 0:
                print(f"  Loaded {idx + 1}/{len(folders_list)} folders ({len(self.images)} images)")
        
        if verbose:
            total_mb = sum(t.element_size() * t.nelement() for t in self.images) / (1024**2)
            print(f"? Loaded {len(self.images)} images ({total_mb:.1f} MB in RAM)")
            print(f"   - Bacteria (1): {self.labels.count(1)}")
            print(f"   - No bacteria (-1): {self.labels.count(-1)}")
            print(f"   - Desconocido (0): {self.labels.count(0)}\n")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Retorna (imagen, etiqueta) desde RAM.
        """
        return self.images[idx], self.labels[idx]


# ================================================================
# OPTIMIZED DATALOADER FACTORY
# ================================================================

def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    device: Optional[str] = None
) -> DataLoader:
    """
    Crea un DataLoader optimitzat segons les millors pràctiques.
    
    Args:
        dataset: Dataset (in-memory o regular)
        batch_size: Mida del batch
        shuffle: Barrejar dades
        num_workers: Nombre de workers CPU (recomanat: 4-8)
        pin_memory: Usar page-locked memory per H2D ràpid
        persistent_workers: Mantenir workers vius entre èpoques
        prefetch_factor: Batches pre-carregats per worker
        device: Device per non-blocking transfers ('cuda' o None)
    
    Returns:
        DataLoader configurat
        
    Notes:
        - pin_memory=True + non_blocking=True en .to(device) ? transferències H2D ràpides
        - persistent_workers=True ? evita reiniciar workers cada època
        - prefetch_factor ? reduce gaps entre batches
    """
    
    # Validació de paràmetres
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False  # No descartar últim batch incomplet
    )
    
    return loader


# ================================================================
# CONVENIENCE FUNCTIONS (mantenen API original)
# ================================================================

def LoadCroppedInMemory(
    folders_list: List[str],
    n_images_per_folder: int,
    size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Funció de compatibilitat amb l'API original.
    Retorna arrays numpy en lloc de Dataset.
    
    Returns:
        (images_array, metadata_df)
    """
    dataset = InMemoryHealthyDataset(folders_list, n_images_per_folder, size)
    
    # Convertir tensors a numpy
    images = np.stack([img.numpy() for img in dataset.images])
    metadata = pd.DataFrame(dataset.metadata)
    
    return images, metadata


def LoadAnnotatedInMemory(
    folders_list: List[str],
    n_images_per_folder: int,
    patient_csv: str,
    size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Funció de compatibilitat amb l'API original per dades anotades.
    
    Returns:
        (images_array, metadata_df)
    """
    dataset = InMemoryAnnotatedDataset(folders_list, n_images_per_folder, patient_csv, size)
    
    images = np.stack([img.numpy() for img in dataset.images])
    
    # Afegir etiquetes al metadata
    metadata = pd.DataFrame(dataset.metadata)
    
    return images, metadata


# ================================================================
# USAGE EXAMPLE
# ================================================================

if __name__ == "__main__":
    """
    Exemple d'ús del dataset in-memory amb DataLoader optimitzat.
    """
    
    # Configuració
    CROPPED_PATH = '/fhome/maed/HelicoDataSet/CrossValidation/Cropped'
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Ajustar segons CPUs disponibles
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("EXAMPLE: IN-MEMORY DATASET + OPTIMIZED DATALOADER")
    print("="*70)
    
    # Exemple: carregar primer 3 pacients
    all_folders = sorted(glob.glob(os.path.join(CROPPED_PATH, '*')))[:3]
    
    # Crear dataset in-memory
    dataset = InMemoryHealthyDataset(
        folders_list=all_folders,
        n_images_per_folder=50,
        size=(256, 256)
    )
    
    # Crear DataLoader optimitzat
    loader = create_optimized_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"Dataset size: {len(dataset)} images")
    print(f"Batches per epoch: {len(loader)}")
    print(f"Device: {DEVICE}\n")
    
    # Simular una època d'entrenament
    print("Simulating one epoch...")
    import time
    
    start = time.time()
    for batch_idx, images in enumerate(loader):
        # Transfer a GPU amb non-blocking per overlapping
        images = images.to(DEVICE, non_blocking=True)
        
        # Aquí aniria el forward pass del model
        # output = model(images)
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Shape: {images.shape}")
    
    elapsed = time.time() - start
    samples_per_sec = len(dataset) / elapsed
    
    print(f"\n? Epoch completed in {elapsed:.2f}s ({samples_per_sec:.1f} samples/sec)")
    print("="*70)
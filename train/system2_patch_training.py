# patch_training.py

import torch
import torch.nn as nn
from typing import Optional, Tuple

from data.datasets import TripletDataset, Standard_Dataset
from utils.triplet_loss import TripletLoss  # si el tens a utils, sinó canvia path

from feature_extractor.patch_models import (
    PatchEmbeddingNet,
    PatchClassifierHead,
    PatchEmbeddingAndClassifier
)


# ---------------------------------------------------------
# 1. ENTRENAMENT CONTRASTIU (TRIPLET LOSS)
# ---------------------------------------------------------

def train_triplet_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: TripletLoss,
    device: torch.device,
    accumulation_steps: int = 8   # 👈 Afegit
) -> float:

    model.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for i, (anchor, positive, negative, _) in enumerate(dataloader):

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        # 🔵 Forward
        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = loss_fn(emb_a, emb_p, emb_n)

        # 🔥 Divideix la loss entre acumulacions
        loss = loss / accumulation_steps
        loss.backward()

        # 🔥 Quan acumules prou gradients → actualitza
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_triplet_epoch(
    model: nn.Module,
    dataloader,
    loss_fn: TripletLoss,
    device: torch.device
) -> float:
    """
    Avaluació (validació) d'una època amb Triplet Loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for anchor, positive, negative, _ in dataloader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = loss_fn(emb_a, emb_p, emb_n)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Calcula pesos per classe per CrossEntropyLoss en cas de desequilibri.
    labels: Tensor 1D amb valors {0,1}
    """
    labels_np = labels.cpu().numpy()
    n0 = (labels_np == 0).sum()
    n1 = (labels_np == 1).sum()
    total = n0 + n1

    if n0 == 0 or n1 == 0:
        # cas degenerat
        return torch.ones(2, dtype=torch.float32)

    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)
    weights = torch.tensor([w0, w1], dtype=torch.float32)
    return weights


# ---------------------------------------------------------
# 2. ENTRENAMENT CLASSIFICADOR SOBRE EMBEDDINGS
# ---------------------------------------------------------

def train_classifier_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Entrena una època el classificador (embedding + classifier wrapper).
    dataloader ha de retornar: imatge, label
    Retorna: loss mitjana, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device).long()

        optimizer.zero_grad()
        logits = model(x)           # [B, 2]
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(n_batches, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def eval_classifier_epoch(
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Avaluació d'una època del classificador.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device).long()

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(n_batches, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

# patch_models.py

import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_extractor.feature_extractors import ResNetFeatureExtractor
from feature_extractor.projection_head import ProjectionHead, NormalizedProjectionHead


class FeatureEmbeddingNet(nn.Module):
    """
    Rep DIRECTAMENT features de ResNet (2048) i
    aprèn l'embedding de 128D amb un MLP.
    Serveix quan ja tens els features precomputats i no vols tornar a usar la ResNet.
    """
    def __init__(self, in_dim=2048, proj_hidden_dim=256, proj_out_dim=128, normalize=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_out_dim)
        )
        self.normalize = normalize

    def forward(self, x):
        # x: (B, 2048)
        z = self.mlp(x)
        if self.normalize:
            z = nn.functional.normalize(z, p=2, dim=1)
        return z


class PatchEmbeddingNet(nn.Module):
    """
    Model que combina:
      - un extractor de característiques (ResNet o AE)
      - una projection head (MLP) per obtenir l'espai contrastiu X'
    """
    def __init__(
        self,
        backbone_type: str,
        proj_out_dim: int = 128,
        proj_hidden_dim: int = 512,
        # Paràmetres per AE (només si backbone_type == "ae")
        ae_model_path: str = None,
        AutoEncoderClass=None,
        inputmodule_paramsEnc: dict = None,
        net_paramsEnc: dict = None,
        inputmodule_paramsDec: dict = None,
        net_paramsDec: dict = None,
        device: torch.device = torch.device("cuda"),
        normalize: bool = False,
    ):
        super().__init__()

        self.backbone_type = backbone_type.lower()
        self.device = device

        if self.backbone_type == "resnet":
            self.backbone = ResNetFeatureExtractor(pretrained=True)
            in_dim = 2048
            for param in self.backbone.parameters():
                param.requires_grad = False

        elif self.backbone_type == "ae":
            if ae_model_path is None or AutoEncoderClass is None:
                raise ValueError("Per backbone_type='ae' cal passar model_path i AutoEncoderClass.")
            self.backbone = AEFeatureExtractor(
                model_path=ae_model_path,
                AutoEncoderClass=AutoEncoderClass,
                inputmodule_paramsEnc=inputmodule_paramsEnc,
                net_paramsEnc=net_paramsEnc,
                inputmodule_paramsDec=inputmodule_paramsDec,
                net_paramsDec=net_paramsDec,
                device=device
            )
            # La dimensió d'entrada dependrà de l'AE; no la sabem a priori.
            # Truquem una vegada de prova si cal (lazy init).
            in_dim = None

        else:
            raise ValueError("backbone_type ha de ser 'resnet' o 'ae'.")

        self.normalize = normalize
        self._proj_out_dim = proj_out_dim
        self._proj_hidden_dim = proj_hidden_dim
        self._proj_in_dim = in_dim  # pot ser None si AE

        if in_dim is not None:
            self._build_projection_head()
        else:
            # es construirà més endavant al primer forward amb AE
            self.proj = None

        self.to(device)

    def _build_projection_head(self):
        if self.normalize:
            self.proj = NormalizedProjectionHead(
                in_dim=self._proj_in_dim,
                hidden_dim=self._proj_hidden_dim,
                out_dim=self._proj_out_dim
            )
        else:
            self.proj = ProjectionHead(
                in_dim=self._proj_in_dim,
                hidden_dim=self._proj_hidden_dim,
                out_dim=self._proj_out_dim
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: imatges [B, 3, H, W]
        return: embeddings [B, proj_out_dim]
        """
        feats = self.backbone(x)  # [B, in_dim_?]

        # Si venim d'un AE i no sabíem la mida, construïm projection_head on-the-fly
        if self._proj_in_dim is None:
            self._proj_in_dim = feats.shape[1]
            self._build_projection_head()
            self.proj.to(self.device)

        z = self.proj(feats)
        return z


class PatchClassifierHead(nn.Module):
    """
    Classificador binari sobre l'embedding X'.
    """
    def __init__(self, emb_dim: int = 128, hidden_dim: int = 64, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: embeddings [B, emb_dim]
        return: logits [B, n_classes]
        """
        return self.net(x)


class PatchEmbeddingAndClassifier(nn.Module):
    """
    Wrapper que combina l'embedding net + classificador en un sol model.
    Ideal per entrenar el classifier (amb embedding ja entrenat).
    """
    def __init__(self, embedding_net: PatchEmbeddingNet, classifier_head: PatchClassifierHead):
        super().__init__()
        self.embedding_net = embedding_net
        self.classifier_head = classifier_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding_net(x)
        logits = self.classifier_head(z)
        return logits

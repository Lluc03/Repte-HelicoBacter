import torch
import torch.nn as nn
import torchvision.models as models
import os

# ------------------------------------------------------------------------------
# 1. RESNET FEATURE EXTRACTOR
# ------------------------------------------------------------------------------

class ResNetFeatureExtractor(nn.Module):
    """
    Extreu característiques de dimensió 2048 utilitzant una ResNet152 pre-entrenada.
    (S’elimina la capa FC final per obtenir l'embedding)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.resnet152(pretrained=pretrained)

        # Eliminem la capa final de classificació
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)
    

# ------------------------------------------------------------------------------
# 2. AUTOENCODER FEATURE EXTRACTOR
# ------------------------------------------------------------------------------

class AEFeatureExtractor(nn.Module):
    """
    Carrega l'AutoEncoderCNN del projecte i n'utilitza només l'encoder.
    Retorna el latent flatten (B, latent_dim).
    """

    def __init__(self, model_path, AutoEncoderClass,
                 inputmodule_paramsEnc, net_paramsEnc,
                 inputmodule_paramsDec, net_paramsDec,
                 device="cuda"):
        super().__init__()

        # Inicialitza el model AE complet
        self.ae = AutoEncoderClass(
            inputmodule_paramsEnc,
            net_paramsEnc,
            inputmodule_paramsDec,
            net_paramsDec
        )

        # Carrega weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model AE no trobat: {model_path}")

        state = torch.load(model_path, map_location=device)
        self.ae.load_state_dict(state)
        self.ae = self.ae.to(device)

        # Ens quedem només amb l’encoder
        self.encoder = self.ae.encoder

    def forward(self, x):
        """
        x: Tensor [B, 3, H, W]
        Output: Tensor flatten [B, latent_dim]
        """
        z = self.encoder(x)                # [B, C, H', W']
        z = torch.flatten(z, start_dim=1)  # [B, latent_dim]
        return z

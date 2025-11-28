# Repte Helicobacter Pylori
## Autor: Lluc Verdaguer Macias
# Descripció
Aquest projecte implementa dos sistemes diferents per detectar H. pylori a partir de patches extrets de biòpsies digitals.
L’objectiu és decidir si un pacient és POSITIU o NEGATIU en base als seus patches.

El procediment general és:
1. Dividir la biòpsia en múltiples patches.
2. Processar cada patch (reconstrucció o embeddings).
3. Combinar tota la informació per donar la predicció final del pacient.

Hi ha dos enfocaments:
**System 1**: AutoEncoder (detector d’anomalies).
**System 2:** ResNet152 + Multiple Instance Learning (MIL).

## 1. System 1 – AutoEncoder
### 1.1 Entrenament de l’AE

L’AutoEncoder s’entrena només amb patches sans.
La idea és que el model aprengui com “hauria de ser” un patch sa, i quan rep un patch amb H. pylori, la reconstrucció empitjora i l’error puja.

*Punts principals:*

Dataset: carpetes Cropped Sanes.

S’entrenen les primeres imatges de cada pacient.

Objectiu: detectar desviacions respecte l’estructura normal del teixit.

### 1.2 Error de reconstrucció

Al principi calculàvem el MSE en RGB, però no donava bona separació entre patches sans i no sans.
Després es va canviar a:

Convertir la imatge a HSV.

Calcular només el MSE del canal H (Hue).

Aquest canvi millora bastant la detecció perquè el canal H destaca millor diferències de color relacionades amb el bacteri.

### 1.3 Validació i mètriques

Per evitar leakage entre pacients es fa servir StratifiedGroupKFold:

Cada pacient només surt en un fold.

Estratificació per classe (POSITIU/NEGATIU).

A nivell pacient s’agafa la mitjana dels errors dels seus patches i es compara amb un threshold òptim (Youden J).

### Resultats generals:
ROC Curve (pacients):

<img width="636" height="503" alt="image" src="https://github.com/user-attachments/assets/e41cfa0d-e6da-4aca-8bf2-3d3ab178a102" />

Confussion Matrix:

<img width="729" height="584" alt="image" src="https://github.com/user-attachments/assets/1f06095d-6975-4e19-bf27-1dfabdc427a6" />

Mètriques:

<img width="1147" height="211" alt="image" src="https://github.com/user-attachments/assets/d6383ab3-d11b-4234-b2fa-0c6aebdeac28" />


## 2. System 2 – Embeddings + MIL

Aquest sistema és més complex però també més potent.

### 2.1 Extracció d’embeddings

Cada patch passa per:

ResNet152 pre-entrenada → vector de 2048 dimensions.

Projection Head (MLP) → embedding de 128 dimensions.

S’entrena amb Triplet Loss perquè els embeddings de patches de la mateixa classe quedin junts.

### 2.2 MIL amb Gated Attention

Un cop tenim els embeddings:

Cada pacient = bag de molts embeddings.

El model MIL amb Gated Attention selecciona quins patches són més rellevants.

Entrenament en 5 folds i predicció final fent ensemble.

### Resultats generals:
ROC Curve (pacients):

<img width="796" height="580" alt="image" src="https://github.com/user-attachments/assets/92c99658-10be-4e4a-97b1-ec181dd46543" />

Confussion Matrix:

<img width="642" height="581" alt="image" src="https://github.com/user-attachments/assets/8c3b937b-3829-4533-9062-2464cae90cfd" />

Mètriques:

<img width="1169" height="305" alt="image" src="https://github.com/user-attachments/assets/47870845-4987-4ee0-b550-3927980bef97" />


### Conclusions

L’AutoEncoder funciona, però és limitat i depèn molt de la mètrica de reconstrucció.

El sistema de ResNet + MIL és molt més estable i dona millors resultats en tots els aspectes.

En un entorn real, el System 2 seria el més recomanable.

### Requisits

Python 3.10+

PyTorch

NumPy

scikit-learn

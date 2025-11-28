#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/maed02/proj_repte3/train
#SBATCH -t 2-00:00
#SBATCH -p tfg
#SBATCH --mem 16000
#SBATCH -o /fhome/maed02/proj_repte3/ResNetClassifier_%u_%j.out
#SBATCH -e /fhome/maed02/proj_repte3/ResNetClassifier_%u_%j.err
#SBATCH --gres gpu:1

echo "Activant entorn virtual..."
source /fhome/maed02/proj_repte3/EntornVirtual/bin/activate

cd /fhome/maed02/proj_repte3/train/
echo "Iniciant ENTRENAMENT CLASSIFIER RESNET utilitzant embedding .pth..."

python - << 'EOF'
from train_system2_resnet_classifier import train_resnet_classifier
train_resnet_classifier()
EOF

echo "Entrenament CLASSIFIER ResNet finalitzat."

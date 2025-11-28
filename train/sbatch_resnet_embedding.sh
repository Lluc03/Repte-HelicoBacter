#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/maed02/proj_repte3/train
#SBATCH -t 4-00:00
#SBATCH -p tfg
#SBATCH --mem 16000
#SBATCH -o /fhome/maed02/proj_repte3/ResNetEmbedding_%u_%j.out
#SBATCH -e /fhome/maed02/proj_repte3/ResNetEmbedding_%u_%j.err
#SBATCH --gres gpu:1

echo "Activant entorn virtual..."
source /fhome/maed02/proj_repte3/EntornVirtual/bin/activate

cd /fhome/maed02/proj_repte3/train/
echo "Iniciant ENTRENAMENT EMBEDDING RESNET..."

python - << 'EOF'
from train_system2_resnet_embedding import train_resnet_embedding
train_resnet_embedding()
EOF

echo "Entrenament d'EMBEDDING ResNet finalitzat."

#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/maed02/proj_repte3/train
#SBATCH -t 2-00:00             # Fins a 2 dies, per si tens many patches
#SBATCH -p tfg
#SBATCH --mem 60000            # 32 GB RAM, important per manejar X_feat gran
#SBATCH -o /fhome/maed02/proj_repte3/extract_feats_%u_%j.out
#SBATCH -e /fhome/maed02/proj_repte3/extract_feats_%u_%j.err
#SBATCH --gres gpu:1           # GPU per processar ResNet

echo "Activant entorn virtual..."
source /fhome/maed02/proj_repte3/EntornVirtual/bin/activate

cd /fhome/maed02/proj_repte3/feature_extractor
echo "Iniciant extracció de features ResNet..."
python -u extract_resnet_patch_features.py
echo "Extracció completada correctament."

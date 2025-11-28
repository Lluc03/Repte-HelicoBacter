#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/maed02/proj_repte3/evaluation
#SBATCH -t 1-00:00
#SBATCH -p tfg
#SBATCH --mem 16000
#SBATCH -o /fhome/maed02/proj_repte3/ROC_%u_%j.out
#SBATCH -e /fhome/maed02/proj_repte3/ROC_%u_%j.err
#SBATCH --gres gpu:1

echo "Activant entorn virtual..."
source /fhome/maed02/proj_repte3/EntornVirtual/bin/activate

cd /fhome/maed02/proj_repte3/evaluation
echo "Generant ROC Curve (System2)..."

python -u system2_roc_curve_patches.py

echo "ROC generada correctament."

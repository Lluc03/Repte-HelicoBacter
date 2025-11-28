#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/maed02/proj_repte3/train
#SBATCH -t 1-00:00                # Normalmente ROC no necesita 4 días
#SBATCH -p tfg
#SBATCH --mem 16000
#SBATCH -o /fhome/maed02/proj_repte3/ROC_%u_%j.out
#SBATCH -e /fhome/maed02/proj_repte3/ROC_%u_%j.err
#SBATCH --gres gpu:1

echo "Activant entorn virtual..."
source /fhome/maed02/proj_repte3/EntornVirtual/bin/activate

cd /fhome/maed02/proj_repte3/evaluation
echo "Generant ROC Curve..."
python roc_curve_patients.py
echo "ROC generada correctament."

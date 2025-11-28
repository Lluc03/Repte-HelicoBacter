#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/maed02/proj_repte3/train
#SBATCH -t 4-00:00
#SBATCH -p tfg
#SBATCH --mem 16000
#SBATCH -o /fhome/maed02/proj_repte3/AEtrain_%u_%j.out
#SBATCH -e /fhome/maed02/proj_repte3/AEtrain_%u_%j.err
#SBATCH --gres gpu:1

echo "Activant entorn virtual..."
source /fhome/maed02/proj_repte3/EntornVirtual/bin/activate

cd /fhome/maed02/proj_repte3/train/
echo "Iniciant entrenament AutoEncoder..."
python autoencoder.py
echo "Entrenament finalitzat."

#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -D /fhome/maed02/proj_repte3/test
#SBATCH -t 1-00:00
#SBATCH -p tfg
#SBATCH --mem 8000
#SBATCH -o /fhome/maed02/proj_repte3/test/testAE_%u_%j.out
#SBATCH -e /fhome/maed02/proj_repte3/test/testAE_%u_%j.err
#SBATCH --gres gpu:1

echo "Activant entorn virtual..."
source /fhome/maed02/proj_repte3/EntornVirtual/bin/activate

echo "Executant test de reconstrucció AutoEncoder..."
python test_autoencoder.py

echo "Test finalitzat correctament."

#!/bin/bash
#SBATCH -n 4
#SBATCH -t 0-06:00
#SBATCH -p tfg
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH -o /fhome/maed02/proj_repte3/pipeline_%j.out
#SBATCH -e /fhome/maed02/proj_repte3/pipeline_%j.err

source ~/proj_repte3/EntornVirtual/bin/activate
python pipeline_complet.py

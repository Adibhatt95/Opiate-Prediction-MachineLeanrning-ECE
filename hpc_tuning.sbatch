#!/bin/bash
#
#SBATCH --job-name=Remove_Repeated_Data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1

module purge
module load pytorch/python3.6/0.2.0_3
source py3.6.3/bin/activate
python apt321_apb462_Project_Notebook_Hyperparameter_Testing.py
#!/bin/bash
#SBATCH --job-name=TF
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10-20:00:00
#SBATCH --partition=blackboxml
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --mail-user=yungdexiong@gmail.com
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --error=TF.err
#SBATCH --output=TF.out

source /ubc/cs/research/plai-scratch/BlackBoxML/TF1_14_0_VAE/run-train.sh

#!/bin/bash

#SBATCH --job-name='truncation_report'
#SBATCH --partition=gpushort
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl

# export WANDB_DISABLED=true  # for some reason this is necessary

module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

logfile='/data/pg-macocu/MT_vs_HT/experiments/truncation_report.out'

cd $HOME/HT-vs-MT/
    python truncation_report.py \
    &> $logfile
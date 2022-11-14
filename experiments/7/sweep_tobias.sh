#!/bin/bash

#SBATCH --job-name='7_train'
#SBATCH --partition=gpushort
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl


# export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
# export WANDB_DISABLED=true  # for some reason this is necessary

EXP_ID=7
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

# wandb sweep --project HT-vs-MT-7 sweep_tobias_params.yaml
wandb agent --count 1 malina03/HT-vs-MT-7/tguqnel3
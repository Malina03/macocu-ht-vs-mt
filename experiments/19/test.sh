#!/bin/bash

#SBATCH --job-name='19_eval'
#SBATCH --partition=gpushort
#SBATCH --time=01:00:00
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


EXP_ID=19
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}

arch="microsoft/deberta-v3-large"
arch_folder="deberta"
learning_rate=1e-05
bsz=32
trained_on="google"
eval_sets=("zh-en" "de-en" "ru-en")
seeds=(1 2 3)

cd $HOME/HT-vs-MT/

for seed in ${seeds[@]}; do    
    checkpoint="${ROOT_DIR}/models/${trained_on}/${arch_folder}_${seed}/checkpoint-*"

    for eval_on in ${eval_sets[@]}; do

        logdir="${ROOT_DIR}/results/${trained_on}/test/${eval_on}/"
        logfile="${logdir}/eval_${seed}.out"
        mkdir -p $logdir
        
        python classifier_trf_hf.py \
        --root_dir $ROOT_DIR \
        --batch_size 16 \
        --arch $arch \
        --mt $trained_on \
        --test
        --test_folder ${eval_on} \
        --load_model $checkpoint \
        &> $logfile
    done
done

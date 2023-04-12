#!/bin/bash

#SBATCH --job-name='40_eval'
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl

# export WANDB_DISABLED=true  # for some reason this is necessary


exp_id=40
ROOT_DIR=/scratch/hb-macocu/MT_vs_HT/experiments/${exp_id}

module purge
module load Python/3.9.6-GCCcore-11.2.0
source /home1/$USER/.envs/macocu/bin/activate

# Hyper-parameters
arch="microsoft/mdeberta-v3-base"
learning_rate=1e-05
bsz=1
max_length=3072
gradient_accumulation_steps=8

trained_on="all"
test_set="test"
eval_sets=("google" "opus" "deepl")
languages=("de" "zh" "ru")
arch_folder="mdeberta_balanced_ht"
seeds=(1 2 3)

cd $HOME/HT-vs-MT/
for seed in ${seeds[@]}; do    
    checkpoint="${ROOT_DIR}/models/${trained_on}/${arch_folder}_${seed}/checkpoint-*"
    for eval_on in ${eval_sets[@]}; do
        for language in ${languages[@]}; do
            logdir="${ROOT_DIR}/results/${arch_folder}/test/${language}-${eval_on}/"
            logfile="${logdir}/eval_${seed}.out"
            mkdir -p $logdir
            
            python classifier_trf_hf.py \
            --root_dir $ROOT_DIR \
            --max_length $max_length \
            --batch_size $bsz \
            --arch $arch \
            --mt ${trained_on} \
            --test "${language}-${eval_on}" \
            --load_model $checkpoint \
            --load_sentence_pairs "multilingual" \
            &> $logfile
        done
    done
done

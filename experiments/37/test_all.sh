#!/bin/bash

#SBATCH --job-name='37_eval'
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
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


EXP_ID=37
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}


arch="microsoft/deberta-v3-large"

learning_rate=1e-05
bsz=32
trained_on="all"
test_set="test"
eval_sets=("google" "opus" "deepl")
arch_folders=("deberta_unbalanced" "deberta_balanced_mt" "deberta_balanced_ht")
seeds=(1 2 3)

cd $HOME/HT-vs-MT/

for arch_folder in ${arch_folders[@]}; do
    for seed in ${seeds[@]}; do    
        checkpoint="${ROOT_DIR}/models/${trained_on}/${arch_folder}_${seed}/checkpoint-*"

        for eval_on in ${eval_sets[@]}; do

            logdir="${ROOT_DIR}/results/${arch_folder}/test/${eval_on}/"
            logfile="${logdir}/eval_${seed}.out"
            mkdir -p $logdir
            
            python classifier_trf_hf.py \
            --root_dir $ROOT_DIR \
            --batch_size 16 \
            --arch $arch \
            --mt ${trained_on} \
            --test ${eval_on} \
            --load_model $checkpoint \
            &> $logfile
        done
    done
done


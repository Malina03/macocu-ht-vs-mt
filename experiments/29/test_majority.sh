#!/bin/bash

#SBATCH --job-name='29_maj'
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
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

EXP_ID=29
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}


# Hyper-parameters
arch="microsoft/mdeberta-v3-base"
arch_folder="mdeberta"
trained_on="google"
# trained_on="deepl"
test_set="test"
eval_sets=("zh-en" "de-en" "ru-en")
seeds=(1 2 3)

cd $HOME/HT-vs-MT/
for seed in ${seeds[@]}; do
 
    checkpoint="/data/pg-macocu/MT_vs_HT/experiments/22/models/${trained_on}/${arch_folder}_${seed}/checkpoint-*"

    for eval_on in ${eval_sets[@]}; do

        logdir="${ROOT_DIR}/results/${trained_on}/test/${eval_on}/"
        logfile="${logdir}/eval_majority_${seed}.out"
        mkdir -p $logdir


        python classifier_trf_hf.py \
        --root_dir $ROOT_DIR \
        --batch_size 8 \
        --arch $arch \
        --mt $trained_on \
        --load_model $checkpoint \
        --load_sentence_pairs\
        --use_majority_classification \
        --test_folder $eval_on \
        --test \
        &> $logfile
    done
done

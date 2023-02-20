#!/bin/bash

#SBATCH --job-name='29_eval'
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

EXP_ID=29
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}


# Hyper-parameters
arch="microsoft/mdeberta-v3-base"
arch_folder="mdeberta"
trained_on="google"
# trained_on="deepl"
eval_sets=("zh" "de" "ru")
# seeds=(1 2 3 4 5 6 7 8 9 10)
seed=1
truncation_vals=(768 1024 2048)

cd $HOME/HT-vs-MT/
for truncation in ${truncation_vals[@]}; do
    checkpoint="${ROOT_DIR}/models/${trained_on}/${arch_folder}_${seed}_${truncation}/checkpoint-*"

    for eval_on in ${eval_sets[@]}; do

        logdir="${ROOT_DIR}/results/${trained_on}/test/${eval_on}/"
        logfile="${logdir}/eval_${seed}_${truncation}.out"
        mkdir -p $logdir

        if [ $trained_on == "google" ]; then
            flags="--use_google_data"
        else
            flags=""
        fi
        
        python classifier_trf_hf.py \
        --root_dir $ROOT_DIR \
        --batch_size 8 \
        --arch $arch \
        --load_model $checkpoint \
        --load_sentence_pairs "multilingual" \
        --test $eval_on \
        --max_length 512 \
        $flags \
        &> $logfile
    done
done

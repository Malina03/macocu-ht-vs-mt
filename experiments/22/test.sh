#!/bin/bash

#SBATCH --job-name='21_eval'
#SBATCH --partition=gpushort
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl

# export WANDB_DISABLED=true  # for some reason this is necessary


exp_id=22
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

# Hyper-parameters
arch="microsoft/mdeberta-v3-base"

learning_rate=5e-05
bsz=32
num_epochs=10
weight_decay=0
max_grad_norm=1
warmup_steps=400
label_smoothing=0.0
dropout=0.1

arch_folder='mdeberta'
trained_on="google"
eval_sets=("zh" "de" "ru")
# seeds=(1 2 3)
seeds=(1)

cd $HOME/HT-vs-MT/
for seed in ${seeds[@]}; do
    if [ $seed == 1 ]; then 
        ckpt=4260
    fi
    if [ $seed == 2 ]; then
        ckpt=8520
    fi
    if [ $seed == 3 ]; then
        ckpt=3408
    fi
    checkpoint="${ROOT_DIR}/models/${trained_on}/${arch_folder}_${seed}/checkpoint-${ckpt}"

    for eval_on in ${eval_sets[@]}; do

        logdir="${ROOT_DIR}/results/${trained_on}/test/${eval_on}/"
        logfile="${logdir}/eval_${seed}.out"
        mkdir -p $logdir

        if [ $trained_on == "google" ]; then
            flags="--use_google_data"
        else
            flags=""
        fi
        
        python classifier_trf_hf.py \
        --root_dir $ROOT_DIR \
        --batch_size 16 \
        --arch $arch \
        --load_model $checkpoint \
        --load_sentence_pairs "multilingual" \
        --test $eval_on \
        $flags \
        $test_flags \
        &> $logfile
    done
done

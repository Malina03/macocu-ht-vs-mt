#!/bin/bash

#SBATCH --job-name='33_pred'
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


EXP_ID=33
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}

# arch="microsoft/mdeberta-v3-base"
# arch_folder="mdeberta"

arch="microsoft/deberta-v3-large"
arch_folder="deberta_ft"
learning_rate=1e-05
bsz=8
trained_on="google"

test_set="test"
eval_sets=("zh" "de" "ru")
seeds=(1 2 3 4 5 6 7 8 9 10)
# seeds=(1)

cd $HOME/HT-vs-MT/

for seed in ${seeds[@]}; do

    checkpoint="${ROOT_DIR}/models/${trained_on}/${arch_folder}_${seed}/checkpoint-*"
    for eval_on in ${eval_sets[@]}; do

        # logdir="${ROOT_DIR}/results/${trained_on}/${test_set}/${eval_on}/"
        logdir="${ROOT_DIR}/results/${trained_on}/${test_set}/${eval_on}/"
        logfile="${logdir}/eval_${seed}.out"
        pred_file="${logdir}/predictions_${seed}.txt"
        mkdir -p $logdir

        if [ $trained_on == "google" ]; then
            flags="--use_google_data"
        else
            flags=""
        fi

        if [ $test_set == "dev" ]; then
            test_flags="--eval"
        else
            test_flags="--test $trained_on"
        fi
        
        python classifier_trf_hf.py \
        --root_dir $ROOT_DIR \
        --predict \
        --prediction_file $pred_file \
        --batch_size 8 \
        --arch $arch \
        --test_on_language ${eval_on} \
        --load_model $checkpoint \
        --max_length 512 \
        $flags \
        $test_flags \
        &> $logfile
    done
done
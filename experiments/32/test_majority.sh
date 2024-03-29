#!/bin/bash

#SBATCH --job-name='32_maj'
#SBATCH --partition=gpushort
#SBATCH --time=01:30:00
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

EXP_ID=32
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}


# Hyper-parameters
arch="microsoft/deberta-v3-large"
arch_folder="deberta"
trained_on="google"
# trained_on="deepl"
test_set="test"
# eval_sets=("zh" "de" "ru")
eval_sets=("de" "ru")
seeds=(1 2 3 4 5 6 7 8 9 10)

cd $HOME/HT-vs-MT/
for seed in ${seeds[@]}; do

    checkpoint="/data/pg-macocu/MT_vs_HT/experiments/21/models/${trained_on}/${arch_folder}_${seed}/checkpoint-*"

    for eval_on in ${eval_sets[@]}; do

        logdir="${ROOT_DIR}/results/${trained_on}/test/${eval_on}/"
        # logdir="/data/$USER/MT_vs_HT/experiments/${EXP_ID}/results/${trained_on}/${test_set}/${eval_on}/"
        logfile="${logdir}/eval_majority_${seed}.out"
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
        --batch_size 8 \
        --arch $arch \
        --load_model $checkpoint \
        --use_majority_classification \
        --test_on_language $eval_on \
        $flags \
        $test_flags \
        &> $logfile
    done
done

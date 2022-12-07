#!/bin/bash

#SBATCH --job-name='14_eval_all'
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


EXP_ID=14
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}

arch="microsoft/mdeberta-v3-base"
arch_folder="mdeberta"

models=("google" "deepl")
sets=("dev" "test")
eval_sets=("google" "deepl")


cd $HOME/HT-vs-MT/
for trained_on in ${models[@]}; do
    for test_set in ${sets[@]}; do
        for eval_on in ${eval_sets[@]}; do

            if [ $trained_on == "google" ]; then
                ckpt="2580"
            else
                ckpt="5155"
            fi
            checkpoint=${ROOT_DIR}/models/${trained_on}/${arch_folder}/checkpoint-${ckpt}

            logdir="${ROOT_DIR}/results/${trained_on}/${test_set}/${eval_on}/"
            logfile="${logdir}/eval.out"
            mkdir -p $logdir

            if [ $eval_on == "google" ]; then
                flags="--use_google_data"
            else
                flags=""
            fi

            if [ $test_set == "dev" ]; then
                test_flags=""
            else
                test_flags="--test $eval_on"
            fi
            
            python classifier_trf_hf.py \
            --root_dir $ROOT_DIR \
            --arch $arch \
            --test $test \
            --load_model $checkpoint \
            $flags \
            $test_flags \
            &> $logfile
        done
    done
done
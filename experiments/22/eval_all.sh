#!/bin/bash

#SBATCH --job-name='22_eval_all'
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


EXP_ID=22
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}

arch="microsoft/mdeberta-v3-base"
arch_folder="mdeberta"

# arch="microsoft/deberta-v3-large"
# arch_folder="deberta"

models=("google" "deepl" "opus")
# sets=("dev" "test")
sets=("test")
# eval_sets=("google" "deepl" "opus")
eval_sets=("opus")
seeds=(1 2 3)


cd $HOME/HT-vs-MT/
for trained_on in ${models[@]}; do
    for test_set in ${sets[@]}; do
        for eval_on in ${eval_sets[@]}; do
            for seed in ${seeds[@]}; do    

                checkpoint="${ROOT_DIR}/models/${trained_on}/${arch_folder}_${seed}/checkpoint-*"
                logdir="${ROOT_DIR}/results/${arch_folder}/${trained_on}/${test_set}/${eval_on}/"
                logfile="${logdir}/eval_${seed}.out"
                mkdir -p $logdir


                if [ $test_set == "test" ]; then
                    test_flags="--test de"
                else
                    test_flags="--eval"
                fi
                
                python classifier_trf_hf.py \
                --root_dir $ROOT_DIR \
                --batch_size 16 \
                --mt $eval_on\
                --arch $arch \
                --load_model $checkpoint \
                --load_sentence_pairs "multilingual" \
                $test_flags \
                &> $logfile
            done
        done
    done
done
#!/bin/bash

#SBATCH --job-name='29_eval'
#SBATCH --partition=gpu
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
arch_folder="mdeberta_ft"
trained_on="google"
# trained_on="deepl"
eval_sets=("zh" "de" "ru")
seeds=(1 2 3 4 5 6 7 8 9 10)

cd $HOME/HT-vs-MT/
for seed in ${seeds[@]}; do
    # if [ $seed == 1 ]; then 
    #     ckpt=1032
    # fi
    # if [ $seed == 2 ]; then
    #     ckpt=2064
    # fi
    # if [ $seed == 3 ]; then
    #     ckpt=4128
    # fi
    checkpoint="${ROOT_DIR}/models/${trained_on}/${arch_folder}_${seed}/checkpoint-*"

    for eval_on in ${eval_sets[@]}; do

        logdir="${ROOT_DIR}/results/${trained_on}/test/${eval_on}/"
        # logdir="/data/$USER/MT_vs_HT/experiments/${EXP_ID}/results/${trained_on}/${test_set}/${eval_on}/"
        logfile="${logdir}/eval_ft_${seed}.out"
        mkdir -p $logdir

        if [ $trained_on == "google" ]; then
            flags="--use_google_data"
        else
            flags=""
        fi
        
        python classifier_trf_hf.py \
        --root_dir $ROOT_DIR \
        --output_dir $logdir \
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

#!/bin/bash

#SBATCH --job-name='13_test_all'
#SBATCH --partition=gpushort
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.chichirau@student.rug.nl

export WANDB_DISABLED=true  # for some reason this is necessary

module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate


EXP_ID=13
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}

# trained_on="google"
# trained_on="deepl"
arch="microsoft/deberta-v3-large"
arch_folder="microsoft-deberta-v3-large"

languages=("de" "fi" "gu" "kk" "lt" "ru" "zh" "dv")
sets=("deepl" "google" "wmt1" "wmt2" "wmt3" "wmt4")
models=("google" "deepl")

cd $HOME/HT-vs-MT/
for trained_on in ${models[@]}; do
    if [ $trained_on == "google" ]; then
        ckpt="1600"
    else
        ckpt="1800"
    fi
    checkpoint=${ROOT_DIR}/models/${trained_on}/${arch_folder}_lr=1e-05_bsz=32_epochs=5_seed=1/checkpoint-${ckpt}

    for lang in ${languages[@]}; do
        for test_set in ${sets[@]}; do

            # if [ $test_set == "deepl" && lang == 'fi' ]; then
            #     continue
            # fi

            if [ $test_set == "deepl" && lang == 'gu' ]; then
                continue
            fi  

            if [ $test_set == "deepl" && lang == 'kk' ]; then
                continue
            fi  

            # if [ $test_set == "deepl" && lang == 'lt' ]; then
            #     continue
            # fi 

            # if [ $test_set == "deepl" && lang == 'ru' ]; then
            #     continue
            # fi  

            # if [ $test_set == "deepl" && lang == 'zh' ]; then
            #     continue
            # fi 

            logdir="${ROOT_DIR}/results/${trained_on}/${lang}/${test_set}/"
            logfile="${logdir}/eval.out"
            mkdir -p $logdir

            if [ $mt == "google" ]; then
                flags="--use_google_data"
            else
                flags=""
            fi
            
            python classifier_trf_hf.py \
            --root_dir $ROOT_DIR \
            --arch $arch \
            --test $test_set \
            --test_on_language $lang \
            --load_model $checkpoint \
            &> $logfile

            # let i++
            # let j++
        done
    done
done
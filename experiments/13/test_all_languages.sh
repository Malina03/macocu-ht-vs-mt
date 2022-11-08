#!/bin/bash

#SBATCH --job-name='13_test_all'
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.chichirau@student.rug.nl
#SBATCH --array=1


module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate


EXP_ID=13
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}

trained_on="google"
arch="microsoft/deberta-v3-large"
checkpoint=${ROOT_DIR}/models/${trained_on}/${arch}_lr=1e-05_bsz=32_epochs=5_seed=1/checkpoint-1600

languages=( "de" "fi" "gu" "kk" "lt" "ru" "zh" )
sets=( "google" "wmt1" "wmt2" "wmt3" "wmt4" )
i=0
j=0
len_i=${#languages[@]}
len_j=${#sets[@]}
while [ $i -lt $len_i ];
do 
    lang=${languages[$i]}
    while [ $j -lt $len_j ];
    do
        test_set=${sets[$j]}
        logdir="${ROOT_DIR}/results/${trained_on}/${lang}/${test_set}/"
        logfile="${logdir}/eval.out"
        mkdir -p $logdir

        if [ $mt == "google" ]; then
            flags="--use_google_data"
        else
            flags=""
        fi

        cd $HOME/HT-vs-MT/
        python classifier_trf_hf.py \
        --root_dir $ROOT_DIR \
        --arch $arch \
        --test $test_set \
        --test_on_language $lang \
        --load_model $checkpoint \
        &> $logfile

        let i++
        let j++
    done
done
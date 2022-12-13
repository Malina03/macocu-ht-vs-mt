#!/bin/bash

#SBATCH --job-name='16_eval'
#SBATCH --partition=gpushort
#SBATCH --time=01:00:00
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


EXP_ID=16
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}

# arch="microsoft/mdeberta-v3-base"
# arch_folder="mdeberta"

arch="microsoft/deberta-v3-large"
arch_folder="deberta"
learning_rate=1e-05
bsz=32
ckpt=189
trained_on="google"
test_set="test"
eval_sets=("zh" "de" "ru")
checkpoint="${ROOT_DIR}/models/${trained_on}/${arch_folder}_lr=${learning_rate}_bsz=${bsz}/checkpoint-${ckpt}"

cd $HOME/HT-vs-MT/

for eval_on in ${eval_sets[@]}; do

    logdir="${ROOT_DIR}/results/${trained_on}/${test_set}/${eval_on}/"
    logfile="${logdir}/eval.out"
    mkdir -p $logdir

    if [ $eval_on == "google" ]; then
        flags="--use_google_data"
    else
        flags=""
    fi

    if [ $test_set == "dev" ]; then
        test_flags="--eval"
    else
        test_flags="--test $eval_on"
    fi
    
    python classifier_trf_hf.py \
    --root_dir $ROOT_DIR \
    --batch_size 16 \
    --arch $arch \
    --test_on_language ${eval_on} \
    --load_model $checkpoint \
    $flags \
    $test_flags \
    &> $logfile
done

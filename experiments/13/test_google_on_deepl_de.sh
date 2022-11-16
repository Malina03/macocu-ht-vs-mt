#!/bin/bash

#SBATCH --job-name='13_google_de'
#SBATCH --partition=gpushort
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl


module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

EXP_ID=13
ROOT_DIR=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}


# trained_on="deepl"
trained_on="google"
arch="microsoft/deberta-v3-large"
arch_folder="microsoft-deberta-v3-large"

if [ $trained_on == "google" ]; then
    ckpt="1600"
else
    ckpt="1800"
fi
checkpoint=${ROOT_DIR}/models/${trained_on}/${arch_folder}_lr=1e-05_bsz=32_epochs=5_seed=1/checkpoint-${ckpt}


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
$flags \
&> $logfile
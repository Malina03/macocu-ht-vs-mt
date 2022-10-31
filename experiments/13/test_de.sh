#!/bin/bash

#SBATCH --job-name='13_test_de'
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

lang="de"
test_set="google"
checkpoint=${ROOT_DIR}/models/google/microsoft-deberta-v3-large_lr=1e-05_bsz=32_epochs=5_seed=1/checkpoint-1600
arch="microsoft/deberta-v3-large"
logdir="${ROOT_DIR}/results/${lang}/${test_set}/"
logfile="${logdir}/eval_seed=${SLURM_ARRAY_TASK_ID}.out"
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
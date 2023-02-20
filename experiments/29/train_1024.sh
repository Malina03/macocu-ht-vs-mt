#!/bin/bash

#SBATCH --job-name='29_1024'
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --array=1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=m.chichirau@student.rug.nl


export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
export WANDB_DISABLED=true  # for some reason this is necessary

exp_id=29
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

# Hyper-parameters
arch="microsoft/mdeberta-v3-base"
mt="google"
learning_rate=1e-05
max_length=1024
bsz=8

num_epochs=10
weight_decay=0
max_grad_norm=1
warmup_steps=400
label_smoothing=0.0
dropout=0.1

seed=${SLURM_ARRAY_TASK_ID}

if [ $mt == "google" ]; then
    flags="--use_google_data"
else
    flags=""
fi

log_model_name="mdeberta"

logdir="${root_dir}/models/${mt}/${log_model_name}_${seed}_${max_length}/"
outputdir="${root_dir}/results/${mt}/dev"
logfile="${outputdir}/train_${seed}.out"
mkdir -p $outputdir
mkdir -p $logdir


cd $HOME/HT-vs-MT/
python classifier_trf_hf.py \
--root_dir $root_dir \
--output_dir $logdir \
--arch $arch \
--learning_rate $learning_rate \
--batch_size $bsz \
--num_epochs $num_epochs \
--weight_decay $weight_decay \
--max_grad_norm $max_grad_norm \
--warmup_steps $warmup_steps \
--label_smoothing $label_smoothing \
--dropout $dropout \
--seed $seed \
--strategy "epoch" \
--load_sentence_pairs "multilingual" \
--max_length $max_length \
$flags \
&> $logfile

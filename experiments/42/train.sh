#!/bin/bash

#SBATCH --job-name='42_ht_train'
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=09:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --array=1-3
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=m.chichirau@student.rug.nl


exp_id=42
root_dir=/scratch/hb-macocu/MT_vs_HT/experiments/${exp_id}

module purge
module load Python/3.9.6-GCCcore-11.2.0
source /home1/$USER/.envs/macocu/bin/activate

# Hyper-parameters
arch="microsoft/mdeberta-v3-base"
mt="all"
learning_rate=1e-05
bsz=1
max_length=3072
gradient_accumulation_steps=8
num_epochs=10
weight_decay=0
max_grad_norm=1
warmup_steps=400
label_smoothing=0.0
dropout=0.1

seed=${SLURM_ARRAY_TASK_ID}

# if [ $mt == "google" ]; then
#     flags="--use_google_data"
# else
#     flags=""
# fi

log_model_name="mdeberta_balanced_ht_ft"
arch_folder="mdeberta_balanced_ht"

checkpoint="/scratch/hb-macocu/MT_vs_HT/experiments/40/models/${mt}/${arch_folder}_${seed}/checkpoint-*"

logdir="${root_dir}/models/${mt}/${log_model_name}_${seed}/"
outputdir="${root_dir}/results/${log_model_name}/${mt}/dev"
logfile="${outputdir}/train_${seed}.out"
mkdir -p $outputdir
mkdir -p $logdir


cd $HOME/HT-vs-MT/
python classifier_trf_hf.py \
--root_dir $root_dir \
--output_dir $logdir \
--arch $arch \
--mt $mt \
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
--balance_data "ht" \
--load_model $checkpoint \
--gradient_accumulation_steps $gradient_accumulation_steps \
&> $logfile
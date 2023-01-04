#!/bin/bash

#SBATCH --job-name='14_g'
#SBATCH --partition=gpu
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl
#SBATCH --array=1-3


# export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
# export WANDB_DISABLED=true  # for some reason this is necessary

EXP_ID=7
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}
seed=${SLURM_ARRAY_TASK_ID}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

# Default Hyper-parameters
arch="microsoft/mdeberta-v3-base"
mt="google"

num_epochs=10
weight_decay=0
max_grad_norm=1
warmup_steps=400
label_smoothing=0.0
dropout=0.1


if [ $mt == "google" ]; then
    flags="--use_google_data"
else
    flags=""
fi

learning_rate=5e-05
bsz=32
log_model_name="mdeberta"
# Make sure the logdir specified below corresponds to the directory defined in the
# main() function of the `classifier_trf_hf.py` script!
logdir="/data/pg-macocu/MT_vs_HT/experiments/14/models/${mt}/${log_model_name}_${seed}/"
logfile="/data/pg-macocu/MT_vs_HT/experiments/14/results/${log_model_name}/${mt}/dev/${mt}/train_${seed}.out"
mkdir -p $logdir

# # Copy source code
# mkdir -p $logdir/src
# cp $HOME/HT-vs-MT/classifier_trf_hf.py $logdir/src

# # Copy this script
# cp $(realpath $0) $logdir

python $HOME/HT-vs-MT/classifier_trf_hf.py \
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
--load_sentence_pairs "default" \
--strategy "epoch" \
$flags \
&> $logfile
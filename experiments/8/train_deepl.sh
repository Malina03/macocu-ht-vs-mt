#!/bin/bash

#SBATCH --job-name='8_d_train'
#SBATCH --partition=gpu
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --array=1-3
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=m.chichirau@student.rug.nl

export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
export WANDB_DISABLED=true  # for some reason this is necessary

exp_id=8
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}
seed=${SLURM_ARRAY_TASK_ID}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

# Hyper-parameters
arch="microsoft/deberta-v3-large"
mt="deepl"
learning_rate=1e-05
bsz=32
num_epochs=10
weight_decay=0
max_grad_norm=1
warmup_steps=200
label_smoothing=0.0
dropout=0.1


if [ $mt == "google" ]; then
    flags="--use_google_data"
else
    flags=""
fi

log_model_name="deberta"
# Make sure the logdir specified below corresponds to the directory defined in the
# main() function of the `classifier_trf_hf.py` script!
logdir="${root_dir}/models/${mt}/${log_model_name}_${seed}/"
res_dir="${root_dir}/results/${log_model_name}/${mt}/dev/${mt}/"
logfile="${res_dir}/train_${seed}.out"
mkdir -p $res_dir
mkdir -p $logdir


# Copy source code
# mkdir -p $logdir/src
# cp $HOME/MaCoCu/student_project_mt_ht/classifier_trf_hf.py $logdir/src

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
--strategy "epoch" \
$flags \
&> $logfile

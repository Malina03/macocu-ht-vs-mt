#!/bin/bash

#SBATCH --job-name='13-deepl_train'
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

exp_id=13
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${exp_id}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

# Hyper-parameters
arch="microsoft/deberta-v3-large"
mt="deepl"
learning_rate=1e-05
bsz=32
num_epochs=5
weight_decay=0
max_grad_norm=1
warmup_steps=200
label_smoothing=0.0
dropout=0.1
seed=${SLURM_ARRAY_TASK_ID}


log_model_name=$(echo $arch | sed 's/\//-/g')
# Make sure the logdir specified below corresponds to the directory defined in the
# main() function of the `classifier_trf_hf.py` script!
logdir="${root_dir}/models/${mt}/${log_model_name}_lr=${learning_rate}_bsz=${bsz}_seed=${seed}/"
logfile="${logdir}/train.out"
mkdir -p $logdir


cd $HOME/HT-vs-MT/
python classifier_trf_hf.py \
--root_dir $root_dir \
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
--mt $mt \
&> $logfile

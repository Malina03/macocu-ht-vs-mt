#!/bin/bash

#SBATCH --job-name='7_train'
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl
#SBATCH --array=1


# export TRANSFORMERS_CACHE=/data/pg-macocu/MT_vs_HT/cache/huggingface
# export WANDB_DISABLED=true  # for some reason this is necessary

EXP_ID=7
root_dir=/data/pg-macocu/MT_vs_HT/experiments/${EXP_ID}
seed=${SLURM_ARRAY_TASK_ID}

module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

# Default Hyper-parameters
arch="xlm-roberta-base"
mt="deepl"

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

learning_rates=( 1e-06 1e-05 1e-04 )
batch_sizes=( 16 32 64 )

for learning_rate in ${learning_rates[@]}; do
    for bsz in ${batch_sizes[@]}; do

        log_model_name=$(echo $arch | sed 's/\//-/g')
        # Make sure the logdir specified below corresponds to the directory defined in the
        # main() function of the `classifier_trf_hf.py` script!
        logdir="${root_dir}/models/${mt}/${log_model_name}_lr=${learning_rate}_bsz=${bsz}_seed=${seed}/"
        logfile="${logdir}/train.out"
        mkdir -p $logdir

        # Copy source code
        mkdir -p $logdir/src
        cp $HOME/HT-vs-MT/classifier_trf_hf.py $logdir/src

        # Copy this script
        cp $(realpath $0) $logdir

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
        --load_sentence_pairs \
        --wandb
        --exp-no ${EXP_ID}
        $flags \
        &> $logfile

        #move back
        cd ../../
    done
done
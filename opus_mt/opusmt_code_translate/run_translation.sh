#!/bin/bash

#SBATCH --job-name='opus_mt'
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl


module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

logfile=/data/$USER/opus_mt/translation.log

python $HOME/HT-vs-MT/opus_mt/opusmt_code_translate/run_translation.py > $logfile
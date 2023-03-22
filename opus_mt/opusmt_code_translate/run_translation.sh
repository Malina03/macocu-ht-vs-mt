#!/bin/bash

#SBATCH --job-name='opus_mt'
#SBATCH --partition=gpushort
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks 1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl


module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

logfile="/data/s3412768/opus_mt/translation.out"

cd $HOME/HT-vs-MT/opus_mt/opusmt_code_translate

python -u translate.py > $logfile
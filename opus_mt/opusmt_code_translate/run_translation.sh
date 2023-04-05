#!/bin/bash

#SBATCH --job-name='opus_mt'
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl


module purge
module load Python/3.9.6-GCCcore-11.2.0
source /home1/$USER/.envs/macocu/bin/activate

logfile="/scratch/s3412768/opus_mt/translation.out"

cd /home1/s3412768/HT-vs-MT/opus_mt/opusmt_code_translate

python translate.py > $logfile
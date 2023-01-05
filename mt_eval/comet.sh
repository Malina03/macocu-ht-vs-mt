#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.chichirau@student.rug.nl

# Evaluate COMET on WMT08-19.

set -euxo pipefail

module purge
module load Python/3.8.6-GCCcore-10.2.0
source /data/$USER/.envs/macocu/bin/activate

# languages=("de" "fi" "gu" "kk" "lt" "ru" "zh" "dv")
languages=("de")
sets=("deepl" "google" "wmt1" "wmt2" "wmt3" "wmt4")

for lang in ${languages[@]}; do
    for test_set in ${sets[@]}; do
        # out=$TMPDIR/out.txt
        # ref=$TMPDIR/ref.txt
        # src=$TMPDIR/src.txt
        dir=/data/pg-macocu/MT_vs_HT/experiments/comet/data/${lang}-en/${test_set}
        if [[ $test_set = "deepl" ]]; then
            out="${dir}/*.deepl.en"
        elif [[ $test_set = "google" ]]; then
            out="${dir}/*.en.google"
        elif [[ $test_set =  wmt{1..4} ]]; then
            out="${dir}/*.wmt"
        else
            echo "$test_set and $lang combination is not a valid."
            exit 1
        fi

        ref="${dir}/trans_*.txt"
        src="${dir}/org_*.txt"
        # echo ${ref} ${src} ${out}
        comet-score -s $src -t $out -r $ref > /data/pg-macocu/MT_vs_HT/experiments/comet/results/${test_set}.${lang}.comet
        # rm -r $TMPDIR/*
    done
done

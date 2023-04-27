#!/bin/bash

# Common variables used in other scripts

MOSESSCRIPTS=$HOME/software/moses-scripts

MBD=/data/pg-wmt20/third/moses-scripts/generic/multi-bleu-detok.perl

MARIAN=$HOME/software/git/marian_2003/marian/build
MARIAN_TRAIN=$MARIAN/marian
MARIAN_DECODER=$MARIAN/marian-decoder
MARIAN_VOCAB=$MARIAN/marian-vocab
MARIAN_SCORER=$MARIAN/marian-score

#SPM=$HOME/software/git/vcpkg/buildtrees/sentencepiece/x64-linux-dbg/src
SPM=$HOME/data/Software/git/sentencepiece/build/src

WORKDIR=/data/p278972/exps/alitra/ennl/ennl_marian_spm/
DATADIR=~/data/alitra/ennl/datasets/201230/
DATADIR2=~/data/alitra/ennl/datasets/210911/
DATADIR3=~/data/alitra/ennl/paracrawl/
DATADIR32=~/data/alitra/ennl/datasets/220526/

function spm_ini {
	module purge
	module load git/2.23.0-GCCcore-8.3.0-nodocs
}

function marian_ini {
        module purge
        module load Boost/1.66.0-intel-2018a-Python-3.6.4
#        module load CMake/3.15.3-GCCcore-8.3.0 #(probably this was needed only for compilation)
        module load CUDA/10.1.243-GCC-8.3.0 
}

function helsinki_ini {
	module purge
	module load CUDA/11.3.1

        source ~/data/Software/anaconda3/etc/profile.d/conda.sh
        conda activate trf
}


# FROM RIK 20220823
# Conda env - needs python version specified or it could error

conda create -n trf python=3.8
conda activate trf

# Load Peregrine CUDA

module load CUDA/11.3.1

# Install pytorch for CUDA 11.3 (takes some time)

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# To see if it works (if you're on pg-gpu)
python
import torch
torch.cuda.is_available()


# Install transformers

pip install transformers


# Install additional huggingface-related pkgs

pip install datasets
#Installing collected packages: pytz, xxhash, python-dateutil, pyarrow, multidict, fsspec, frozenlist, dill, attrs, async-timeout, yarl, responses, pandas, multiprocess, aiosignal, aiohttp, datasets
#Successfully installed aiohttp-3.8.1 aiosignal-1.2.0 async-timeout-4.0.2 attrs-22.1.0 datasets-2.4.0 dill-0.3.5.1 frozenlist-1.3.1 fsspec-2022.7.1 multidict-6.0.2 multiprocess-0.70.13 pandas-1.4.3 pyarrow-9.0.0 python-dateutil-2.8.2 pytz-2022.2.1 responses-0.18.0 xxhash-3.0.0 yarl-1.8.

pip install sacremoses
#Installing collected packages: joblib, click, sacremoses
#Successfully installed click-8.1.3 joblib-1.1.0 sacremoses-0.0.53




# IGNORE BELOW!!!

# Notes about the installation of vecalign. Based on those from LWP on 20200404
# 20200826


function transformers_install {
	# Create conda environment
	conda create  --force -y --name transformers python=3.8

	# Activate new environment
	#eval "$(conda shell.bash hook)"
	conda activate transformers

	# Install required packages
	

	conda install -c huggingface transformers
}


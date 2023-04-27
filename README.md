# MT vs. HT

All code, data, experiment scripts and results for the EAMT 2023 paper Automatic Discrimination of Human and Neural
Machine Translation in Multilingual Scenarios" ([link to pdf](tbd)).
This work was done at the University of Groningen as part of the 
[MaCoCu project](https://macocu.eu/).

The code is adapted from the work of Tobias van der Werff ([link to repo](https://github.com/tobiasvanderwerff/HT-vs-MT)), which was used for the EAMT 2022 paper "Automatic
Discrimination of Human and Neural Machine Translation: A Study with Multiple
Pre-Trained Models and Longer Context" ([link to pdf](https://aclanthology.org/2022.eamt-1.19/)).

Most scripts for running experiments are written for the SLURM workload manager, which
is used on the local High Performance Cluster. For the most part, these are simply bash
scripts with some additional SLURM-specific parameters defined at the top of the script.

## Citation
If you want to cite the paper, you can use the following Bibtex entry: 

```
Tbd
```

## How to run
The `classifier_trf_hf.py` script is the main entry point for training a classifier
using a pretrained language model.

In order to train a classifier, first install the dependencies:

```shell
python -m venv env  # create a new virtual environment
source env/bin/activate  # activate the environment
pip install -r requirements.txt  # install the dependencies
```

Then, run the main script. For example:

```shell
python classifier_trf_hf.py --arch microsoft/deberta-v3-large --learning_rate 1e-5 --batch_size 32
```

Many more arguments can be passed than those shown above; if you want to see a list of
all possible arguments, run

```shell
python classifier_trf_hf.py -h
```

## Data Folder Structure

The structure of the data folder is shown below. You need a data folder in each experiment folder. You can train models on different data sets (eg. google, deepl) as part of a single experiment. When training/testing a model, you need to specify a dataset, using the `--mt <dataset>` flag. For instance `--mt google`, google is also the default dataset. Furthermore, if you want to use data from mts other than google, opus and deepl, you will need to modify the accepted values by the argument parser in `utils.py`.

There can be additional subdirectories in the test folder (e.g "de"), such models can be evaluated on multiple test sets in the same experiment. In this case, the `--test_folder {test_subdirectory}` argument must be used along with the name of the test subdirectory (e.g. `--test_folder de`). You will still need to use the `--test` flag to indicate that the model is used for evaluation.

```
|- data
    |- deepl
         |- train
         |- dev
         |- test
            |- de
            |- zh
            |- ru
    |- google
         |- train
         |- dev
         |- test
```

## File Name Conventions

Especially when training/testing bilingual classifiers it is important to adhere to these naming conventions in order to correctlty match the source file with the translations. When training bilingual classifiers, make sure to use the `--load_sentence_pairs` flag. In the case of test files produced by the systems submitted to WMT19. The table also includes the naming conventions for translationese files. They can be placed in the train/dev/test folders along with the regular files and there is no additional flag required to include them. 

|                                    	| Source file             	| Human Translation       	| Machine Translation                                                                                                           	|
|------------------------------------	|-------------------------	|-------------------------	|-------------------------------------------------------------------------------------------------------------------------------	|
| oriinal language = xx                     	| org_xxen_xx_wmt00.txt   	| trans_xxen_en_wmt00.txt 	| org_xxen_xx_wmt00.txt.en.google<br>org_xxen_xx_wmt00.txt.deepl.en<br>org_xxen_xx_wmt00.opus.en<br>org_xxen_xx_wmt00.wmt 	|
| original language = en<br>(translationese) 	| trans_enxx_xx_wmt00.txt 	| org_enxx_en_wmt00.txt   	| trans_enxx_xx_wmt00.txt.en.google<br>trans_enxx_xx_wmt00.txt.deepl.en<br>trans_enxx_xx_wmt00.opus.en                          	|



## Experiments

The folder `experiments` contains example scripts for running the experiments discussed in the paper. You will notice there are more experiments here than in the paper, since several experiments were grouped as a single larger experiment in the paper. This section provides a guide for which experiments scripts correspond to which experiments presented in the paper.

### Experiment 1 - Monolingual classifier tested on different source languages

we analyse the performance of our classifier when testing a target-only model on English translations from a different source language. The corresponding experiment is 13.
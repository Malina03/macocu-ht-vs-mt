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
| original language = xx                     	| org_xxen_xx_wmt00.txt   	| trans_xxen_en_wmt00.txt 	| org_xxen_xx_wmt00.txt.en.google<br>org_xxen_xx_wmt00.txt.deepl.en<br>org_xxen_xx_wmt00.opus.en<br>org_xxen_xx_wmt00.wmt 	|
| original language = en<br>(translationese) 	| trans_enxx_xx_wmt00.txt 	| org_enxx_en_wmt00.txt   	| trans_enxx_xx_wmt00.txt.en.google<br>trans_enxx_xx_wmt00.txt.deepl.en<br>trans_enxx_xx_wmt00.opus.en                          	|



## Experiments

MAKE SURE YOU ARE ON THE MAIN BRANCH. Otherwise, the numbers of the experiments might not make sense.

The folder `experiments` contains example scripts for running the experiments discussed in the paper. You will notice there are more experiments here than in the paper, since several experiments were grouped as a single larger experiment in the paper. This section provides a guide for which experiments scripts correspond to which experiments presented in the paper.


### Experiment 1 - Testing on Translations from Different Source Languages

We analyse the performance of a classifier when testing a target-only model on English translations from a different source language. The corresponding experiment is 13.


### Experiment 2 - Source-only vs Source+Target Classifiers

The corresponding experiment folder is experiment 7, which contains the scripts for training both the monolingual and the bilingual classifiers, and the archived data for training bilingual classifiers. The data for training monolingual classifiers is used in experiment 8. So, the scripts for training monolingual classifiers use the data folder for experiment 8/data.


### Experiment 3 - Cross-system Evaluation

The scripts for training and evaluating the monolingual classifiers are in experiment 8, and the scripts for the bilingual classifiers are in experiment 14.


### Experiment 4 - Training on Multiple Source Languages

Experiments using the monolingual classifier are 15-21, and the ones using the bilingual classifier are 22-28. For each classifier, the data used in each experiment is in the same order as the order of the results presented in Table 5, from the paper. For instance, experiments 15 and 21 only use German for training (first row of the table), experiments 19 and 26 use German and Russian (5th row from table). 


### Experiment 5: Sentence- vs Document-level

The sentence-level results were achieved by evaluating models trained in expriments 15, 21, 22 and 28 using the ``--use_majority_classification`` flag, and indicating a rootdiectory and test folders that contain document-level data.

Experiment 31 and 32 correspond to the monolingual classifier trained on German only and German + Russian + Chinese, respectively. The scripts for fine-tuning sentence-level classifiers are also in these folders. Similarly, expriments 29 and 30 correspond to the bilingual classifier trained on German and German + Russian + Chinese.
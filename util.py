import argparse
from collections import defaultdict
from typing import Optional, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import TrainingArguments, EvalPrediction

import yaml


class HFDataset(torch.utils.data.Dataset):
    """Dataset for using HuggingFace Transformers."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(
    pred: EvalPrediction, idx_to_docid: Optional[Dict[int, int]] = None
):
    labels = pred.label_ids
    # Sometimes the output is a tuple, take first argument then.
    if isinstance(pred.predictions, tuple):
        pred = pred.predictions[0]
    else:
        pred = pred.predictions
    preds = pred.argmax(-1)
    if idx_to_docid is not None:
        # Majority voting: take the most common prediction per document.
        assert len(idx_to_docid) == len(preds), f"{len(idx_to_docid)} vs {len(preds)}"
        docid_to_preds = defaultdict(list)
        docid_to_label = dict()
        for idx, (p, l) in enumerate(zip(preds, labels)):
            docid = idx_to_docid[idx]
            docid_to_preds[docid].append(p)
            docid_to_label[docid] = l
        preds_new = []
        for docid, doc_preds in docid_to_preds.items():
            # Take the majority prediction.
            perc = sum(doc_preds) / len(doc_preds)
            preds_new.append(1 if perc >= 0.5 else 0)
        preds = np.array(preds_new)
        labels = np.array(list(docid_to_label.values()))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def get_training_arguments(args):
    """Load all training arguments here. There are a lot more not specified, check:
    https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py#L72"""

    return TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.strategy,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.use_fp16,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        logging_strategy=args.strategy,
        logging_steps=args.logging_steps,
        save_strategy=args.strategy,
        save_steps=args.save_steps,
        seed=args.seed,
        load_best_model_at_end=(True if args.strategy != 'no' else False),
        label_smoothing_factor=args.label_smoothing,
        log_level="debug",
        metric_for_best_model="accuracy",
        save_total_limit= (1 if args.strategy != 'no' else None)
    )



def _parse_args(parser, config_parser):
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    return args


def parse_args_hf():
    """
    Parse CLI arguments for the script and return them.
    :return: Namespace of parsed CLI arguments.
    """

    # config_parser = argparse.ArgumentParser(
    #     description="Training Configuration", add_help=False
    # )

    parser = argparse.ArgumentParser(
        description="Arguments for running the classifier."
    )

    # parser.add_argument(
    #     "-c",
    #     "--config",
    #     type=str,
    #     metavar="FILE",
    #     default="",
    #     help="Path to the YAML config file specifying the default parameters.",
    # )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="./experiments/8",
        help="Location of the root directory. By default, this is "
        "the data from WMT08-19, without Translationese.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Location of the output directory. If not specified, it will be created from the model name and parameters.",
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Whether to save predictions.",
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        help="Location of the predictions file.",
    )

    parser.add_argument(
        "--load_model",
        type=str,
        help="Initialize training from the model specified at this " "path location.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        help=("Huggingface transformer architecture to use, " "e.g. `bert-base-cased`"),
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "-wd", "--weight_decay", default=0, type=float, help="Weight decay"
    )
    parser.add_argument(
        "-mgn", "--max_grad_norm", default=1, type=float, help="Max grad norm"
    )
    # parser.add_argument("-wr", "--warmup_ratio", default=0.1, type=float,
    #                     help="Ratio of total training steps used for a linear warmup "
    #                          "from 0 to learning_rate.")
    parser.add_argument(
        "-wr",
        "--warmup_steps",
        default=200,
        type=int,
        help="Number of steps used for a linear warmup from 0 to " "learning_rate",
    )
    parser.add_argument(
        "-ls",
        "--label_smoothing",
        default=0.0,
        type=float,
        help="Label smoothing percentage, 0-1",
    )
    parser.add_argument(
        "-dr",
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout applied to the classifier layer",
    )
    parser.add_argument(
        "-str",
        "--strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default="steps",
        help="Strategy for evaluating/saving/logging",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Number of update steps between two evaluations",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1000,
        help="Number of update steps between two logs",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Number of update steps before two checkpoints saves",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--early_stopping_patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument(
        "--use_fp16", action="store_true", help="Use mixed 16-bit precision"
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1e30)
    parser.add_argument(
        "--load_sentence_pairs",
        type=str,
        nargs = '?',
        choices = ["default", "mean_embeddings", "reverse", "multilingual"],
        help="Set this flag to classify HT vs. MT for "
        "source/translation pairs, rather than just "
        "translations.",
    )
    parser.add_argument(
        "--use_google_data",
        action="store_true",
        help="Use Google Translate data instead of DeepL data for train/dev/test.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Report training metrics to weights and biases.",
    )

    parser.add_argument(
        "--exp_no",
        type=int,
        help="Value indicates the experiments number.",
    )

    parser.add_argument(
        "--use_normalized_data",
        action="store_true",
        help="Use translations that have been post-processed by applying "
        "a Moses normalization script to them. Right now only works for "
        "monolingual sentences",
    )
    parser.add_argument(
        "--test_on_language",
        type=str,
        nargs = '?',
        choices = ["de", "fi", "gu", "kk", "lt", "ru", "zh", "dv"],
        help="Test a classifier on bilingual test sets from different lanuages"
        "from the WMT 19 news dataset.",
    )
    parser.add_argument(
        "--use_majority_classification",
        action="store_true",
        help="Make predictions by predicting each segment in a "
        "document and taking the majority prediction. This is "
        "only used for evaluating an already trained "
        "sentence-level model on documents.",
    )
    parser.add_argument(
        "--test",
        type=str,
        # choices=["deepl", "google", "opus","wmt1", "wmt2", "wmt3", "wmt4", "zh", "de", "ru"],
        help="Test a classifier on one of the test sets. For WMT "
        "submissions there are 4 options, originating from the "
        "WMT 19 test sets, for each language."
        "For exact systems: https://www.statmt.org/wmt19/pdf/53/WMT01.pdf "
        "- wmt1: best system"
        "- wmt2: 2nd best system"
        "- wmt3: 2nd worst system"
        "- wmt4: worst system",
    )

    parser.add_argument(
        "--mt",
        type=str,
        default="google",
        choices=["deepl", "google", "opus","all"],
        help="Which dataset to use for training and testing.",
    )

    
    parser.add_argument(
        "--balance_data",
        type=str,
        default="None",
        choices=["ht", "mt"],
        help="When training on 3 mt systems, balance training/dev data. HT means that all mt data is used and ht is added 3x times to compensate.MT means all ht data is used and 1/3 mt is used from each system.",
    )

    parser.add_argument(
        "--eval", action="store_true", help="Evaluate on dev set using a trained model"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random number generator seed."
    )

    # args = _parse_args(parser, config_parser)

    return parser.parse_args()

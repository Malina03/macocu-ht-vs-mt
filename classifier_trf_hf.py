from functools import partial
import sys
from pathlib import Path

import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    EarlyStoppingCallback,
)

from data import load_corpus, load_sentence_pairs, load_sentence_pairs, load_sentence_pairs_balanced_mt, load_corpus_balanced_mt
from util import get_training_arguments, compute_metrics, parse_args_hf, check_required_args




def main():
    """
    Train a model using the Huggingface Trainer API.
    """
    # Get arguments.
    args = parse_args_hf()

    check_required_args(args)

    # Set random seed.
    np.random.seed(args.seed)

    # Set directories.
    if not args.root_dir:
        root_dir = Path("/data/pg-macocu/MT_vs_HT/experiments/").joinpath(args.exp_no)
    else:
        root_dir = Path(args.root_dir)

    model_name = args.arch.replace("/", "-")

    
    eff_bsz = args.gradient_accumulation_steps * args.batch_size

    if not args.output_dir:
        output_dir = (
            root_dir
            / f"models/{args.mt}/{model_name}_lr={args.learning_rate}_bsz={eff_bsz}_epochs={args.num_epochs}_seed={args.seed}/"
        )
    else:
        output_dir = args.output_dir
        

    if args.eval:
        output_dir = Path(output_dir.parent) / (output_dir.name + "_eval")
    elif args.test:
        output_dir = Path(output_dir.parent) / (output_dir.name + "_test")
    args.output_dir = output_dir

    # Load the data.
    idx_to_docid = None
    test_or_dev = "test" if args.test else "dev"

    if args.load_sentence_pairs:
        if args.balance_data == 'mt':
            train_data,_ = load_sentence_pairs_balanced_mt(args, "train")
            eval_data, idx_to_docid = load_sentence_pairs_balanced_mt(args, test_or_dev, split_docs_by_sentence=args.use_majority_classification)
        else:
            train_data,_ = load_sentence_pairs(args, "train")
            eval_data, idx_to_docid = load_sentence_pairs(args, test_or_dev, split_docs_by_sentence=args.use_majority_classification)
    else:  # load only translations (monolingual)
        if args.balance_data == 'mt':
            train_data,_ = load_corpus_balanced_mt(args, "train")
            eval_data, idx_to_docid = load_corpus_balanced_mt(args, test_or_dev, split_docs_by_sentence=args.use_majority_classification)
        else:
            train_data, _ = load_corpus(args, "train")
            eval_data, idx_to_docid = load_corpus(
                args, test_or_dev, split_docs_by_sentence=args.use_majority_classification
            )

    # Load the model.
    if args.load_model is not None:  # start from a trained model
        print(f"Loading model at {args.load_model}")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.load_model, local_files_only=True, max_position_embeddings=args.max_length
            )
    else:
        model_name = args.arch
        print(f"Loading LM: {model_name}")
        config = AutoConfig.from_pretrained(
            model_name, num_labels=2, classifier_dropout=args.dropout, max_position_embeddings=args.max_length
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, local_files_only=False
        )

    # Setup Huggingface training arguments.
    training_args = get_training_arguments(args)

    # For logging purposes.
    print("Generated by command:\npython", " ".join(sys.argv))
    print("Logging training settings\n", training_args)

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
    ]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=partial(compute_metrics, idx_to_docid=idx_to_docid),
        callbacks=callbacks,
    )

    # Start training/evaluation.
    if args.predict:
        if not args.prediction_file:
            prediction_file = Path(output_dir).joinpath("predictions.tsv")
        else:
            prediction_file = Path(args.prediction_file)
        predictions = trainer.predict(test_dataset=eval_data)
        predicted_labels = list(np.argmax(predictions.predictions, axis=1))
        true_labels = list(predictions.label_ids)
        with open(prediction_file, "w+") as f:
            f.write("predicted_label\ttrue_label\n")
            for pred, true in zip(predicted_labels, true_labels):
                    f.write(f"{pred}\t{true}\n")
        print("\nInfo:\n", predictions.metrics, "\n")
     
    elif args.test or args.eval or args.use_majority_classification:
        mets = trainer.evaluate()
        print("\nInfo:\n", mets, "\n")  
    else:
        mets = trainer.train()
        print("\nInfo:\n", mets, "\n")


if __name__ == "__main__":
    main()

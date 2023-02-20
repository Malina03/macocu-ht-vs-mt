import re
import itertools
from pathlib import Path
import os

from transformers import AutoTokenizer


def truncation_monolingual_train(phase, root_dir, use_google_data, split_docs_by_sentence, arch, max_length, use_normalized_data):    
    if phase not in ("train", "dev", "test"):
        raise ValueError("Phase should be one of 'train', 'dev', 'test'")

    print(f"=> Loading {phase} corpus...")

    corpus_data = []
    root_dir = Path(root_dir).resolve()
    mt = mt_name = "google" if use_google_data else "deepl"
    if mt_name.startswith("wmt"):
        mt = "wmt_submissions"
    apdx = "normalized" if use_normalized_data else ""
    paths = {
        0: list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.txt")),
        1: (
            list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.deepl.en"))
            + list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.en.google"))
        ),
    }  # all the text files per class
    if mt_name == "wmt1":
        paths[1] = [
            root_dir
            / f"data/wmt_submissions/{phase}/{apdx}/newstest2019.Facebook_FAIR.6750.wmt"
        ]
    if mt_name == "wmt2":
        paths[1] = [
            root_dir
            / f"data/wmt_submissions/{phase}/{apdx}/newstest2019.RWTH_Aachen_System.6818.wmt"
        ]
    if mt_name == "wmt3":
        paths[1] = [
            root_dir
            / f"data/wmt_submissions/{phase}/{apdx}/newstest2019.online-X.0.wmt"
        ]
    if mt_name == "wmt4":
        paths[1] = [
            root_dir
            / f"data/wmt_submissions/{phase}/{apdx}/newstest2019.PROMT_NMT_DE-EN.6683.wmt"
        ]

    print(f"paths: {paths}")

    assert (
        len(paths[0]) != 0 and len(paths[1]) != 0
    ), f"{len(paths[0])}, {len(paths[1])}"

    idx_to_docid = dict() if split_docs_by_sentence else None
    doc_id = 0
    for label, path_lst in paths.items():
        for path in path_lst:
            with open(path, encoding="utf-8") as corpus:
                for line in corpus:
                    if split_docs_by_sentence:
                        # In this case, a single line contains a full document.
                        for seg in line.split(". "):
                            corpus_data.append([f"{seg.rstrip()}.", label])
                            idx_to_docid[len(corpus_data) - 1] = doc_id
                    else:
                        corpus_data.append([line.rstrip(), label])
                    doc_id += 1
    sents, labels = zip(*corpus_data)
    sents = list(sents)

    # Encode the sentences using the HuggingFace tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        arch, model_max_length=None
    )
    sents_enc = tokenizer(sents, padding=True, truncation=True)
    input_lengths = sorted([len([i for i in seq if i!=0]) for seq in sents_enc.input_ids])
    truncation_percentage = [(l - max_length)/l for l in input_lengths if l > max_length]
    truncated_inputs = sum(i > max_length for i in input_lengths)
    pecentage_truncated = truncated_inputs/len(input_lengths)
    longest_seq = max(input_lengths)
    top_10 = input_lengths[-10:]
    print("Monolingual training data for experiment {}".format(root_dir.split("/")[1]))
    print("Number of truncated docs: {} out of {}, which is {}".format(truncated_inputs, len(input_lengths), pecentage_truncated))
    if len(truncation_percentage) > 0:
        print("truncation percentage: ", sum(truncation_percentage)/len(truncation_percentage))
    print("longest doc: ", longest_seq)
    print("top 10 longest docs: ", top_10)

def truncation_monolingual_testing(phase, root_dir, test_on_language, test, arch, max_length, split_docs_by_sentence):
    if phase != 'test':
            raise ValueError("Phase should be 'test'")
        

    if not test_on_language:
        raise ValueError("A language has to be specified")

    apdx = test_on_language
    apdx_name = apdx + '-en'
    print(f"=> Loading {phase} corpus for {apdx} ...")

    corpus_data = []
    root_dir = Path(root_dir).resolve()

    mt_name = test

    if mt_name.startswith("wmt"):
        mt = "wmt_submissions"
        paths = {
            0: list((root_dir / f"data/{mt}/{phase}/{apdx_name}/{mt_name}/").glob("*.txt")),
            1: (
                list((root_dir / f"data/{mt}/{phase}/{apdx_name}/{mt_name}/").glob("*.wmt"))
            ),
        }  # all the text files per class
    else:
        mt = mt_name
        paths = {
            0: list((root_dir / f"data/{mt}/{phase}/{apdx_name}/").glob("*.txt")),
            1: (
            list((root_dir / f"data/{mt}/{phase}/{apdx_name}").glob("*.deepl.en"))
            + list((root_dir / f"data/{mt}/{phase}/{apdx_name}").glob("*.en.google"))
            ),
        }  # all the text files per class
    
    print(f"paths: {paths}")

    assert (
        len(paths[0]) != 0 and len(paths[1]) != 0
    ), f"{len(paths[0])}, {len(paths[1])}"

    # print(paths[0])
    # print(paths[1])

    idx_to_docid = dict() if split_docs_by_sentence else None
    doc_id = 0
    for label, path_lst in paths.items():
        for path in path_lst:
            with open(path, encoding="utf-8") as corpus:
                for line in corpus:
                    if split_docs_by_sentence:
                        # In this case, a single line contains a full document.
                        for seg in line.split(". "):
                            corpus_data.append([f"{seg.rstrip()}.", label])
                            idx_to_docid[len(corpus_data) - 1] = doc_id
                    else:
                        corpus_data.append([line.rstrip(), label])
                    doc_id += 1
    sents, labels = zip(*corpus_data)
    sents = list(sents)

        # Encode the sentences using the HuggingFace tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        arch, model_max_length=None
    )
    sents_enc = tokenizer(sents, padding=True, truncation=True)
    input_lengths = sorted([len([i for i in seq if i!=0]) for seq in sents_enc.input_ids])
    truncation_percentage = [(l - max_length)/l for l in input_lengths if l > max_length]
    truncated_inputs = sum(i > max_length for i in input_lengths)
    pecentage_truncated = truncated_inputs/len(input_lengths)
    longest_seq = max(input_lengths)
    top_10 = input_lengths[-10:]
    print("Data for {} on {} with {}:".format(phase, test_on_language, mt_name))    
    print("Number of truncated docs: {} out of {}, which is {}".format(truncated_inputs, len(input_lengths), pecentage_truncated))
    if len(truncation_percentage) > 0:
        print("truncation percentage: ", sum(truncation_percentage)/len(truncation_percentage))
    print("longest doc: ", longest_seq)
    print("top 10 longest docs: ", top_10)

def truncation_bilingual(phase, root_dir, use_google_data, test, arch, max_length, split_docs_by_sentence):
   
    if phase not in ("train", "dev", "test"):
            raise ValueError("Phase should be one of 'train', test or 'dev'")
            
    print("=> Loading {} corpus...".format(phase))

    _mt_suffixes = [".txt.en.google", '.deepl.en']

    corpus_data = []
    root_dir = Path(root_dir).resolve()

    mt = "google" if use_google_data else "deepl"
    # if mt != "google":
    #     raise NotImplementedError("Only Google data is supported for now.")

    if phase == "test":
        lang_apdx = test
        paths = {
            # No translationsese data for testing => trans_*.txt only matches ht sentences from original data
            0: list((root_dir / f"data/{mt}/{phase}/{lang_apdx}-en/").glob(f"trans*.txt")),
            1: list(
                itertools.chain.from_iterable(
                    (root_dir / f"data/{mt}/{phase}/{lang_apdx}-en/").glob(f"*{sfx}")
                    for sfx in _mt_suffixes
                )
            ),
        }
    else:
        paths = {
            # adapted for translationese
            0: (list((root_dir / f"data/{mt}/{phase}/").glob("trans_??en_en_wmt??.txt")) + list((root_dir / f"data/{mt}/{phase}/").glob("org_en??_en_wmt??.txt"))),
            1: list(
                itertools.chain.from_iterable(
                    (root_dir / f"data/{mt}/{phase}/").glob(f"*{sfx}")
                    for sfx in _mt_suffixes
                )
            ),
        }

    print(root_dir / f"data/{mt}/{phase}/")
    print(f"paths: {paths}")

    assert len(paths[0]) != 0 and len(paths[1]) != 0

    # Match source files with files containing translations.
    # path_A = original, path_B = translation
    idx_to_docid = dict() if split_docs_by_sentence else None
    doc_id = 0
    for label, path_lst in paths.items():
        for path_B in path_lst:
            wmt_year = re.search(r"[0-9]{2}", path_B.name).group(0)
            lang = re.search(r"(de|ru|zh)", path_B.name).group(1)
            if path_B.name in [
                f"trans_{lang}en_en_wmt{wmt_year}.txt",
                f"org_{lang}en_{lang}_wmt{wmt_year}.deepl.en",
                f"org_{lang}en_{lang}_wmt{wmt_year}.txt.en.google"
            ]:
                # Translation from original text.
                if phase == "test":
                    path_A = root_dir / f"data/{mt}/{phase}/{lang}-en/org_{lang}en_{lang}_wmt{wmt_year}.txt"
                else:
                    path_A = root_dir / f"data/{mt}/{phase}/org_{lang}en_{lang}_wmt{wmt_year}.txt"
            elif path_B.name in [
                f"org_en{lang}_en_wmt{wmt_year}.txt",
                f"trans_en{lang}_{lang}_wmt{wmt_year}.deepl.en",
                f"trans_en{lang}_{lang}_wmt{wmt_year}.txt.en.google"
            ]:
                # Translation from translatinese -> not in test data
                path_A = root_dir / f"data/{mt}/{phase}/trans_en{lang}_{lang}_wmt{wmt_year}.txt"
            else:  # fail
                raise RuntimeError(
                    f"Unrecognized file name: {path_B.name}. Take a look "
                    f"at the file naming convention in "
                    f"load_corpus_sentence_pairs()` to see why this "
                    f"is unrecognized."
                )
            assert path_A.is_file(), (
                f"Sentence pairs incomplete, missing: {path_A.name}. Make "
                f"sure all translated sentences are coupled with "
                f"a corresponding untranslated sentence."
            )
            with open(path_A, encoding="utf-8") as sents_A:
                with open(path_B, encoding="utf-8") as sents_B:
                    if split_docs_by_sentence:
                        # In this case, a single line contains a full document.
                        for line_A, line_B in zip(sents_A, sents_B):
                            for seg_A, seg_B in zip(line_A.split(". "), line_B.split(". ")):
                                corpus_data.append(
                                    [f"{seg_A.rstrip()}.", f"{seg_B.rstrip()}.", label]
                                )
                                idx_to_docid[len(corpus_data) - 1] = doc_id
                            doc_id += 1
                    else:
                        for line_A, line_B in zip(sents_A, sents_B):
                            corpus_data.append([line_A.rstrip(), line_B.rstrip(), label])


    # Encode the sentences using the HuggingFace tokenizer.
    sentsA, sentsB, labels = zip(*corpus_data)
    sentsA, sentsB = list(sentsA), list(sentsB)

    tokenizer = AutoTokenizer.from_pretrained(
            arch, model_max_length=None
        )

    sents_enc = tokenizer(sentsA, sentsB, padding=True, truncation=True)

    input_lengths = sorted([len([i for i in seq if i!=0]) for seq in sents_enc.input_ids])
    truncation_percentage = [(l - max_length)/l for l in input_lengths if l > max_length]
    truncated_inputs = sum(i > max_length for i in input_lengths)
    pecentage_truncated = truncated_inputs/len(input_lengths)
    longest_seq = max(input_lengths)
    top_10 = input_lengths[-10:]
    print("Data for {} on {} with {}".format(phase, test, mt))
    print("Number of truncated docs: {} out of {}, which is {}".format(truncated_inputs, len(input_lengths), pecentage_truncated))
    if len(truncation_percentage) > 0:
        print("truncation percentage: ", sum(truncation_percentage)/len(truncation_percentage))
    print("longest doc: ", longest_seq)
    print("top 10 longest docs: ", top_10)


def main():
    languages = ["de", "ru", "zh"]
    phases = ["train", "dev" "test"]
    models = ['bilingual', 'monolingual']
    truncation_vals = [768, 1024, 2048]
    root_dir_bilingual = Path("/data/pg-macocu/experiments/29/")
    root_dir_monolingual = Path("/data/pg-macocu/experiments/31/")

    for model in models:
        if model == 'bilingual':
            root_dir = root_dir_bilingual
            arch = "microsoft/mdeberta-v3-base"
            for max_length in truncation_vals:
                for phase in phases:
                    if phase == "test":
                        for language in languages:
                            truncation_bilingual(phase, root_dir,True, language,arch, max_length, False)
                    truncation_bilingual(phase, root_dir,True, None, arch, max_length, False)
        else:
            root_dir = root_dir_monolingual
            arch = "microsoft/deberta-v3-large"
            for phase in phases:
                if phase == "test":
                    for language in languages:
                        truncation_monolingual_testing(phase, root_dir, language, "google", arch, max_length, split_docs_by_sentence=False)
                else:
                    truncation_monolingual_train(phase, root_dir, True, False, arch, max_length, False)
                

if __name__ == "__main__":
    main()
import re
import itertools
from pathlib import Path

from transformers import AutoTokenizer

from util import HFDataset




 

def load_language_tests(args, phase, split_docs_by_sentence=False):
    """
    Load sentence-label pairs from disk.

    Args:
        args: arguments as processed by parse_args()
        phase: phase for which data should be loaded
        split_docs_by_sentence: whether to split documents into sentences for the
            purpose of majority classification
    Returns:
        HFDataset returning sentence-label pairs
    """

    if phase != 'test':
        raise ValueError("Phase should be 'test'")
    
    if args.use_normalized_data:
        raise NotImplementedError()

    if not args.test_on_language:
        raise ValueError("A language has to be specified")
    
    apdx = args.test_on_language
    apdx_name = apdx + '-en'
    print(f"=> Loading {phase} corpus for {apdx} ...")

    corpus_data = []
    root_dir = Path(args.root_dir).resolve()

    mt_name = args.test

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
            + list((root_dir / f"data/{mt}/{phase}/{apdx_name}").glob("*.opus.en"))
            ),
        }  # all the text files per class
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
        args.arch, model_max_length=args.max_length
    )
    sents_enc = tokenizer(sents, padding=True, truncation=True)

    return HFDataset(sents_enc, labels), idx_to_docid

def load_corpus_balanced_mt(args, phase, split_docs_by_sentence=False):
    """
    Load sentence-label pairs from disk.

    Args:
        args: arguments as processed by parse_args()
        phase: phase for which data should be loaded
        split_docs_by_sentence: whether to split documents into sentences for the
            purpose of majority classification
    Returns:
        HFDataset returning sentence-label pairs
    """
    if phase not in ("train", "dev"):
        raise ValueError("Phase should be one of 'train', 'dev'")

    print(f"=> Loading {phase} corpus...")

    corpus_data = []
    root_dir = Path(args.root_dir).resolve()
    # mt = mt_name = "google" if args.use_google_data else "deepl"
    mt = mt_name = args.mt if args.mt else "google"


    apdx = "normalized" if args.use_normalized_data else ""
    paths = {
        0: list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.txt")),
        1: (
            list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.deepl.en"))
            + list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.en.google"))
            + list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.opus.en"))
        ),
    }  # all the text files per class
    
    assert (
        len(paths[0]) != 0 and len(paths[1]) != 0
    ), f"{len(paths[0])}, {len(paths[1])}"



    idx_to_docid = dict() if split_docs_by_sentence else None
    doc_id = 0

    google_data = []
    deepl_data = []
    opus_data = []

    paths[1] = sorted(paths[1], key=lambda x: int(re.findall(r"\d+", x.name)[0]))
    # sort paths in paths[0] by paths[1]
    paths[0] = sorted(paths[0], key=lambda x: int(re.findall(r"\d+", x.name)[0]))

    for label, path_lst in paths.items():
        for path in path_lst:
            with open(path, encoding="utf-8") as corpus:
                for line in corpus:
                    if split_docs_by_sentence:
                    # In this case, a single line contains a full document.
                        for seg in line.split(". "):
                            if "google" in path.name:
                                google_data.append([f"{seg.rstrip()}.", label])
                            elif "deepl" in path.name:
                                deepl_data.append([f"{seg.rstrip()}.", label])
                            elif "opus" in path.name:
                                opus_data.append([f"{seg.rstrip()}.", label])
                            else:
                                corpus_data.append([f"{seg.rstrip()}.", label])
                            idx_to_docid[len(corpus_data) - 1] = doc_id
                    else:
                        if "google" in path.name:
                            google_data.append([line.rstrip(), label])
                        elif "deepl" in path.name:
                            deepl_data.append([line.rstrip(), label])
                        elif "opus" in path.name:
                            opus_data.append([line.rstrip(), label])
                        else:
                            corpus_data.append([line.rstrip(), label])
                    doc_id += 1

    # append non-overlapping 1/3 google, 1/3 deepl, 1/3 opus
    google_idx = [i for i in range(0, len(google_data), 3)]  
    deepl_idx = [i for i in range(1, len(deepl_data), 3)]
    opus_idx = [i for i in range(2, len(opus_data), 3)]
    # append data from these indices to corpus_data
    corpus_data += [deepl_data[i] for i in deepl_idx]
    corpus_data += [opus_data[i] for i in opus_idx]
    corpus_data += [google_data[i] for i in google_idx]


    sents, labels = zip(*corpus_data)
    sents = list(sents)

    # Encode the sentences using the HuggingFace tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        args.arch, model_max_length=args.max_length
    )
    sents_enc = tokenizer(sents, padding=True, truncation=True)

    return HFDataset(sents_enc, labels), idx_to_docid



def load_corpus(args, phase, split_docs_by_sentence=False):
    """
    Load sentence-label pairs from disk.

    Args:
        args: arguments as processed by parse_args()
        phase: phase for which data should be loaded
        split_docs_by_sentence: whether to split documents into sentences for the
            purpose of majority classification
    Returns:
        HFDataset returning sentence-label pairs
    """
    if phase not in ("train", "dev", "test"):
        raise ValueError("Phase should be one of 'train', 'dev', 'test'")

    print(f"=> Loading {phase} corpus...")

    corpus_data = []
    root_dir = Path(args.root_dir).resolve()

    mt = args.mt if args.mt else "google"
    apdx = args.test if phase == 'test' else ""

    paths = {
        0: list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.txt")),
        1: (
            list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.deepl.en"))
            + list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.en.google"))
            + list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.opus.en"))
            + list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.wmt1"))
            + list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.wmt2"))
            + list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.wmt3"))
            + list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.wmt4"))
        ),
    }  # all the text files per class


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
                            if args.balance_data == 'ht' and label == 0:
                                # add ht 3 times
                                corpus_data.extend([[f"{seg.rstrip()}.", label]]*2)
                                idx_to_docid.extend([doc_id]*2)
                    else:
                        corpus_data.append([line.rstrip(), label])
                        if args.balance_data == 'ht' and label == 0:
                            corpus_data.extend([[line.rstrip(), label]]*2)
                    doc_id += 1
            
    sents, labels = zip(*corpus_data)
    sents = list(sents)

    # Encode the sentences using the HuggingFace tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        args.arch, model_max_length=args.max_length
    )
    sents_enc = tokenizer(sents, padding=True, truncation=True)

    return HFDataset(sents_enc, labels), idx_to_docid

def load_corpus_multilingual_sentence_pairs(args, phase, split_docs_by_sentence=False):
    """
    Loads data from disk, where instead of individual sentences, bilingual sentence
    pairs are loaded (German-English/Russian-English/Chinese-English).

    Args:
        args: arguments as processed by parse_args()
        phase: phase for which data should be loaded
    Returns:
        HFDataset returning sentence pair/label pairs
    """
    if phase not in ("train", "dev", "test"):
        raise ValueError("Phase should be one of 'train', test or 'dev'")
        
    print("=> Loading {} corpus...".format(phase))

    _mt_suffixes = [".txt.en.google", '.deepl.en', ".opus.en", 'wmt1', 'wmt2', 'wmt3','wmt4']

    corpus_data = []
    root_dir = Path(args.root_dir).resolve()
    
    # mt = "google" if args.use_google_data else "deepl"
    mt = args.mt if args.mt else "google"
    # if mt != "google":
    #     raise NotImplementedError("Only Google data is supported for now.")
    
    if phase == "test":
        test_folder = args.test
        paths = {
            # No translationsese data for testing => trans_*.txt only matches ht sentences from original data
            0: list((root_dir / f"data/{mt}/{phase}/{test_folder}/").glob(f"trans*.txt")),
            1: list(
                itertools.chain.from_iterable(
                    (root_dir / f"data/{mt}/{phase}/{test_folder}/").glob(f"*{sfx}")
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
                f"org_{lang}en_{lang}_wmt{wmt_year}.txt.en.google",
                f"org_{lang}en_{lang}_wmt{wmt_year}.opus.en",
                f"org_{lang}en_{lang}_wmt{wmt_year}.wmt1",
                f"org_{lang}en_{lang}_wmt{wmt_year}.wmt2",
                f"org_{lang}en_{lang}_wmt{wmt_year}.wmt3",
                f"org_{lang}en_{lang}_wmt{wmt_year}.wmt4"
            ]:
                # Translation from original text.
                if phase == "test":
                    path_A = root_dir / f"data/{mt}/{phase}/{test_folder}/org_{lang}en_{lang}_wmt{wmt_year}.txt"
                else:
                    path_A = root_dir / f"data/{mt}/{phase}/org_{lang}en_{lang}_wmt{wmt_year}.txt"
            elif path_B.name in [
                f"org_en{lang}_en_wmt{wmt_year}.txt",
                f"trans_en{lang}_{lang}_wmt{wmt_year}.deepl.en",
                f"trans_en{lang}_{lang}_wmt{wmt_year}.txt.en.google",
                f"trans_en{lang}_{lang}_wmt{wmt_year}.opus.en",
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
                                if args.balance_data == 'ht' and label == 0:
                                    corpus_data.extend([[f"{seg_A.rstrip()}.", f"{seg_B.rstrip()}.", label]]*2)
                                    idx_to_docid.extend([doc_id]*2)
                            doc_id += 1
                    else:
                        for line_A, line_B in zip(sents_A, sents_B):
                            corpus_data.append([line_A.rstrip(), line_B.rstrip(), label])
                            if args.balance_data == 'ht' and label == 0:
                                corpus_data.extend([[line_A.rstrip(), line_B.rstrip(), label]]*2)


    # Encode the sentences using the HuggingFace tokenizer.
    sentsA, sentsB, labels = zip(*corpus_data)
    sentsA, sentsB = list(sentsA), list(sentsB)
    tokenizer = AutoTokenizer.from_pretrained(
        args.arch, model_max_length=args.max_length
    )
    # encode input as translated + original
    if args.load_sentence_pairs == 'reverse':
        sents_enc = tokenizer(sentsB, sentsA, padding=True, truncation=True)
    else:
    # encode input as original + translated
        sents_enc = tokenizer(sentsA, sentsB, padding=True, truncation=True)

    return HFDataset(sents_enc, labels), idx_to_docid


def load_corpus_multilingual_sentence_pairs_balanced_mt(args, phase, split_docs_by_sentence=False):
    """
    Loads data from disk, where instead of individual sentences, bilingual sentence
    pairs are loaded (German-English/Russian-English/Chinese-English).

    Args:
        args: arguments as processed by parse_args()
        phase: phase for which data should be loaded
    Returns:
        HFDataset returning sentence pair/label pairs
    """
    if phase not in ("train", "dev"):
        raise ValueError("Phase should be one of 'train' or 'dev'")
        
    print("=> Loading {} corpus...".format(phase))

    _mt_suffixes = [".txt.en.google", '.deepl.en', ".opus.en"]

    corpus_data = []
    root_dir = Path(args.root_dir).resolve()
    
    # mt = "google" if args.use_google_data else "deepl"
    mt = args.mt if args.mt else "google"
    # if mt != "google":
    #     raise NotImplementedError("Only Google data is supported for now.")
    

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

    assert len(paths[0]) != 0 and len(paths[1]) != 0

    google_data = []
    deepl_data = []
    opus_data = []

    paths[1] = sorted(paths[1], key=lambda x: int(re.findall(r"\d+", x.name)[0]))
    # sort paths in paths[0] by paths[1]
    paths[0] = sorted(paths[0], key=lambda x: int(re.findall(r"\d+", x.name)[0]))

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
                f"org_{lang}en_{lang}_wmt{wmt_year}.txt.en.google",
                f"org_{lang}en_{lang}_wmt{wmt_year}.opus.en"
            ]:
                # Translation from original text.
                if phase == "test":
                    path_A = root_dir / f"data/{mt}/{phase}/{lang}-en/org_{lang}en_{lang}_wmt{wmt_year}.txt"
                else:
                    path_A = root_dir / f"data/{mt}/{phase}/org_{lang}en_{lang}_wmt{wmt_year}.txt"
            elif path_B.name in [
                f"org_en{lang}_en_wmt{wmt_year}.txt",
                f"trans_en{lang}_{lang}_wmt{wmt_year}.deepl.en",
                f"trans_en{lang}_{lang}_wmt{wmt_year}.txt.en.google",
                f"trans_en{lang}_{lang}_wmt{wmt_year}.opus.en",
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
                                if 'google' in path_B.name:
                                    google_data.append([seg_A.rstrip(), seg_B.rstrip(), label])
                                elif 'deepl' in path_B.name:
                                    deepl_data.append([seg_A.rstrip(), seg_B.rstrip(), label])
                                elif 'opus' in path_B.name:
                                    opus_data.append([seg_A.rstrip(), seg_B.rstrip(), label])
                                else:
                                    corpus_data.append(
                                        [f"{seg_A.rstrip()}.", f"{seg_B.rstrip()}.", label]
                                    )
                                idx_to_docid[len(corpus_data) - 1] = doc_id
                            doc_id += 1
                    else:
                        for line_A, line_B in zip(sents_A, sents_B):
                            if 'google' in path_B.name:
                                google_data.append([line_A.rstrip(), line_B.rstrip(), label])
                            elif 'deepl' in path_B.name:
                                deepl_data.append([line_A.rstrip(), line_B.rstrip(), label])
                            elif 'opus' in path_B.name:
                                opus_data.append([line_A.rstrip(), line_B.rstrip(), label])
                            else:
                                corpus_data.append([line_A.rstrip(), line_B.rstrip(), label])

    google_idx = [i for i in range(0, len(google_data), 3)]  
    deepl_idx = [i for i in range(1, len(deepl_data), 3)]
    opus_idx = [i for i in range(2, len(opus_data), 3)]
    # append data from these indices to corpus_data
    corpus_data += [deepl_data[i] for i in deepl_idx]
    corpus_data += [opus_data[i] for i in opus_idx]
    corpus_data += [google_data[i] for i in google_idx]         

    # Encode the sentences using the HuggingFace tokenizer.
    sentsA, sentsB, labels = zip(*corpus_data)
    sentsA, sentsB = list(sentsA), list(sentsB)
    tokenizer = AutoTokenizer.from_pretrained(
        args.arch, model_max_length=args.max_length
    )
    # encode input as translated + original
    if args.load_sentence_pairs == 'reverse':
        sents_enc = tokenizer(sentsB, sentsA, padding=True, truncation=True)
    else:
    # encode input as original + translated
        sents_enc = tokenizer(sentsA, sentsB, padding=True, truncation=True)

    return HFDataset(sents_enc, labels), idx_to_docid




def load_corpus_sentence_pairs(args, phase):
    """
    Loads data from disk, where instead of individual sentences, bilingual sentence
    pairs are loaded (German-English).

    Args:
        args: arguments as processed by parse_args()
        phase: phase for which data should be loaded
    Returns:
        HFDataset returning sentence pair/label pairs
    """
    if phase not in ("train", "dev", "test"):
        raise ValueError("Phase should be one of 'train', 'dev', 'test'")

    print("=> Loading {} corpus...".format(phase))

    _mt_suffixes = [".deepl.en", ".txt.en.google", ".wmt", ".opus.en"]

    corpus_data = []
    root_dir = Path(args.root_dir).resolve()
    # mt = "google" if args.use_google_data else "deepl"
    mt = args.mt if args.mt else "google"
    if phase == "test":
        mt = args.test
    paths = {
        0: list((root_dir / f"data/{mt}/{phase}/").glob("trans_en*.txt")),
        1: list(
            itertools.chain.from_iterable(
                (root_dir / f"data/{mt}/{phase}/").glob(f"*{sfx}")
                for sfx in _mt_suffixes
            )
        ),
    }
    if mt == "wmt_submissions":
        raise NotImplementedError()

    assert len(paths[0]) != 0 and len(paths[1]) != 0

    # Match source files with files containing translations.
    # path_A = original, path_B = translation
    for label, path_lst in paths.items():
        for path_B in path_lst:
            wmt_year = re.search(r"[0-9]{2}", path_B.name).group(0)
            if path_B.name in [
                f"trans_en_wmt{wmt_year}.txt",
                f"org_de_wmt{wmt_year}.deepl.en",
                f"org_de_wmt{wmt_year}.txt.en.google",
                f"org_de_wmt{wmt_year}.wmt",
            ]:
                # Translation from original text.
                path_A = root_dir / f"data/{mt}/{phase}/org_de_wmt{wmt_year}.txt"
            elif path_B.name in [
                f"trans_de_wmt{wmt_year}.deepl.en",
                f"trans_de_wmt{wmt_year}.txt.en.google",
                f"trans_de_wmt{wmt_year}.wmt",
            ]:
                # Translation from Translationese.
                path_A = root_dir / f"data/{mt}/{phase}/trans_de_wmt{wmt_year}.txt"
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
                    for line_A, line_B in zip(sents_A, sents_B):
                        corpus_data.append([line_A.rstrip(), line_B.rstrip(), label])

    # Encode the sentences using the HuggingFace tokenizer.
    sentsA, sentsB, labels = zip(*corpus_data)
    sentsA, sentsB = list(sentsA), list(sentsB)
    tokenizer = AutoTokenizer.from_pretrained(
        args.arch, model_max_length=args.max_length
    )
    # encode input as translated + original
    if args.load_sentence_pairs == 'reverse':
        sents_enc = tokenizer(sentsB, sentsA, padding=True, truncation=True)
    else:
    # encode input as original + translated
        sents_enc = tokenizer(sentsA, sentsB, padding=True, truncation=True)

    return HFDataset(sents_enc, labels)

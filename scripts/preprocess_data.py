"""
Preprocess the downloaded dataset and word embeddings for the ESIM model.
"""
# Aurelien Coet, 2018.

import os
import pickle
import string
import fnmatch
import numpy as np
from collections import Counter


def read_data(filepath, lower=False, ignore_punct=False):
    """
    Read the premises, hypotheses and labels from a file in some NLI
    dataset and return them in a dictionary.

    Args:
        filepath: The path to the file containing the premises, hypotheses
            and labels that must be read. The file should be in the same
            form as the SNLI dataset (or MultiNLI).
        lower: Boolean indicating whether the words in the premises and
            hypotheses must be lowercased.
        ignore_punct: Boolean indicating whether to ignore punctuation in
            the sentences.

    Returns:
        A dictionary containing three lists, one for the premises, one for the
        hypotheses, and one for the labels.
    """
    with open(filepath, 'r') as input_data:
        premises, hypotheses, labels = [], [], []

        # Translation tables to remove parentheses and punctuation from
        # strings.
        parentheses_table = str.maketrans({'(': None, ')': None})
        punct_table = str.maketrans({key: ' ' for key in string.punctuation})

        # Ignore the headers on the first line in the SNLI file.
        next(input_data)

        for line in input_data:
            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            # Remove '(' and ')' from the premises and hypotheses.
            premise = premise.translate(parentheses_table)
            hypothesis = hypothesis.translate(parentheses_table)

            if lower:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punct:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append(premise.rstrip().split())
            hypotheses.append(hypothesis.rstrip().split())
            labels.append(line[0])

        return {"premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


def build_worddict(data, num_words=None):
    """
    Build a dictionary associating words from a set of premises and
    hypotheses to indices.

    Args:
        data: A dictionary containing the premises and hypotheses for which
            a worddict must be built. The dictionary should have been built
            by the read_data function of this module.
        num_words: Integer indicating the maximum number of words to
            keep in the worddict. If specified, only the 'num_words' most
            frequent words will be kept. If set to None, all words are
            kept.

    Returns:
        A dictionary associating words to indices.
    """
    words = []
    [words.extend(sentence) for sentence in data['premises']]
    [words.extend(sentence) for sentence in data['hypotheses']]

    counts = Counter(words)
    if num_words is None:
        num_words = len(counts)

    worddict = {word[0]: i+4
                for i, word in enumerate(counts.most_common(num_words))}
    # Special indices are used for padding, out-of-vocabulary words, and
    # beginning and end of sentence tokens.
    worddict["_PAD_"] = 0
    worddict["_OOV_"] = 1
    worddict["_BOS_"] = 2
    worddict["_EOS_"] = 3

    return worddict


def words_to_indices(sentence, worddict):
    """
    Transform the words in a sentence to indices.

    Args:
        sentence: A list of words that must be transformed to indices.
        worddict: A dictionary associating words to indices.

    Returns:
        A list of indices.
    """
    indices = [worddict["_BOS_"]]
    for word in sentence:
        if word in worddict:
            index = worddict[word]
        else:
            # Words absent from 'worddict' are treated as a special
            # out-of-vocabulary word (OOV).
            index = worddict['_OOV_']
        indices.append(index)
    indices.append(worddict["_EOS_"])

    return indices


def transform_to_indices(data, worddict, labeldict):
    """
    Transform the words in the premises and hypotheses of a dataset, as well
    as their associated labels, to integer indices.

    Args:
        data: A dictionary containing lists of premises, hypotheses
            and labels.
        worddict: A dictionary associating words to indices.
        labeldict: A dictionary associating labels to indices.

    Returns:
        A dictionary containing the transformed premises, hypotheses and
        labels.
    """
    transformed_data = {"premises": [], "hypotheses": [], "labels": []}

    for i, premise in enumerate(data['premises']):
        # Ignore sentences that have a label for which no index was
        # defined in 'labeldict'.
        label = data["labels"][i]
        if label not in labeldict:
            continue

        transformed_data["labels"].append(labeldict[label])

        indices = words_to_indices(premise, worddict)
        transformed_data["premises"].append(indices)

        indices = words_to_indices(data["hypotheses"][i], worddict)
        transformed_data["hypotheses"].append(indices)

    return transformed_data


def build_embedding_matrix(worddict, embeddings_file):
    """
    Build an embedding matrix with pretrained weights for a given worddict.

    Args:
        worddict: A dictionary associating words to unique indices.
        embeddings_file: A file containing pretrained word embeddings.

    Returns:
        A numpy matrix of size (num_words+2 x embedding_dim) containing
        pretrained word embeddings (the +2 is for the padding and
        out-of-vocabulary tokens).
    """
    # Load the word embeddings in a dictionnary.
    embeddings = {}
    with open(embeddings_file, 'r', encoding='utf8') as input_data:
        for line in input_data:
            line = line.split()

            try:
                # Check that the second element on the line is the start
                # of the embedding and not another word.
                float(line[1])
                word = line[0]
                if word in worddict:
                    embeddings[word] = line[1:]

            # Ignore lines corresponding to multiple words separated
            # by spaces.
            except ValueError:
                continue

    num_words = len(worddict)
    embedding_dim = len(list(embeddings.values())[0])
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # Actual building of the embedding matrix.
    for word, i in worddict.items():
        if word in embeddings:
            embedding_matrix[i] = np.array(embeddings[word], dtype=float)
        else:
            if word == "_PAD_":
                continue
            # Out of vocabulary words are initialised with random gaussian
            # samples.
            embedding_matrix[i] = np.random.normal(size=(embedding_dim))

    return embedding_matrix


def preprocess_NLI(inputdir, targetdir, embeddings_file, lower=False,
                   ignore_punct=False, num_words=None):
    """
    Preprocess the data from some NLI corpus so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the NLI corpus.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        embedding_file: The path to the file containing the pretrained
            word vectors to build the embedding matrix.
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    train_file = ""
    dev_file = ""
    test_file = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, '*_train.txt'):
            train_file = file
        elif fnmatch.fnmatch(file, '*_dev.txt'):
            dev_file = file
        elif fnmatch.fnmatch(file, '*_test.txt'):
            test_file = file

    print(20*"=", " Preprocessing train set ", 20*"=")
    print("\t* Reading data...")
    data = read_data(os.path.join(inputdir, train_file), lower=lower,
                     ignore_punct=ignore_punct)

    print("\t* Computing worddict and saving it...")
    worddict = build_worddict(data, num_words=num_words)
    with open(os.path.join(targetdir, "worddict.pkl"), 'wb') as pkl_file:
        pickle.dump(worddict, pkl_file)

    print("\t* Transforming words in premises and hypotheses to indices...")
    labeldict = {"entailment": 0, "neutral": 1, "contradiction": 2}
    transformed_data = transform_to_indices(data, worddict, labeldict)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    print(20*"=", " Preprocessing dev set ", 20*"=")
    print("\t* Reading data...")
    data = read_data(os.path.join(inputdir, dev_file), lower=lower,
                     ignore_punct=ignore_punct)

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = transform_to_indices(data, worddict, labeldict)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    print(20*"=", " Preprocessing test set ", 20*"=")
    print("\t* Reading data...")
    data = read_data(os.path.join(inputdir, test_file), lower=lower,
                     ignore_punct=ignore_punct)

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = transform_to_indices(data, worddict, labeldict)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "test_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    print(20*"=", " Preprocessing embeddings ", 20*"=")
    print("\t* Building embedding matrix and saving it...")
    embed_matrix = build_embedding_matrix(worddict, embeddings_file)
    with open(os.path.join(targetdir, "embeddings.pkl"), 'wb') as pkl_file:
        pickle.dump(embed_matrix, pkl_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess some NLI dataset\
 for ESIM')
    parser.add_argument('data_dir', help='Path to the NLI dataset to\
 preprocess')
    parser.add_argument('embeddings_file', help='Path to a file containing\
 pretrained word embeddings')
    parser.add_argument('target_dir', help='Path to the directory where the\
 preprocessed data must be saved')

    parser.add_argument('--lower', default=False, type=bool,
                        help='Boolean indicating whether to lowercase words in\
 the premises and hypotheses')
    parser.add_argument('--ignore_punct', default=False, type=bool,
                        help='Boolean indicating whether to ignore punctuation\
 in the hypotheses and premises')
    parser.add_argument('--num_words', default=None, type=int,
                        help='Number of words to use for the embeddgings')

    args = parser.parse_args()

    preprocess_NLI(args.data_dir, args.target_dir, args.embeddings_file,
                   lower=args.lower, ignore_punct=args.ignore_punct,
                   num_words=args.num_words)

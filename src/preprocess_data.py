"""
Preprocess the data for the ESIM model.
"""
# Aurelien Coet, 2018.

import string
from collections import Counter


def read_data(filepath):
    """
    Read the premises, hypotheses and labels from a file in the SNLI
    dataset and return them in a dictionary of lists.

    Args:
        filepath: The path to the file containing the premises, hypotheses
            and labels that must be read.

    Returns:
        A dictionary containing three lists, one for the premises, one for the
        hypotheses, and one for the labels.
    """
    with open(filepath, 'r') as input_data:
        premises, hypotheses, labels = [], [], []

        # Translation table to remove punctuation from strings.
        table = str.maketrans({key: None for key in string.punctuation})

        # Ignore the headers on the first line in the SNLI file.
        next(input_data)

        for line in input_data:
            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            # Each premise and hypothesis is split into a list of words.
            # Punctuation is removed.
            premises.append(line[5].translate(table).rstrip().split())
            hypotheses.append(line[6].translate(table).rstrip().split())
            labels.append(line[0])

        return {"premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


def build_worddict(data, num_words=None):
    """
    Build a dictionary associating words from a set of premises and
    hypotheses to indices.

    Args:
        data: A dictionary with at least two elements that are lists
            of lists containing words and that have 'premises' and
            'hypotheses' as keys.
        num_words: If specified, indicates the maximum number of words to
            keep in the worddict.

    Returns:
        A dictionary associating words to indices.
    """
    words = []
    [words.extend(sentence) for sentence in data['premises']]
    [words.extend(sentence) for sentence in data['hypotheses']]

    counts = Counter(words)
    if num_words is None:
        num_words = len(counts)

    worddict = {word[0]: i+2
                for i, word in enumerate(counts.most_common(num_words))}
    worddict["_PAD_"] = 0
    worddict["_OOV_"] = 1

    return worddict


def transform_to_indices(data, worddict, labeldict):
    """
    Transform the words in the premises and hypotheses of a dataset, as well
    as their associated labels, to integer indices.

    Args:
        data: A dictionary containing a list of premises, hypotheses
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
        transformed_data["premises"].append([])
        transformed_data["hypotheses"].append([])

        for word in premise:
            if word in worddict:
                transformed_data["premises"][i].append(worddict[word])
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                transformed_data["premises"][i].append(worddict["_OOV_"])

        for word in data["hypotheses"][i]:
            if word in worddict:
                transformed_data["hypotheses"][i].append(worddict[word])
            else:
                transformed_data["hypotheses"][i].append(worddict["_OOV_"])

    return transformed_data


def build_embedding_matrix(worddict, embeddings_file):
    """
    """
    # TODO
    pass


def preprocess_SNLI(inputdir, targetdir):
    """
    """
    # TODO
    pass


if __name__ == "__main__":
    pass

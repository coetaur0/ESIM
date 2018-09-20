"""
Dataset definition for NLI datasets.
"""
# Aurelien Coet, 2018.

import torch
from torch.utils.data import Dataset


class NLIDataset(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 data,
                 padding_idx=0,
                 max_premise_length=None,
                 max_hypothesis_length=None):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {"premises": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * padding_idx,
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}

        for i, premise in enumerate(data["premises"]):
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index],
                                      self.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index],
                                         self.max_hypothesis_length),
                "label": self.data["labels"][index]}

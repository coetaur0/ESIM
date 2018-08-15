"""
Dataset definition for the data that must be fed to the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
from torch.utils.data import Dataset


class ESIMDataset(Dataset):
    """
    Dataset for the ESIM model.
    """

    def __init__(self, data, pad_idx=0, max_prem_len=None, max_hyp_len=None):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            pad_idx: An integer indicating the index being used for the
                padding token in the data.
            max_prem_len: An integer indicating what is the maximum length
                accepted for the sequences in the premises.
                If set to None, the length of the longest premise in 'data'
                is used.
            max_hyp_len: An integer indicating what is the maximum length
                accepted for the sequences in the hypotheses.
                If set to None, the length of the longest hypothesis in
                'data' is used.
        """
        if max_prem_len is None:
            max_prem_len = max([len(seq) for seq in data["premises"]])

        if max_hyp_len is None:
            max_hyp_len = max([len(seq) for seq in data["hypotheses"]])

        self.num_seqs = len(data["premises"])

        self.data = {"premises": torch.ones((self.num_seqs, max_prem_len),
                                            dtype=torch.int64) * pad_idx,
                     "hypotheses": torch.ones((self.num_seqs, max_hyp_len),
                                              dtype=torch.int64) * pad_idx,
                     "labels": torch.tensor(data["labels"])}

        for i, premise in enumerate(data["premises"]):
            end = min(len(premise), max_prem_len)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), max_hyp_len)
            self.data["hypotheses"][i][:end] =\
                torch.tensor(hypothesis[:end])

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        return {"premise": self.data["premises"][idx],
                "hypothesis": self.data["hypotheses"][idx],
                "label": self.data["labels"][idx]}

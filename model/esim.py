"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn

from model.layers import Seq2seqEncoder, SoftmaxAttention
from model.utils import get_mask, replace_masked


class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self, embeddings, hidden_size, padding_idx=0, num_classes=3,
                 dropout=0.5, device="cpu"):
        """
        """
        super(ESIM, self).__init__()

        self.vocab_size, self.embedding_dim = embeddings.size()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        self._encoding = Seq2seqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        hidden_size,
                                        bidirectional=True,
                                        device=device)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU(),
                                         nn.Dropout(p=self.dropout))

        self._composition = Seq2seqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True,
                                           device=device)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes),
                                             nn.Softmax())

    def forward(self, premise, premise_len, hypothesis, hypothesis_len):
        """
        """
        premise_mask = get_mask(premise, premise_len).to(self.device)
        hypothesis_mask = get_mask(hypothesis, hypothesis_len).to(self.device)

        embedded_premise = self._word_embedding(premise)
        embedded_hypothesis = self._word_embedding(hypothesis)

        encoded_premise = self._encoding(embedded_premise,
                                         premise_len)
        encoded_hypothesis = self._encoding(embedded_hypothesis,
                                            hypothesis_len)

        attended_premise, attended_hypothesis =\
            self._attention(encoded_premise, premise_mask,
                            encoded_hypothesis, hypothesis_mask)

        enhanced_premise = torch.cat([encoded_premise,
                                      attended_premise,
                                      encoded_premise-attended_premise,
                                      encoded_premise*attended_premise],
                                     dim=-1)
        enhanced_hypothesis = torch.cat([encoded_hypothesis,
                                         attended_hypothesis,
                                         encoded_hypothesis -
                                         attended_hypothesis,
                                         encoded_hypothesis *
                                         attended_hypothesis],
                                        dim=-1)

        projected_premise = self._projection(enhanced_premise)
        projected_hypothesis = self._projection(enhanced_hypothesis)

        v_ai = self._composition(projected_premise, premise_len)
        v_bj = self._composition(projected_hypothesis, hypothesis_len)

        v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(1)
                                               .transpose(2, 1), dim=1)\
            / torch.sum(premise_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypothesis_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypothesis_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premise_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypothesis_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        return self._classification(v)

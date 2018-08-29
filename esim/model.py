"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn

from .layers import RNNDropout, Seq2seqEncoder, SoftmaxAttention
from .utils import get_mask, replace_masked


class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self, embeddings, hidden_size, padding_idx=0, num_classes=3,
                 dropout=0.5, device="cpu"):
        """
        Args:
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings.
            hidden_size: The size of the hidden layers in the network.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            dropout: The dropout rate to use on the outputs of the feedforward
                layers of the network. Defaults to 0.5. A dropout rate of 0
                corresponds to using no dropout at all.
            device: The device on which the model is executed. Defaults to 
                'cpu'. 
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

        self._rnn_dropout = RNNDropout(p=dropout)

        self._encoding = Seq2seqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        hidden_size,
                                        bidirectional=True,
                                        device=device)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

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
                                             nn.Softmax(dim=-1))

        # Initialize all weights and biases in the model.
        self.apply(_init_weights)

    def forward(self, premise, premise_len, hypothesis, hypothesis_len):
        """
        Args:
            premise: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, max_premise_len).
            premise_len: A 1D tensor containing the lengths of the premises
                in 'premise'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, max_hypothesis_len).
            premise_len: A 1D tensor containing the lengths of the hypotheses
                in 'hypothesis'.

        Returns:
            A tensor of size (batch, num_classes) containing the probability
            of each class.
        """
        premise_mask = get_mask(premise, premise_len).to(self.device)
        hypothesis_mask = get_mask(hypothesis, hypothesis_len).to(self.device)

        embedded_premise = self._word_embedding(premise)
        embedded_hypothesis = self._word_embedding(hypothesis)

        embedded_premise = self._rnn_dropout(embedded_premise)
        embedded_hypothesis = self._rnn_dropout(embedded_hypothesis)

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

        projected_premise = self._rnn_dropout(projected_premise)
        projected_hypothesis = self._rnn_dropout(projected_hypothesis)

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


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0

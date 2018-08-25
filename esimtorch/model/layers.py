"""
Definition of the layers of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn

from .utils import sort_by_seq_len, masked_softmax, weighted_sum


class Seq2seqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.

    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to their original order.
    """

    def __init__(self, rnn_type, input_size, hidden_size,
                 num_layers=1, bias=True, dropout=0.0, bidirectional=False,
                 device="cpu"):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used by the module as encoder.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout:  If non-zero, introduces a Dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to dropout. Defaults to 0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase),\
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2seqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def _init_hidden(self, batch_size):
        """
        Return an initial state for the RNN.

        Args:
            batch_size: The size of the batch for which a new initial state
                must be computed.

        Returns:
            A tuple containing the initial state for the LSTM.
        """
        num_directions = 1
        if self.bidirectional:
            num_directions = 2

        if self.rnn_type == nn.LSTM:
            return (torch.zeros(self.num_layers*num_directions,
                                batch_size,
                                self.hidden_size,
                                dtype=torch.float).to(self.device),
                    torch.zeros(self.num_layers*num_directions,
                                batch_size,
                                self.hidden_size,
                                dtype=torch.float).to(self.device))

        return (torch.zeros(self.num_layers*num_directions,
                            batch_size,
                            self.hidden_size,
                            dtype=torch.float).to(self.device))

    def forward(self, seq_batch, seq_lens):
        batch_size = seq_batch.shape[0]

        sorted_batch, sorted_lens, _, restoration_idx =\
            sort_by_seq_len(seq_batch, seq_lens)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lens,
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, self._init_hidden(batch_size))

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs


class SoftmaxAttention(nn.Module):
    """
    Layer taking premises and hypotheses encoded by an RNN and computing the
    soft attention between their elements.

    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the premises.
    """

    def forward(self, premise_batch, premise_mask, hypothesis_batch,
                hypothesis_mask):
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                                              .contiguous())

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                                        .contiguous(),
                                       premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch,
                                         prem_hyp_attn,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_prem_attn,
                                           hypothesis_mask)

        return attended_premises, attended_hypotheses

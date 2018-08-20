"""
Utilities for the ESIM model.
"""
# Aurelien Coet, 2018.

import torch


# Code inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def sort_by_seq_len(batch, seq_lens, decreasing=True):
    """
    Sort a batch of padded variable length sequences by length.

    Args:
        batch: A batch of padded variable length sequences. The batch should
            have the dimensions (batch_size x max_sequence_length x *).
        seq_lens: A tensor containing the lengths of the sequences in the
            input batch. The tensor should be of size (batch_size).
        decreasing: A boolean value indicating whether to sort the sequences
            by their lengths in decreasing order. Defaults to True.

    Returns:
        sorted_batch: A tensor containing the input batch reordered by
            sequences lengths.
        sorted_seq_lens: A tensor containing the sorted lengths of the
            sequences in the input batch.
        sorting_idx: A tensor containing the indices used to permute the input
            batch in order to get 'sorted_batch'.
        restoration_idx: A tensor containing the indices that can be used to
            restore the order of the sequences in 'sorted_batch' so that it
            matches the input batch.
    """
    sorted_seq_lens, sorting_idx = seq_lens.sort(0, decreasing=decreasing)

    sorted_batch = batch.index_select(0, sorting_idx)

    idx_range = seq_lens.new_tensor(torch.arange(0, len(seq_lens)))
    _, reverse_mapping = sorting_idx.sort(0, descending=False)
    restoration_idx = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_idx, restoration_idx

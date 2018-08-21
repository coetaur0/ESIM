"""
Utilities for the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn


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
    sorted_seq_lens, sorting_idx = seq_lens.sort(0, descending=decreasing)

    sorted_batch = batch.index_select(0, sorting_idx)

    idx_range = seq_lens.new_tensor(torch.arange(0, len(seq_lens)))
    _, reverse_mapping = sorting_idx.sort(0, descending=False)
    restoration_idx = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_idx, restoration_idx


# Code inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).

    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.

    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)

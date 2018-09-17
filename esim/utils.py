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


def get_mask(seq_batch, seq_lens):
    """
    Get the mask for a batch of padded variable length sequences.

    Args:
        seq_batch: A batch of padded variable length sequences containing
            word indices. It is a 2-dimensional tensor with size 
            (batch, sequence).
        seq_lens: A tensor containing the lengths of the sequences in
            'seq_batch'. It is of size (batch).

    Returns:
        A mask of size (batch, max_seq_len), where max_seq_len is the length
        of the longest sequence in the batch.
    """
    batch_size = seq_batch.size()[0]
    max_len = torch.max(seq_lens)
    mask = torch.ones(batch_size, max_len)
    mask[seq_batch[:, :max_len] == 0] = 0
    return mask


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

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.

    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.

    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


# Code inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.

    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to use in the masked vectors of 'tensor'.

    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


def correct_preds(out_probs, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.

    Args:
        out_probs: A tensor of probabilities for different output classes.
        targets: The indices of the actual target classes.

    Returns:
        The number of correct predictions.
    """
    _, out_classes = out_probs.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

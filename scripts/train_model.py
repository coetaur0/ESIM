"""
Train the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import argparse
import time
import torch

from torch.utils.data import DataLoader
from model.dataset import NLIDataset
from model.esim import ESIM


def train(dataloader, model, optimizer, criterion, epoch, device, print_freq):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.

    Args:
        dataloader: A DataLoader object to iterate over the training data.
        model: A torch module that must be trained on the input data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch: The number of the epoch for which training is performed.
        device: A device on which training must be performed.
        print_freq: An integer value indicating at which frequency training
            information must be printed out.

    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0

    for i, batch in enumerate(dataloader):
        batch_start = time.time()

        # Move input and output data to the GPU if one is used.
        premises = batch['premise'].to(device)
        premise_lens = batch['premise_len'].to(device)
        hypotheses = batch['hypothesis'].to(device)
        hypothesis_lens = batch['hypothesis_len'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(premises, premise_lens, hypotheses, hypothesis_lens)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

        if i % print_freq == 0:
            print("Epoch {}, batch {}:".format(epoch, i))
            print("\t* Avg. batch processing time: {:.4f}s"
                  .format(batch_time_avg/(i+1)))
            print("\t* Loss: {:.4f}"
                  .format(running_loss/((i+1)*dataloader.batch_size)))

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)

    return epoch_time, epoch_loss


def validate(dataloader, model, criterion, epoch, device, print_freq):
    """
    """
    # Switch to evaluate mode.
    model.eval()

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            premises = batch['premise'].to(device)
            premise_lens = batch['premise_len'].to(device)
            hypotheses = batch['hypothesis'].to(device)
            hypothesis_lens = batch['hypothesis_len'].to(device)
            labels = batch['label'].to(device)

            outputs = model(premises, premise_lens, hypotheses, hypothesis_lens)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_accuracy += correct_preds(outputs, labels)

            batch_time_avg += time.time() - batch_start

            if i % print_freq == 0:
                print("Epoch {}, batch {}:".format(epoch, i))
                print("\t* Avg. batch processing time: {:.4f}s"
                  .format(batch_time_avg/(i+1)))
                print("\t* Loss: {:.4f}"
                  .format(running_loss/((i+1)*dataloader.batch_size)))
                print("\t* Accuracy: {:.4f}%"
                  .format((running_accuracy/((i+1)*dataloader.batch_size))*100))

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader)

    return epoch_time, epoch_loss, epoch_accuracy


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

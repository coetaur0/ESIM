"""
Test the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import os
import time
import pickle
import argparse
import torch
import json

from torch.utils.data import DataLoader
from esim.dataset import NLIDataset
from esim.model import ESIM
from esim.utils import correct_preds


def test(dataloader, model, device):
    """
    Test the accuracy of a model on some dataset.

    Args:
        dataloader: A DataLoader object to iterate over some dataset.
        model: The torch model to test.
        device: The device on which the model is being executed.

    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            premises = batch['premise'].to(device)
            premise_lens = batch['premise_len'].to(device)
            hypotheses = batch['hypothesis'].to(device)
            hypothesis_lens = batch['hypothesis_len'].to(device)
            labels = batch['label'].to(device)

            outputs = model(premises, premise_lens,
                            hypotheses, hypothesis_lens)

            accuracy += correct_preds(outputs, labels)
            batch_time += time.time() - batch_start

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, accuracy


def main(test_file, pretrained_file, vocab_size, embedding_dim,
         hidden_size=300, num_classes=3, batch_size=32):
    """
    Test the ESIM model with pretrained weights on some dataset.

    Args:
        test_file: The path to a file containing preprocessed NLI data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Testing ESIM model on device: {}".format(device))

    print("- Loading test data...")
    with open(test_file, 'rb') as pkl:
        test_data = NLIDataset(pickle.load(pkl))

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("- Building model...")
    model = ESIM(vocab_size, embedding_dim, hidden_size,
                 num_classes=num_classes, device=device).to(device)

    checkpoint = torch.load(pretrained_file)
    model.load_state_dict(checkpoint['state_dict'])

    print("- Testing model...")
    batch_time, total_time, accuracy = test(test_loader, model, device)

    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s, accuracy: {:.4f}%".format(batch_time, total_time, (accuracy*100)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the ESIM model on\
 some dataset')

    parser.add_argument('checkpoint', help="Path to a checkpoint produced by\
 the 'train_model' script, containing a pretrained model")
    parser.add_argument('--config', default='../config/test_cfg.json',
                        help='Path to a configuration file')
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as config_file:
        config = json.load(config_file)

    main(os.path.normpath(config['test_data']),
         args.checkpoint,
         config['vocab_size'],
         config['embedding_dim'],
         config['hidden_size'],
         config['num_classes'],
         config['batch_size'])

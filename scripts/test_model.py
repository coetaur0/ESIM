"""
Test the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import time
import pickle
import argparse
import torch

from torch.utils.data import DataLoader
from esim.dataset import NLIDataset
from esim.model import ESIM


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


def main(test_file, embeddings_file, pretrained_file, batch_size=32,
         hidden_size=300, num_classes=3, dropout=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Testing ESIM model on device: {}".format(device))

    print("- Loading test data...")
    with open(test_file, 'rb') as pkl:
        test_data = NLIDataset(pickle.load(pkl))

    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    print("- Building model...")
    with open(embeddings_file, 'rb') as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
                     .to(device)

    model = ESIM(embeddings, hidden_size, num_classes=num_classes,
                 dropout=dropout, device=device).to(device)

    checkpoint = torch.load(pretrained_file)
    model.load_state_dict(checkpoint['state_dict'])

    print("- Testing model...")
    batch_time, total_time, accuracy = test(test_loader, model, device)

    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s, accuracy: {:.4f}%".format(batch_time, total_time, (accuracy*100)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the ESIM model on\
 some dataset')

    parser.add_argument('test_file', help='The path to a file containing some\
 preprocessed data to test the model on')
    parser.add_argument('embeddings_file', help='The path to a file containing\
 some preprocessed word embeddings')
    parser.add_argument('pretrained_file', help='The path to a saved pretrained\
 model')

    parser.add_argument('--batch_size', default=32, type=int, help='The batch\
 size')
    parser.add_argument('--hidden_size', default=300, type=int, help='The\
 hidden size to use for the layers in the model')
    parser.add_argument('--num_classes', default=3, type=int, help='The number\
 of classes in the targets')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='The dropout rate to use in the model')

    args = parser.parse_args()

    main(args.test_file, args.embeddings_file, args.pretrained_file,
         args.batch_size, args.hidden_size, args.num_classes, args.dropout)

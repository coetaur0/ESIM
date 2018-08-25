"""
Train the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import os
import argparse
import time
import pickle
import torch

import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils.data import DataLoader
from esimtorch.model.dataset import NLIDataset
from esimtorch.model.esim import ESIM


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
            print("\t- Batch {}:".format(i))
            print("\t\t* Avg. batch processing time: {:.4f}s"
                  .format(batch_time_avg/(i+1)))
            print("\t\t* Loss: {:.4f}"
                  .format(running_loss/(i+1)))

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)

    return epoch_time, epoch_loss


def validate(dataloader, model, criterion, epoch, device, print_freq):
    """
    Compute the loss and accuracy of a model on a validation dataset.

    Args:
        dataloader: A DataLoader object to iterate over the validation data.
        model: A torch module for which the loss and accuracy must be
            computed.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is.
        print_freq: An integer value indicating at which frequency validation
            information must be printed out.

    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    running_accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            premises = batch['premise'].to(device)
            premise_lens = batch['premise_len'].to(device)
            hypotheses = batch['hypothesis'].to(device)
            hypothesis_lens = batch['hypothesis_len'].to(device)
            labels = batch['label'].to(device)

            outputs = model(premises, premise_lens,
                            hypotheses, hypothesis_lens)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_accuracy += correct_preds(outputs, labels)

            batch_time_avg += time.time() - batch_start

            if i % print_freq == 0:
                print("\t- Batch {}:".format(i))
                print("\t\t* Avg. batch processing time: {:.4f}s"
                      .format(batch_time_avg/(i+1)))
                print("\t\t* Loss: {:.4f}"
                      .format(running_loss/(i+1)))
                print("\t\t* Accuracy: {:.4f}%"
                      .format((running_accuracy/((i+1)*dataloader.batch_size))
                              * 100))

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader) *
                                         dataloader.batch_size)

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


def main(train_file, valid_file, embeddings_file, target_dir, epochs,
         batch_size, hidden_size, num_classes, dropout, patience, print_freq):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(train_file, 'rb') as pkl:
        train_data = NLIDataset(pickle.load(pkl))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    with open(valid_file, 'rb') as pkl:
        valid_data = NLIDataset(pickle.load(pkl))

    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

    with open(embeddings_file, 'rb') as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
                     .to(device)

    model = ESIM(embeddings, hidden_size, num_classes=num_classes,
                 dropout=dropout, device=device).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)

    best_score = 0.0
    patience_counter = 0
    epochs_count = []
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        epochs_count.append(epoch)
        print("Epoch: {}".format(epoch))

        epoch_time, epoch_loss = train(train_loader, model, optimizer,
                                       criterion, epoch, device, print_freq)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, training loss = {:.4f}"
              .format(epoch_time, epoch_loss))

        epoch_time, epoch_loss, epoch_accuracy = validate(valid_loader, model,
                                                          criterion, epoch,
                                                          device,
                                                          print_freq/2)

        valid_losses.append(epoch_loss)
        print("-> Validation time: {:.4f}s, validation loss: {:.4f}"
              .format(epoch_time, epoch_loss))
        print("-> Accuracy: {:.4f}%".format((epoch_accuracy*100)))

        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'best_score': best_score,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(target_dir, "esim_{}.pth.tar"
                                                .format(epoch)))

        if patience_counter == patience:
            break

    plt.figure()
    plt.plot(epochs_count, train_losses, '-r')
    plt.plot(epochs_count, valid_losses, '-b')
    plt.xlabel('epoch')
    plt.ylabel('cross entropy loss')
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the ESIM model')

    parser.add_argument('train_file', help='A path to a file containing some\
 preprocessed training data')
    parser.add_argument('valid_file', help='A path to a file containing some\
 preprocessed validation data')
    parser.add_argument('embeddings_file', help='A path to a file containing\
 some preprocessed word embeddings')

    parser.add_argument('--target_dir', default=os.path.join('..', '..',
                                                             'data',
                                                             'pretrained'),
                        help='The path to the directory where the trained\
 model\'s parameters must be saved')
    parser.add_argument('--epochs', default=64, type=int, help='The maximum\
 number of epochs to apply for training')
    parser.add_argument('--batch_size', default=32, type=int, help='The batch\
 size')
    parser.add_argument('--hidden_size', default=300, type=int, help='The\
 hidden size to use for the layers in the model')
    parser.add_argument('--num_classes', default=3, type=int, help='The number\
 of classes in the targets')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='The dropout rate to use in the model')
    parser.add_argument('--patience', default=5, type=int, help='The patience\
 to use during training for early stopping')
    parser.add_argument('--print_freq', default=1000, type=int,
                        help='The number of batches after which information\
 must be printed during training')

    args = parser.parse_args()

    main(args.train_file, args.valid_file, args.embeddings_file,
         args.target_dir, args.epochs, args.batch_size, args.hidden_size,
         args.num_classes, args.dropout, args.patience, args.print_freq)

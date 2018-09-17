"""
Train the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import os
import argparse
import time
import pickle
import torch
import json

import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils.data import DataLoader
from esim.dataset import NLIDataset
from esim.model import ESIM
from esim.utils import correct_preds


def train(model, dataloader, optimizer, criterion, epoch, max_grad_norm,
          device, print_freq):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.

    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch: The number of the epoch for which training is performed.
        max_grad_norm: Max. norm for gradient norm clipping.
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

        # Move input and output data to the GPU if it is used.
        premises = batch['premise'].to(device)
        premise_lens = batch['premise_len'].to(device)
        hypotheses = batch['hypothesis'].to(device)
        hypothesis_lens = batch['hypothesis_len'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(premises, premise_lens, hypotheses, hypothesis_lens)
        loss = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

        if i % print_freq == 0:
            print("\t* Batch {}:".format(i))
            print("\t\t** Avg. batch processing time: {:.4f}s"
                  .format(batch_time_avg/(i+1)))
            print("\t\t** Loss: {:.4f}"
                  .format(running_loss/(i+1)))

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)

    return epoch_time, epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Compute the loss and accuracy of a model on some validation dataset.

    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.

    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
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

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy


def main(train_file, valid_file, embeddings_file, target_dir,
         hidden_size=300, num_classes=3, dropout=0.5,
         epochs=64, batch_size=32, patience=5, max_grad_norm=10.0,
         print_freq=1000, checkpoint=None):
    """
    Train the ESIM model on some dataset.

    Args:
        train_file: A path to some preprocessed dataset that must be used
            to train the model.
        valid_file: A path to some preprocessed dataset that must be used
            to validate the model.
        embeddings_file: A path to some preprocessed word embeddings that
            must be used to initialise the model.
        target_dir: The path to a directory where the trained model must
            be saved.
        hidden_size: The size of the hidden layers in the model. Defaults
            to 300.
        num_classes: The number of classes in the output of the model.
            Defaults to 3.
        dropout: The dropout rate to use in the model. Defaults to 0.5.
        epochs: The maximum number of epochs for training. Defaults to 64.
        batch_size: The size of the batches for training. Defaults to 32.
        patience: The patience to use for early stopping. Defaults to 5.
        print_freq: The frequency at which training information must be
            printed out.
        checkpoint: A checkpoint from which to continue training. If None,
            training starts from scratch. Defaults to None.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Training ESIM model on device: {}".format(device))

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    print("- Loading training data...")
    with open(train_file, 'rb') as pkl:
        train_data = NLIDataset(pickle.load(pkl))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("- Loading validation data...")
    with open(valid_file, 'rb') as pkl:
        valid_data = NLIDataset(pickle.load(pkl))

    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

    # -------------------- Model definition ------------------- #
    print('- Building model...')
    with open(embeddings_file, 'rb') as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
                     .to(device)

    model = ESIM(embeddings.shape[0], embeddings.shape[1], hidden_size,
                 embeddings=embeddings, num_classes=num_classes,
                 dropout=dropout, device=device).to(device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.5,
                                                           patience=0)

    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']
        best_score = checkpoint['best_score']

        print("- Continuing training on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epochs_count = checkpoint['epochs_count']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy = validate(model, valid_loader,
                                             criterion, device)
    print("Validation loss before training: {:.4f}, accuracy: {:.4f}%"
          .format(valid_loss, (valid_accuracy*100)))

    # -------------------- Training epochs ------------------- #
    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)

        print("- Training epoch {}:".format(epoch))
        epoch_time, epoch_loss = train(model, train_loader, optimizer,
                                       criterion, epoch, max_grad_norm,
                                       device, print_freq)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}"
              .format(epoch_time, epoch_loss))

        print("- Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model, valid_loader,
                                                          criterion, device)

        valid_losses.append(epoch_loss)
        print("-> Validation time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0

            # Save the model if the score is better or as good as
            # previous ones.
            torch.save({'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'best_score': best_score,
                        'optimizer': optimizer.state_dict(),
                        'epochs_count': epochs_count,
                        'train_losses': train_losses,
                        'valid_losses': valid_losses},
                       os.path.join(target_dir, "esim_{}.pth.tar"
                                                .format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    # Plotting of the loss curves for the train and validation sets.
    plt.figure()
    plt.plot(epochs_count, train_losses, '-r')
    plt.plot(epochs_count, valid_losses, '-b')
    plt.xlabel('epoch')
    plt.ylabel('cross entropy loss')
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the ESIM model')

    parser.add_argument('--config', default="../config/train_cfg.json",
                        help='Path to a json configuration file')
    parser.add_argument('--checkpoint', default=None,
                        help='path to a checkpoint file to resume training')
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as config_file:
        config = json.load(config_file)

    main(os.path.normpath(config["train_data"]),
         os.path.normpath(config["valid_data"]),
         os.path.normpath(config["embeddings"]),
         os.path.normpath(config["target_dir"]),
         config["hidden_size"], config["num_classes"], config["dropout"],
         config["epochs"], config["batch_size"], config["patience"],
         config["max_grad_norm"], config["print_freq"],
         args.checkpoint)

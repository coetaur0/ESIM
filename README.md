# ESIM - Enhanced Sequential Inference Model
Implementation of the ESIM model for natural language inference with PyTorch

This repository contains an implementation with PyTorch of the sequential model presented in the paper 
["Enhanced LSTM for Natural Language Inference"](https://arxiv.org/pdf/1609.06038.pdf) by Chen et al. in 2016.

## How to
### Install the package
To use the model defined in this repository, first install PyTorch on you machine by following the steps described on the
package's [official page](https://pytorch.org/get-started/locally/). Then, to install the dependencies necessary to run
the model, simply execute the command `pip install --upgrade .` from within the cloned repository (at the root, and preferably
inside of a [virtual environment](https://docs.python.org/3/tutorial/venv.html)).

### Fetch the data to train and test the model
The *fetch_data.py* script located in the *scripts/* folder of this repository can be used to download some NLI dataset and
pretrained word embeddings. By default, the script fetches the [SNLI](https://nlp.stanford.edu/projects/snli/) corpus and
the [GloVe 840B 300d](https://nlp.stanford.edu/projects/glove/) embeddings. Other datasets can be downloaded by simply passing
their URL as argument to the script (for example, the [MultNLI dataset](https://www.nyu.edu/projects/bowman/multinli/)).

The script's usage is the following:
```
fetch_data.py [-h] [--dataset_url DATASET_URL]
              [--embeddings_url EMBEDDINGS_URL]
              [--target_dir TARGET_DIR]
```
where `taget_dir` is the path to a directory where the downloaded data must be saved (defaults to *../data/*).

### Preprocess the data
Before the downloaded corpus and embeddings can be used in the ESIM model, they need to be preprocessed. This can be done with
the *preprocess_data.py* script in the *scripts/* folder of this repository. 

The script's usage is the following:
```
preprocess_data.py [-h] [--config CONFIG]
```
where `config` is the path to a configuration file defining the parameters to be used for preprocessing. A default configuration
file can be found in the *config/* folder of this repository.

### Train the model
The *train_model.py* script can be used to train the ESIM model on some training data and validate it on some validation data.

The script's usage is:
```
train_model.py [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
```
where `config` is a configuration file (a default one is located in the *config/* folder), and `checkpoint` is an optional
checkpoint from which training can be resumed. Checkpoints are created by the script after each training epoch, with the name
*esim_\*.pth.tar*, where '\*' indicates the epoch's number.

### Test the model
The *test_model.py* script can be used to test the model on some test data.

Its usage is:
```
test_model.py [-h] [--config CONFIG] checkpoint
```
where `config` is a configuration file (again, a default one is available in *config/*) and `checkpoint` is either one of the 
checkpoints created after the training epochs, or the best model seen during training, which is saved in 
*data/checkpoints/best.pth.tar* (the difference between the *esim_\*.pth.tar* files and *best.pth.tar* is that the latter cannot
be used to resume training, as it doesn't contain the optimizer's state).

## Results
A pretrained model is made available in the *data/checkpoints* folder of this repository. The model was trained with the
parameters defined in the default configuration files provided in *config/*.
To test it, simply execute `python test_model.py ../data/checkpoints/best.pth.tar` from within the *scripts/* folder.

The pretrained model achieves the following performance on the SNLI dataset:

| Split | Accuracy (%) |
|-------|--------------|
| Train |     93.2     |
| Dev   |     88.4     |
| Test  |     88.0     |

The results are in line with those presented in the paper by Chen et al.

On the [Breaking NLI](https://github.com/BIU-NLP/Breaking_NLI) dataset, published by [Glockner et al. in 2018](https://arxiv.org/pdf/1805.02266.pdf), the model reaches **65.5%** accuracy, as reported in the paper.

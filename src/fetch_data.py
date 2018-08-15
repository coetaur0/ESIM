"""
Fetch the following data for the ESIM model:
    - The SNLI corpus;
    - GloVe word embeddings.
"""
# Aurelien Coet, 2018.

import os
import zipfile
import wget


def download(url, targetdir):
    """
    Download a file and save it in some target directory.

    Args:
        url: The url from which the file must be downloaded.
        targetdir: The path to the directory where the file must be saved.
    """
    print("\t* Downloading data from {}".format(url))
    filepath = os.path.join(targetdir, url.split('/')[-1])
    wget.download(url, filepath)
    return filepath


def unzip(filepath):
    """
    Extract the data from a zipped file.

    Args:
        filepath: The path to the zipped file.
    """
    print("\n\t* Extracting: {}".format(filepath))
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        for name in zf.namelist():
            # Ignore useless files in archives.
            if "__MACOSX" in name or\
               ".DS_Store" in name or\
               "Icon" in name:
                continue
            zf.extract(name, dirpath)
    # Delete the archive once data has been extracted.
    os.remove(filepath)


def download_unzip(url, targetdir):
    """
    Download and unzip data from some url and save it in a target directory.

    Args:
        url: The url to download the data from.
        targetdir: The target directory in which to download and unzip the
                   data.
    """
    filepath = os.path.join(targetdir, url.split('/')[-1])

    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Download and unzip if the target directory is empty.
    if not os.listdir(targetdir):
        unzip(download(url, targetdir))
    # Skip downloading if the zipped data is already available.
    elif os.path.exists(filepath):
        print("\t* Found zipped data - skipping download")
        unzip(filepath)
    # Skip download and unzipping if the unzipped data is already available.
    else:
        print("\t* Found unzipped data for {} - skipping download and unzipping"
              .format(targetdir))


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    snli_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    print(20*'=' + "Fetching the SNLI data:" + 20*'=')
    download_unzip(snli_url, os.path.join(data_dir, "snli"))

    glove_url = "http://www-nlp.stanford.edu/data/glove.840B.300d.zip"
    print(20*'=' + "Fetching the GloVe data:" + 20*'=')
    download_unzip(glove_url, os.path.join(data_dir, "glove"))

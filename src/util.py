import os
import random


def get_files_in_directory(directory):
    """
    Lists all files in a given directory.

    :param directory: The directory to be listed.

    :return: A list of absolute file paths to files in the given directory.
    """
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if not f.startswith(".")
    ]


def split_train_test(data, train_split=0.8, random_seed=None):
    """
    Splits the given data into a training and a test set.

    :param data: The data to be split.
    :param train_split: The percentage of the data to be used for training.
    :param random_seed: The random seed to be used for shuffling the data. If None, the current system time is used.

    :return: A tuple containing the training and the test data.
    """
    if random_seed:
        random.seed(random_seed)
    random.shuffle(data)
    n_train = int(len(data) * train_split)
    return data[:n_train], data[n_train:]

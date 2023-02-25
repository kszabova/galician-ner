"""
This module provides functionality for training a named entity recognition model with spaCy.
"""

import spacy
import os
import random

from thinc.api import Config
from spacy.tokens import DocBin
from spacy.cli.train import train
from xml_webanno_parser import get_entities_for_spacy


def create_spacy_data(data, nlp, output_file):
    """
    Creates a spaCy NER training data from a list of tuples containing the text and the entities.

    :param data: A list of tuples containing the text and the entities.
    :param nlp: A spaCy nlp object.
    :param output_file: The path to the directory where the model should be stored.
    """
    db = DocBin()
    for text, annotations in data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_file)


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


def train_ner_spacy(config_file, output_path, create_data=False, *, data_dir=None):
    """
    Trains a NER model from the provided configuration file.

    :param config_file: The path to the configuration file.
    :param output_path: The path to the directory where the model should be stored.
    :param create_data: Whether the data should be created from the provided data directory.
    :param data_dir: The path to the directory containing the training data.
    """
    config = Config().from_disk(config_file)
    spacy.util.fix_random_seed(config["training"]["seed"])
    if create_data:
        if not data_dir:
            raise ValueError("Please provide a data directory.")
        lang = config["nlp"]["lang"]
        nlp = spacy.blank(lang)
        files = get_files_in_directory(data_dir)
        data = get_entities_for_spacy(files, ["BODYPART", "CLINENTITY"])
        train_data, test_data = split_train_test(
            data, random_seed=config["training"]["seed"]
        )
        create_spacy_data(train_data, nlp, config["paths"]["train"])
        create_spacy_data(test_data, nlp, config["paths"]["dev"])
    train(config_file, output_path=output_path)


def demo_spacy(model_path):
    """
    Uses a pre-trained model to predict named entities in user input.

    :param model_path: The path to the pre-trained model.
    """
    nlp = spacy.load(model_path)
    while True:
        text = input("Text ('exit' to quit):\n")
        if text.lower() == "exit":
            break
        doc = nlp(text)
        print("Entities:", *[(ent.text, ent.label_) for ent in doc.ents], sep="\n")


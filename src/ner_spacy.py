"""
This module provides functionality for training a named entity recognition model with spaCy.
"""

import spacy

from thinc.api import Config
from spacy.tokens import DocBin
from spacy.cli.train import train
from tsv_webanno_parser import get_entities_for_spacy
from util import get_files_in_directory, split_train_test


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
            if span is None:
                print("Skipping entity")
                continue
            ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_file)


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
        data = get_entities_for_spacy(files, None)  # TODO make this configurable
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


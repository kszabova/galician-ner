"""
This module provides functionality to parse XML files in WebAnno format.
"""

import xml.etree.ElementTree as ET

import spacy
from spacy.training import offsets_to_biluo_tags, biluo_to_iob


def parse_tree(file, remove_namespaces=False):
    """
    Parse an XML file and return the root element.

    :param file: The file to parse.
    :param remove_namespaces: Whether to remove namespaces from the tags.

    :return: The root element of the XML file.
    """
    tree = ET.iterparse(file)
    if remove_namespaces:
        for _, el in tree:
            _, _, el.tag = el.tag.rpartition("}")
    root = tree.root
    return root


def find_text(root):
    """
    Retrieves the original text of the annotated file from the root element.

    :param root: The root element of the XML file.

    :return: The original text of the annotated file.
    """
    sofa = root.findall("Sofa")
    if not sofa:
        raise ValueError(
            "No 'sofa' element has been found. Please provide a valid annotation file."
        )
    if len(sofa) > 1:
        raise ValueError(
            "Multiple 'sofa' elements have been found. Please provide a valid annotation file."
        )
    return sofa[0].attrib["sofaString"]


def get_sentences(root, text):
    """
    Retrieves individual sentences from the original text.

    :param: root: The root element of the XML file.
    :param: text: The original text of the annotated file.

    :return: A list of tuples containing the sentence text, the begin index and the end index of the sentence.
    """
    sentences_els = root.findall("Sentence")
    sentences = []
    for sentence in sentences_els:
        begin, end = int(sentence.attrib["begin"]), int(sentence.attrib["end"])
        sentence_text = text[begin:end]
        sentences.append((sentence_text, begin, end))
    return sentences


def assign_sentence_to_tag(tag_el, sentences):
    """
    Finds the corresponding sentence for a given tag.

    :param tag_el: The tag element.
    :param sentences: A list of tuples containing the sentence text, the begin index and the end index of the sentence.

    :return: A tuple containing the sentence text, the begin index and the end index of the entity within the sentence.
    """
    begin, end = int(tag_el.attrib["begin"]), int(tag_el.attrib["end"])
    sentence_tuple = None
    found = False
    for sentence in sentences:
        if begin >= sentence[1] and end <= sentence[2]:
            sentence_tuple = sentence
            found = True
            break
    if not found:
        raise ValueError(
            "The tag was not found within a sentence. Please provide a valid annotation file."
        )
    begin_new = int(begin) - int(sentence_tuple[1])
    end_new = int(end) - int(sentence_tuple[1])
    return (sentence_tuple[0], begin_new, end_new)


def label_sentence_IOB(sentence, entities, nlp):
    """
    Labels the sentence with the IOB tags.

    :param sentence: The sentence to be labeled.
    :param entities: A list of tuples containing the begin index and the end index of the entity within the sentence.

    :return: A list of tuples containing the sentence and the IOB labels.
    """
    doc = nlp(sentence)
    tags = biluo_to_iob(offsets_to_biluo_tags(doc, entities))
    return tags


def create_spacy_entities_from_tree(root, tags, allow_overlap=False):
    """
    Creates a list of tuples containing the sentence text and the entities within the sentence,
    as expected by the spaCy NER model.

    :param root: The root element of the XML file.
    :param tags: A list of tags to be considered as entities.

    :return: A list of tuples containing the sentence text and the entities within the sentence.
    """
    entities_map = {}
    text = find_text(root)
    sentences = get_sentences(root, text)
    for child in root:
        if child.tag in tags:
            sentence_entity = assign_sentence_to_tag(child, sentences)
            sentence_data = entities_map.setdefault(sentence_entity[0], [])
            sentence_data.append((sentence_entity[1], sentence_entity[2], child.tag))
    if not allow_overlap:
        remove_overlapping_entities(entities_map)
    return list(entities_map.items())


def remove_overlapping_entities(entities_map):
    """
    Removes overlapping entities from the list of entities.

    :param entities_map: A map of sentence - annotation pairs.

    :return: A map of sentence - annotation pairs without overlapping entities.
    """
    for annotations in entities_map.values():
        last_end = annotations[0][1]
        for annotation in annotations[1:]:
            if annotation[0] < last_end:
                annotations.remove(annotation)
            else:
                last_end = annotation[1]


def get_entities_for_spacy(files, tags):
    """
    Creates a list of data for the spaCy NER model from the specified annotated files.

    :param files: A list of XML files to be parsed. Must be in WebAnno format.
    :param tags: A list of tags to be considered as entities.
    """
    entities = []
    for file in files:
        try:
            root = parse_tree(file, remove_namespaces=True)
            entities.extend(create_spacy_entities_from_tree(root, tags))
        except:
            print(f"Error parsing file: {file}. Skipping.")
    return entities


def get_entities_for_transformers(files, tags, nlp):
    entities = get_entities_for_spacy(files, tags)
    entities_transformers = []
    for data in entities:
        tokens = [tok.text for tok in nlp.tokenizer(data[0])]
        entities_transformers.append(
            (tokens, label_sentence_IOB(data[0], data[1], nlp))
        )
    return entities_transformers


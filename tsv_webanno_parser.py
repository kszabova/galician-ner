from web_anno_tsv import open_web_anno_tsv
from spacy.training import offsets_to_biluo_tags, biluo_to_iob


def get_entities_from_file(file, tags):
    """
    Creates a list of data for the spaCy NER model from the specified annotated file.

    :param file: A TSV file to be parsed. Must be in WebAnno format.
    :param tags: A list of tags to be considered as entities.
    """
    entities = []
    with open_web_anno_tsv(file) as f:
        try:
            for sentence in f:
                annotations = []
                for annotation in sentence.annotations:
                    if tags is None or annotation.label in tags:
                        annotations.append(
                            (annotation.start, annotation.stop, annotation.label)
                        )
                entities.append((sentence.text, annotations))
        except:
            print(f"Error parsing file: {file}. Skipping.")
    return entities


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


def get_entities_for_spacy(files, tags):
    """
    Creates a list of data for the spaCy NER model from the specified annotated files.

    :param files: A list of XML files to be parsed. Must be in WebAnno format.
    :param tags: A list of tags to be considered as entities.
    """
    entities = []
    for file in files:
        file_ents = get_entities_from_file(file, tags)
        entities.extend(file_ents)
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


# from util import get_files_in_directory

# files = get_files_in_directory(
#     "/Users/kristina/Documents/school/upv/building_language_resources/project/data/galician"
# )
# get_entities_for_spacy(files, None)
# counter = 0
# for file in files:
#     with open_web_anno_tsv(file) as f:
#         try:
#             for sentence in f:
#                 pass
#         except:
#             print(f"Error in file {file}")
#             counter += 0
# print(counter)

with open_web_anno_tsv("/Users/kristina/Downloads/ES100947_belen.tsv") as f:
    f_iter = iter(f)
    while True:
        try:
            print(next(f_iter))
        except StopIteration:
            break
        except Exception as e:
            print(e)
            continue

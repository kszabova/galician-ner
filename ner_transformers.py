import pandas as pd
import numpy as np
import spacy
import evaluate
import json
import torch
import sys

from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

from tsv_webanno_parser import get_entities_for_transformers
from util import get_files_in_directory, split_train_test

tags = [
    "*",
    "Cell or Molecular Dysfunction",
    "Mental or Behavioral Dysfunction",
    "Signs or Symptoms",
    "Sign or Symptom",
    "Neoplastic Process",
    "Acquired Abnormality",
    "Injury or Poisoning",
    "Anatomical Abnormality",
    "Pathologic Function",
    "Congenital Abnormality",
    "Finding",
    "Disease or Syndrome",
]
tags_iob = ["O"] + [f"B-{tag}" for tag in tags] + [f"I-{tag}" for tag in tags]


def compute_metrics(eval_pred):
    np.set_printoptions(threshold=sys.maxsize)
    logits, labels = eval_pred
    labels = labels.flatten()
    predictions = np.argmax(logits, axis=-1).flatten()

    # only take into account the positions whose gold labels are not -100
    labels_selected = labels[labels != -100]
    predictions_selected = predictions[labels != -100]

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    accuracy_results = accuracy.compute(
        predictions=predictions_selected, references=labels_selected
    )
    f1_results = f1.compute(
        predictions=predictions_selected, references=labels_selected, average="micro"
    )
    return {
        "accuracy": accuracy_results,
        "f1": f1_results,
    }


def map_labels_and_ids(df):
    labels_to_ids = {label: i for i, label in enumerate(tags_iob)}
    ids_to_labels = {i: label for i, label in enumerate(tags_iob)}
    return labels_to_ids, ids_to_labels


def create_labels_for_tokenized_input(input, labels, tokenizer, labels_to_ids):
    tokenized = tokenizer(
        input, is_split_into_words=True, padding="max_length", truncation=True
    )
    word_ids = tokenized.word_ids()
    new_labels = []
    for i, word_id in enumerate(word_ids):
        if word_id is None or word_id == word_ids[i - 1]:
            new_labels.append(-100)
        else:
            new_labels.append(labels_to_ids.get(labels[word_id], labels_to_ids["O"]))
    return tokenized, new_labels


def get_collate_fn(tokenizer, labels_to_ids):
    def collate_fn(example):
        tokenized_text, labels = create_labels_for_tokenized_input(
            example["text"], example["tags"], tokenizer, labels_to_ids
        )
        return {
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
            "labels": labels,
        }

    return collate_fn


def prepare_data(data_dir, tags):
    nlp = spacy.blank("xx")
    files = get_files_in_directory(data_dir)
    data = get_entities_for_transformers(files, tags, nlp)

    train_data, test_data = split_train_test(data, random_seed=42)

    df_train = pd.DataFrame(train_data, columns=["text", "tags"])
    df_test = pd.DataFrame(test_data, columns=["text", "tags"])

    return df_train, df_test


def dataframe_to_dataset(df, tokenizer, labels_to_ids):
    return (
        Dataset.from_pandas(df)
        .map(get_collate_fn(tokenizer, labels_to_ids), batched=False)
        .remove_columns(["text", "tags"])
    )


def prepare_trainer(train, test, tags):
    model = BertForTokenClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=len(tags_iob)
    )

    training_args = TrainingArguments(
        output_dir="transformer_output",
        evaluation_strategy="epoch",
        num_train_epochs=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=compute_metrics,
    )

    return trainer


def train_model(trainer, model_dir):
    trainer.train()
    trainer.save_model(model_dir)


def run_model(train, demo, data_dir, tags, model_dir):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    if train:
        df_train, df_test = prepare_data(data_dir, tags)
        labels_to_ids, ids_to_labels = map_labels_and_ids(df_train)
        with open("labels.json", "w") as f:
            label_mapping = {
                "labels_to_ids": labels_to_ids,
                "ids_to_labels": ids_to_labels,
            }
            f.write(json.dumps(label_mapping, indent=4))
        train_dataset = dataframe_to_dataset(df_train, tokenizer, labels_to_ids)
        test_dataset = dataframe_to_dataset(df_test, tokenizer, labels_to_ids)
        trainer = prepare_trainer(train_dataset, test_dataset, tags)
        train_model(trainer, model_dir)
    if demo:
        saved_model = BertForTokenClassification.from_pretrained(model_dir)
        nlp = spacy.blank("xx")
        with open("labels.json", "r") as f:
            label_mapping = json.load(f)
            ids_to_labels = label_mapping["ids_to_labels"]
        while True:
            text = input("Enter text (or 'exit' to quit):\n")
            if text == "exit":
                break
            tokens = [tok.text for tok in nlp.tokenizer(text)]
            tokenized = tokenizer(
                tokens,
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            predicted_label_ids = saved_model(**tokenized).logits.argmax(-1)
            labels = []
            previous_word_id = None
            for word_id, label in zip(tokenized.word_ids(), predicted_label_ids[0]):
                if word_id is None or word_id == previous_word_id:
                    continue
                else:
                    labels.append(ids_to_labels[str(label.item())])
                    previous_word_id = word_id
            print(*zip(tokens, labels), sep="\n")


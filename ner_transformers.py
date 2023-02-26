import pandas as pd
import numpy as np
import spacy
import evaluate
import json

from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

from xml_webanno_parser import get_entities_for_transformers
from util import get_files_in_directory, split_train_test


def compute_metrics(eval_pred):
    # TODO this is not working
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)


def map_labels_and_ids(df):
    unique_labels = set()
    for labels in df["tags"].values.tolist():
        for label in labels:
            unique_labels.add(label)
    labels_to_ids = {label: i for i, label in enumerate(unique_labels)}
    ids_to_labels = {i: label for i, label in enumerate(unique_labels)}
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
            new_labels.append(labels_to_ids[labels[word_id]])
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
    nlp = spacy.blank("en")
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
        "bert-base-multilingual-cased", num_labels=len(tags) * 2 + 1
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
        # compute_metrics=compute_metrics,
    )

    return trainer


def train_model(trainer):
    trainer.train()
    trainer.save_model("transformer_model")


def run_model(train, demo, data_dir, tags):
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
        train_model(trainer)
    if demo:
        saved_model = BertForTokenClassification.from_pretrained("transformer_model")
        nlp = spacy.blank("en")
        with open("labels.json", "r") as f:
            label_mapping = json.load(f)
            ids_to_labels = label_mapping["ids_to_labels"]
        while True:
            text = input("Enter text (or 'exit' to quit):\n")
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
                    labels.append(ids_to_labels[label.item()])
                    previous_word_id = word_id
            print(*zip(tokens, labels), sep="\n")


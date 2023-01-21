import re

import pandas as pd
from datasets import load_dataset


def get_sents(x):
    dialogue = x['dialogue'].strip()
    sentences = re.split(r'[\s]*#Person\d#: ', dialogue)[1:]
    sentences = [stripped_sent for sent in sentences if (stripped_sent := sent.strip()) != '']
    return {'sent1': sentences[:-1], 'sent2': sentences[1:]}


def process_dataset(dataset):
    dataset = dataset.map(get_sents, remove_columns=['dialogue', 'topic', 'summary', 'id'])

    flatten_sent1 = [sent for sents in dataset['sent1'] for sent in sents]
    flatten_sent2 = [sent for sents in dataset['sent2'] for sent in sents]

    dropped_df = pd.DataFrame({'sent1': flatten_sent1, 'sent2': flatten_sent2}).drop_duplicates().dropna()
    return dropped_df


if __name__ == "__main__":
    datasets = load_dataset("knkarthick/dialogsum")

    train_data = datasets["train"]
    val_data = datasets["validation"]
    test_data = datasets["test"]

    processed_train = process_dataset(train_data)
    processed_val = process_dataset(val_data)
    processed_test = process_dataset(test_data)

    processed_train.to_csv("processed_data/train_pairs.csv")
    processed_val.to_csv("processed_data/validation_pairs.csv")
    processed_test.to_csv("processed_data/test_pairs.csv")

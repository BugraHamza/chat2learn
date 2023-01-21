from __future__ import annotations

from typing import Union

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dialogue_models.utils.base_utils import *
from dialogue_models.utils.tokenizer import *


class LSTMDataset(Dataset):
    def __init__(self, dataset_path: str, tokenizer: Union[CustomTokenizer | None]):
        super(LSTMDataset, self).__init__()

        self.data = pd.read_csv(dataset_path)
        self.tokenizer = self._get_tokenizer(tokenizer)

    def __getitem__(self, i):
        sent1 = self.data.sent1[i]
        sent2 = self.data.sent2[i]
        return self.tokenizer.tokenize(sent1, sent2)

    def __len__(self):
        return len(self.tokens)

    def _get_tokenizer(self, tokenizer) -> object:
        if tokenizer is None:
            from nltk import TreebankWordTokenizer
            from nltk import TreebankWordDetokenizer

            tok_fn = TreebankWordTokenizer().tokenize
            detok_fn = TreebankWordDetokenizer().detokenize
            tokenizer = CustomTokenizer(tokenize_fn=tok_fn, detokenize_fn=detok_fn,
                                        bos_token='<bos>', eos_token='<eos>',
                                        pad_token='<pad>', unk_token='<unk>')
            all_sents = self.data['sent1'].tolist() + self.data['sent2'].tolist()
            tokenizer.fit(all_sents)

        elif isinstance(tokenizer, str):
            tokenizer = CustomTokenizer.from_pretrained(tokenizer)

        return tokenizer


class LstmModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, n_layers=32, n_hidden=128, dropout_rate=0.2):
        super(LstmModel, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.embed = nn.Embedding(self.vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(embed_dim, n_hidden, num_layers=n_layers,
                            dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(n_hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, states=None):
        out = self.dropout(self.embed(x))
        out, states = self.lstm(out, states)
        out = self.fc(out)
        out = self.softmax(out)

        return out, states

    def init_states(self, batch_size, device):
        h = torch.zeros((self.n_layers, batch_size, self.n_hidden), device=device)
        c = torch.zeros((self.n_layers, batch_size, self.n_hidden), device=device)

        return (h, c)


class LSTMTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        self.model = LstmModel(kwargs)

    def model_forward(self, x):
        return self.model(x)

    def fit(self, dataloader, epochs, **kwargs):
        return super().fit(dataloader, epochs, **kwargs)

    def evaluate(self, dataloader, **kwargs):
        return super().evaluate(dataloader, **kwargs)

    def save(self, path):
        super().save(path)


if __name__ == '__main__':
    dataset = LSTMDataset('../dataset/processed_data/train_pairs.csv',
                          '/Users/quimba/Desktop/chat2learn/dialogue_models/dataset/processed_data/tokenizer.pkl')
    print(dataset[0])


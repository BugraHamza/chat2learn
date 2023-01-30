from __future__ import annotations

import argparse
from typing import Union

import pandas as pd

import torch.nn as nn
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from dialogue_models.utils.base_utils import *
from dialogue_models.utils.tokenizer import *

import matplotlib.pyplot as plt


class LSTMDataset(Dataset):
    LONGEST = 0
    AS_IS = -1

    def __init__(self, dataset_path: str, tokenizer: Union[CustomTokenizer | None], **kwargs):
        super(LSTMDataset, self).__init__()

        self.data = pd.read_csv(dataset_path)
        self.tokenizer = self._get_tokenizer(tokenizer)
        self.device = kwargs.get('device', 'cpu')
        self.max_len = kwargs.get('max_len', self.AS_IS)

        if self.max_len == self.LONGEST:
            sent1_max = self.data.sent1.apply(lambda x: len(self.tokenizer.tokenize(x))).max()
            sent2_max = self.data.sent2.apply(lambda x: len(self.tokenizer.tokenize(x))).max()
            self.max_len = 3 + sent1_max + sent2_max

    def __getitem__(self, i):
        sent1 = self.data.sent1[i]
        sent2 = self.data.sent2[i]

        return self.tokenizer.tokenize(sent1, sent2, device=self.device, max_len=self.max_len),

    def __len__(self):
        return len(self.data)

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
    def __init__(self, vocab_size, embed_dim=128, n_layers=4, n_hidden=128, dropout_rate=0.2, **kwargs):
        super(LstmModel, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.pad_id = kwargs.get('pad_id', 0)

        self.embed = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(embed_dim, n_hidden, num_layers=n_layers,
                            dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(n_hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        max_len = x.shape[1]
        out = self.dropout(self.embed(x))
        out = pack_padded_sequence(out, lengths=x.ne(self.pad_id).sum(dim=1).cpu(),
                                   batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(out)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=max_len)
        out = self.fc(out)
        out = self.softmax(out)

        return out

    def init_states(self, batch_size, device):
        h = torch.zeros((self.n_layers, batch_size, self.n_hidden), device=device)
        c = torch.zeros((self.n_layers, batch_size, self.n_hidden), device=device)

        return (h, c)


class LSTMTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(LSTMTrainer, self).__init__(**kwargs)
        self.device = kwargs.get('device', 'cpu')
        self.model = kwargs['model'].to(self.device)

    def model_forward(self, x):
        return self.model(x)

    def fit(self, dataloader, epochs, **kwargs):
        return super().fit(dataloader, epochs, **kwargs)

    def evaluate(self, dataloader, **kwargs):
        return super().evaluate(dataloader, **kwargs)

    def save(self, path):
        script_model = torch.jit.script(self.model)
        script_model.save(path)


def train_lstm(train_path: str, tokenizer_path: str, **kwargs):
    max_len = kwargs.get('max_len', 100)
    device = kwargs.get('device', 'cpu')
    batch_size = kwargs.get('batch_size', 8)
    validation_path = kwargs.get('valset_path', None)
    reduction_method = kwargs.get('reduction_method', 'mean')
    metrics = kwargs.get('metrics', 'rouge')
    model_path = kwargs.get('model_path', None)
    model_kwargs = kwargs.get('model_kwargs', {})

    train_dataset = LSTMDataset(train_path,
                                tokenizer_path,
                                max_len=max_len, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    tokenizer = train_dataset.tokenizer
    pad_id = tokenizer.w2i[tokenizer.pad_token]
    model = LstmModel(vocab_size=len(tokenizer), pad_id=pad_id, **model_kwargs)

    trainer = LSTMTrainer(model=model, criterion=nn.NLLLoss(),
                          metrics=metrics, tokenizer=tokenizer, device=device)

    fitH = trainer.fit(train_loader, epochs=1, optimizer=optim.Adadelta, learning_rate=1)

    if validation_path is not None:
        val_dataset = LSTMDataset(validation_path, tokenizer_path, max_len=max_len, device=device)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        eval_H = trainer.evaluate(val_loader, reduction_method=reduction_method)

    if model_path is not None:
        trainer.save(model_path)

    return fitH, eval_H


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train-path', type=str, required=True)
    arg_parser.add_argument('--val-path', type=str, required=False)
    arg_parser.add_argument('--tokenizer-path', type=str, required=True)
    arg_parser.add_argument('--model-path', type=str, required=False)
    arg_parser.add_argument('--max-len', type=int, required=False)
    arg_parser.add_argument('--batch_size', type=int, required=False)
    arg_parser.add_argument('--device', type=str, required=False)
    arg_parser.add_argument('--fig-path', type=str, required=False)
    arg_parser.add_argument('--reduction-method', type=str, required=False)
    arg_parser.add_argument('--metrics', type=list, required=False)
    arg_parser.add_argument('--embed-dim', type=int, required=False)
    arg_parser.add_argument('--n-hidden', type=int, required=False)
    arg_parser.add_argument('--n-layers', type=int, required=False)
    arg_parser.add_argument('--dropout-rate', type=float, required=False)

    args = arg_parser.parse_args()
    print(vars(args))
    fitH, eval_H = train_lstm(**vars(args))

    if args.fig_path is not None:
        plt.plot(fitH, 'r', label='Training')
        plt.plot(eval_H, 'b', label='Validation')
        plt.legend()
        plt.savefig(args.fig_path)

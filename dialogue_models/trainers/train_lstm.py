from __future__ import annotations

import argparse
from multiprocessing import Process
from typing import Union

import pandas as pd

import torch.nn as nn
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from dialogue_models.utils.base_utils import *
from dialogue_models.utils.tokenizer import *

import optuna


class LSTMDataset(Dataset):
    LONGEST = 0
    AS_IS = -1

    def __init__(self, dataset_path: str, tokenizer: Union[CustomTokenizer | None], **kwargs):
        super(LSTMDataset, self).__init__()

        self.data = pd.read_csv(dataset_path)
        self.tokenizer = self._get_tokenizer(tokenizer)
        self.device = kwargs.get('device', 'cpu')
        self.max_len = kwargs.get('max_len', LSTMDataset.AS_IS)

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
        self.lstm = nn.LSTM(embed_dim, n_hidden, num_layers=n_layers, batch_first=True)
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
        super(LSTMTrainer, self).__init__(kwargs['model'], kwargs['criterion'],
                                          kwargs.get('metric_name', None),
                                          kwargs.get('tokenizer', None),
                                          kwargs.get('device', 'cpu'))

        self.model = kwargs['model'].to(self.device)

    def model_forward(self, x):
        return self.model(x)

    def fit(self, dataloader, epochs, **kwargs):
        kwargs = {**kwargs, 'epochs': epochs}
        return super().train(dataloader, **kwargs)

    def evaluate(self, dataloader, **kwargs):
        return super().evaluate(dataloader, **kwargs)

    def save(self, path):
        script_model = torch.jit.script(self.model)
        script_model.save(path)


def train_lstm(train_path: str, tokenizer_path: str, **kwargs):
    max_len = kwargs.get('max_len', 100)
    device = kwargs.get('device', 'cpu')
    batch_size = kwargs.get('batch_size', 8)
    learning_rate = kwargs.get('learning_rate', 1.0)
    epochs = kwargs.get('epochs', 1)
    validation_path = kwargs.get('valset_path', None)
    metric = kwargs.get('metric', 'rouge')
    model_path = kwargs.get('model_path', None)

    # model params
    embed_dim = kwargs.get('embed_dim', 64)
    n_layers = kwargs.get('n_layers', 1)
    n_hidden = kwargs.get('n_hidden', 128)
    dropout_rate = kwargs.get('dropout_rate', 0.2)

    train_dataset = LSTMDataset(train_path,
                                tokenizer_path,
                                max_len=max_len, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    tokenizer = train_dataset.tokenizer
    pad_id = tokenizer.w2i[tokenizer.pad_token]
    model = LstmModel(vocab_size=len(tokenizer), pad_id=pad_id, embed_dim=embed_dim, n_layers=n_layers,
                      n_hidden=n_hidden, dropout_rate=dropout_rate)

    trainer = LSTMTrainer(model=model, criterion=nn.NLLLoss(),
                          metric_name=metric, tokenizer=tokenizer, device=device)

    trainer.fit(train_loader, epochs=epochs, optimizer=optim.Adadelta, learning_rate=learning_rate)

    if validation_path is not None:
        val_dataset = LSTMDataset(validation_path, tokenizer_path, max_len=max_len, device=device)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_loss, val_metric = trainer.evaluate(val_loader)

    if model_path is not None:
        trainer.save(model_path)

    return val_metric


def objective(trial, device='cpu'):
    # constant params
    train_path = 'dialogue_models/dataset/processed_data/train_pairs.csv'
    val_path = 'dialogue_models/dataset/processed_data/val_pairs.csv'
    tokenizer_path = 'dialogue_models/dataset/tokenizers/tokenizer.pkl'
    dropout_rate = 0.2
    max_len = 100

    # model params
    embed_dim = trial.suggest_int('embed_dim', 32, 256)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    n_hidden = trial.suggest_int('n_hidden', 64, 256)

    batch_size = trial.suggest_int('batch_size', 4, 64)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)

    return train_lstm(train_path=train_path, valset_path=val_path, tokenizer_path=tokenizer_path, metric='rouge',
                      embed_dim=embed_dim, n_layers=n_layers, n_hidden=n_hidden, dropout_rate=dropout_rate,
                      batch_size=batch_size, learning_rate=learning_rate, max_len=max_len, epochs=7, device=device)


def main(device):
    study = optuna.create_study(study_name='lstm_optimization', storage='sqlite:///lstm_optimization.db', direction='maximize',
                                load_if_exists=True, pruner=optuna.pruners.SuccessiveHalvingPruner())

    study.optimize(lambda trial: objective(trial, device=device), n_trials=100,  gc_after_trial=True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('--train-path', type=str, required=True)
    # arg_parser.add_argument('--val-path', type=str, required=False)
    # arg_parser.add_argument('--tokenizer-path', type=str, required=True)
    # arg_parser.add_argument('--model-path', type=str, required=False)
    # arg_parser.add_argument('--max-len', type=int, default=100)
    # arg_parser.add_argument('--batch_size', type=int, default=8)
    arg_parser.add_argument('--device', type=str, default='cpu')
    # arg_parser.add_argument('--metric', type=str, default='rouge')
    # arg_parser.add_argument('--embed-dim', type=int, default=64)
    # arg_parser.add_argument('--n-hidden', type=int, default=128)
    # arg_parser.add_argument('--n-layers', type=int, default=1)
    # arg_parser.add_argument('--dropout-rate', type=float, default=0.2)

    args = arg_parser.parse_args()
    # val_metric = train_lstm(**{k: v for k, v in vars(args).items() if v is not None})

    if args.device == 'cpu' or args.device == 'cuda':
        main(args.device)

    elif args.device == 'double_cuda':
        p1 = Process(target=main, args='cuda:0')
        p2 = Process(target=main, args='cuda:1')

        p1.start()
        p2.start()

        p1.join()
        p2.join()

    else:
        raise ValueError('Invalid device name! (cpu, cuda, double_cuda). You are probably using a TPU.')

import re

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torch.optim import *

from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nltk.tokenize.treebank import TreebankWordDetokenizer

from datasets import load_dataset

from tqdm.notebook import tqdm

import optuna


class LMDataset(Dataset):
    def __init__(self, dataset, tokenizer, vocab, maxlen, special_tokens, device):
        super(LMDataset, self).__init__()

        self.maxlen = maxlen
        self.special_tokens = special_tokens

        self.tokenizer = get_tokenizer('spacy') if tokenizer is None else tokenizer
        self.vocab = self.get_vocab(dataset) if vocab is None else vocab

        sentence_pairs = dataset.map(self.create_sentence_pairs)['sentence_pairs']
        flatten_tokens = [sent for sents in sentence_pairs for sent in sents]
        self.tokens = torch.zeros((len(flatten_tokens), self.maxlen), dtype=torch.long, device=device)
        for i, sent in enumerate(flatten_tokens):
            self.tokens[i, :] = torch.tensor(self.vocab(sent))

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def create_sentence_pairs(self, x):
        sentences = re.split(r'[\s]*#Person\d#: ', x['dialogue'])[1:]

        sentence_pairs = []
        for sent1, sent2 in zip(sentences[:-1], sentences[1:]):
            sent_pair = ' '.join([self.special_tokens['bos_token'], sent1,
                                  self.special_tokens['pad_token'], sent2])

            sent = [tok for tok in self.tokenizer(sent_pair)]
            sent = sent[:self.maxlen]
            sent = sent + [self.special_tokens['eos_token']] * (self.maxlen - len(sent))
            sentence_pairs.append(sent)

        return {'sentence_pairs': sentence_pairs}

    def get_vocab(self, dataset):
        sentence_pairs = dataset.map(self.create_sentence_pairs)['sentence_pairs']
        flatten_tokens = [sent for sents in sentence_pairs for sent in sents]
        vocab = build_vocab_from_iterator(flatten_tokens, min_freq=5,
                                          specials=list(self.special_tokens.values()))
        vocab.set_default_index(vocab['|UNK|'])

        return vocab


class LstmModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, n_layers=32, n_hidden=128, dropout_rate=0.2):
        super(LstmModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate

        self.embed = nn.Embedding(self.vocab_size, embedding_dim=self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.n_hidden, num_layers=self.n_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

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


def train_step(model, loader, optimizer, criterion, device):
    model.train()

    pbar = tqdm(loader)
    batch_losses = []

    for i, batch in enumerate(pbar):
        batch = batch.to(device)

        states = model.init_states(batch.size(0), device=device)

        optimizer.zero_grad()
        states = [state.detach() for state in states]
        y_pred, states = model(batch[:, :-1], states)
        loss = criterion(y_pred.moveaxis(1, -1), batch[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        batch_losses.append(loss.item())
        pbar.set_description(f'Batch Loss: {loss.item():.3f} Train Loss: {np.mean(batch_losses):.3f}')

    return np.mean(batch_losses)


def eval_step(model, loader, criterion, device):
    model.eval()

    pbar = tqdm(loader)
    batch_losses = []

    for i, batch in enumerate(pbar):
        batch = batch.to(device)

        y_pred, _ = model(batch[:, :-1])
        loss = criterion(y_pred.moveaxis(1, -1), batch[:, 1:])

        batch_losses.append(loss.item())
        pbar.set_description(f'Batch Loss: {loss.item():.3f} Validation Loss: {np.mean(batch_losses):.3f}')

    return np.mean(batch_losses)


def answer(model, sent, tokenizer, vocab, maxlen, special_tokens, device):
    model.eval()

    detokenizer = TreebankWordDetokenizer()

    sent = ' '.join([special_tokens['bos_token'], sent, special_tokens['pad_token']])
    sent = vocab(tokenizer(sent))
    sent = torch.tensor(sent, device=device)

    with torch.no_grad():
        y_pred, states = model(sent)

        pred_tokens = y_pred.argmax(dim=-1, keepdim=True)
        # sent_preds = vocab.lookup_tokens(list(pred_tokens))

        answer = []
        for j in range(maxlen - len(sent)):
            last_idx = pred_tokens[-1]
            answer.append(vocab.lookup_token(last_idx))

            if answer[-1] == special_tokens['eos_token']:
                break

            y_pred, states = model(last_idx, states)
            pred_tokens = y_pred.argmax(dim=-1, keepdim=True)

        return detokenizer.detokenize(answer)


def train(max_len, epochs, bs, lr, embed_dim, n_layers, n_hidden, device):
    special_tokens = {'bos_token': '|BOS|',
                      'pad_token': '|PAD|',
                      'eos_token': '|EOS|',
                      'unk_token': '|UNK|'}

    train_dataset = load_dataset('knkarthick/dialogsum', split='train')
    val_dataset = load_dataset('knkarthick/dialogsum', split='validation')

    lm_train = LMDataset(train_dataset, None, None, max_len, special_tokens, device)
    lm_valid = LMDataset(val_dataset, lm_train.tokenizer, lm_train.vocab, max_len, special_tokens, device)

    train_loader = DataLoader(lm_train, batch_size=bs, shuffle=True)
    val_loader = DataLoader(lm_valid, batch_size=bs)

    model = LstmModel(len(lm_train.vocab), embed_dim=embed_dim, n_layers=n_layers, n_hidden=n_hidden, dropout_rate=0.1)
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        val_loss = eval_step(model, val_loader, criterion, device)
    print(answer(model, 'Hi, how are you?', lm_train.tokenizer, lm_train.vocab, max_len, special_tokens, device))

    return val_loss


def objective(trial):
    params = {'bs': trial.suggest_int('bs', 8, 64),
              'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
              'embed_dim': trial.suggest_int('embed_dim', 16, 256, log=True),
              'n_layers': trial.suggest_int('n_layers', 1, 8),
              'n_hidden': trial.suggest_int('n_hidden', 16, 256, log=True)}

    return train(max_len=100, epochs=10, device='cuda', **params)


def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    print(study.best_params)


if __name__ == '__main__':
    main()

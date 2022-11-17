import os
import re
from typing import Union, List

import torch
from torch import nn
from torchtext.vocab import GloVe

from base_utils import BaseChatter, BaseTokenizer, BaseModel, DataPreparer


class LSTMChatter(BaseChatter):
    def tokenize(self, text: str) -> List[str]:
        pass

    def from_pretrained(self, model_path: str, tokenizer_path: str):
        pass

    def chat(self, text: str) -> str:
        pass

class LSTMTokenizer(BaseTokenizer):
    def __init__(self, vocab_path=None, split_regex: str = r'\s+', **kwargs):
        super(LSTMTokenizer, self).__init__(split_regex=split_regex, **kwargs)
        self.vocab_path = vocab_path
        self.glove_obj = kwargs['glove_obj'] if 'glove_obj' in kwargs.keys() else None

        if self.vocab_path and os.path.exists(self.vocab_path):
            print('[INFO]: Loading vocab from', self.vocab_path)
            with open(self.vocab_path, 'r') as f:
                for idx, word in enumerate(f):
                    self.vocab2idx[word.strip()] = idx
                    self.idx2vocab[idx] = word.strip()
        elif self.glove_obj:
            # add special tokens to glove embeddings as zero vectors
            self.glove_obj.vectors = torch.cat(
                [self.glove_obj.vectors, torch.zeros(len(self.special_tokens), self.glove_obj.dim)], dim=0)

            # add special tokens to glove_obj
            for token in self.special_tokens.values():
                if token not in self.glove_obj.stoi.keys():
                    self.glove_obj.stoi[token] = len(self.glove_obj.stoi)
                    self.glove_obj.itos.append(token)

            # use glove_obj to create vocab
            self.vocab2idx = self.glove_obj.stoi
            self.idx2vocab = self.glove_obj.itos
        else:
            raise ValueError(f'Either vocab path or glove_obj must be provided')

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        # remove all non-alphanumeric characters
        texts = [re.sub(r'[^a-zA-Z0-9\s]', '', text) for text in texts]
        return super(LSTMTokenizer, self)._tokenize(texts)


class LSTMModel(nn.Module, BaseModel):
    def __init__(self, padding_idx: int = -1, **kwargs):
        super(LSTMModel, self).__init__()

        self.padding_idx = padding_idx

        if 'glove_obj' in kwargs.keys():
            self.glove_obj = kwargs['glove_obj']
            self.vocab_size = len(self.glove_obj.stoi)
            self.embedding_dim = self.glove_obj.vectors.shape[1]
        elif 'vocab_size' in kwargs.keys():
            self.vocab_size = kwargs['vocab_size']
            self.embedding_dim = kwargs['embedding_dim']
        else:
            raise ValueError('Either glove_obj or vocab_size must be provided')

        if self.glove_obj:
            # use glove embeddings
            self.embedding = nn.Embedding.from_pretrained(self.glove_obj.vectors, freeze=True,
                                                          padding_idx=self.padding_idx)
        else:
            # use random embeddings
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)

        self.fc = nn.Linear(128, self.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

    def fit(self, dataloader, **kwargs):
        epochs = kwargs['epochs'] if 'epochs' in kwargs.keys() else 1
        optimizer = kwargs['optimizer'] if 'optimizer' in kwargs.keys() else torch.optim.Adam(self.parameters())
        criterion = kwargs['criterion'] if 'criterion' in kwargs.keys() else nn.CrossEntropyLoss()
        device = kwargs['device'] if 'device' in kwargs.keys() else 'cpu'

        # move model to device
        self.to(device)

        self.train()
        for epoch in range(epochs):
            for idx, (x, y) in enumerate(dataloader, start=1):
                optimizer.zero_grad()
                y_pred = self(x)
                loss = criterion(y_pred, y[:, 0])
                loss.backward()
                optimizer.step()

                print(f'Epoch: {epoch + 1}/{epochs}, Batch: {idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

    def predict(self, x, eos_idx: int = 1, max_len: int = 100):
        x = torch.tensor(x)
        for i in range(max_len):
            y_pred = self(x)
            y_pred = torch.argmax(y_pred, dim=1)
            x = torch.cat([x, y_pred.unsqueeze(1)], dim=1)
            if y_pred == eos_idx:
                break
        return x

    def evaluate(self, dataloader, **kwargs):
        device = kwargs['device'] if 'device' in kwargs.keys() else 'cpu'
        self.to(device)

        self.eval()
        loss = 0
        for idx, (x, y) in enumerate(dataloader, start=1):
            y_pred = self(x)
            loss += (y_pred.argmax(dim=-1) == y[:, 0]).sum().item()
            print(f'Batch: {idx}/{len(dataloader)}, Accuracy: {loss / idx:.4f}')

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f'[INFO]: Model saved to {path}')

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f'[INFO]: Model loaded from {path}')


if __name__ == '__main__':
    # load tokenizer
    # tokenizer = LSTMTokenizer()

    # HYPERPARAMETERS
    BS = 256
    LR = 0.001
    EPOCHS = 100

    # load glove embeddings
    glove_obj = GloVe(name='840B', dim=300)

    tokenizer = LSTMTokenizer(glove_obj=glove_obj, special_tokens={'pad': '<PAD>', 'eos': '<EOS>', 'bos': '<BOS>'})

    # load model
    model = LSTMModel(glove_obj=glove_obj)

    # initialize data preparer for seq2seq training
    data_prep = DataPreparer(tokenizer, 'knkarthick/dialogsum', 'next_word_prediction', is_lowercase=True)

    # get train and test dataloaders
    train_dataloader = data_prep('train', 'dialogue', batch_size=BS, shuffle=True)
    test_dataloader = data_prep('validation', 'dialogue', batch_size=BS, shuffle=True)

    # fit model
    # model.fit(train_dataloader, epochs=1, criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters()))

    # evaluate model
    # model.evaluate(test_dataloader)

    # predict
    # sent = 'What will you do today?'
    # enc_sent = tokenizer.encode(sent)
    # pred = model.predict(enc_sent, eos_idx=tokenizer.special_tokens['eos'], max_len=100)

    # print(f'Input: {sent}')
    # print(f'Output: {tokenizer.decode(pred)}')

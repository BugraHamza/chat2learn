import os
import re
from typing import Union, List

import torch
from torch import nn
from torchtext.data import get_tokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

from base_utils import BaseChatter


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, n_layers=32, n_hidden=128):
        super(LanguageModel, self).__init__()
        DROPOUT_RATE = 0.2

        self.embed = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(embed_dim, n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, vocab_size)

        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x, states=None):
        x = self.dropout(self.embed(x))
        x, states = self.lstm(x, states)
        x = self.fc(x)

        return x, states


class LSTMChatter(BaseChatter):
    def __init__(self, model_path, tokenizer_path):
        self.model_path = model_path

        self.tokenizer = get_tokenizer('spacy')
        self.detokenizer = TreebankWordDetokenizer()
        self.vocab = torch.load(tokenizer_path)

        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()

        self.special_tokens = {'bos_token': '|BOS|', 'pad_token': '|PAD|',
                               'eos_token': '|EOS|', 'unk_token': '|UNK|'}

    def tokenize(self, text: str) -> List[str]:
        text = ' '.join([self.special_tokens['bos_token'], text, self.special_tokens['pad_token']])
        text = self.vocab(self.tokenizer(text))
        return torch.tensor(text)

    def chat(self, text: str, states=None) -> str:
        tokenized_text = self.tokenize(text)

        if states is None:
            n_layer = self.model.lstm.num_layers
            n_hidden = self.model.lstm.hidden_size
            states = (torch.zeros(n_layer, n_hidden), torch.zeros(n_layer, n_hidden))

        clip = lambda x: 10 * (x * (abs(x) < 0.2))
        states = [clip(state) for state in states]

        y_pred, states = self.model(tokenized_text, states=states)

        answer = []
        for i in range(100):
            last_ids = y_pred.argsort(dim=-1, descending=True)[-1]
            for idx in last_ids:
                last_token = self.vocab.lookup_token(idx.item())
                if last_token in ['|BOS|', '|PAD|', '|EOS|']:
                    break
                elif last_token != '|UNK|':
                    answer.append(last_token)
                    break

            if last_token == self.special_tokens['eos_token']:
                break

            y_pred, states = self.model(idx.unsqueeze(0), states)

        return self.detokenizer.detokenize(answer), states


if __name__ == '__main__':
    lstm_chatter = LSTMChatter(model_path='../saved models/lstm-models/Model_50.pt',
                               tokenizer_path='../saved models/lstm_model/lstm_tokenizer.pth')

    states = None
    while True:
        sent = input("You: ")
        ans, states = lstm_chatter.chat(sent, states)
        print(f'Lissy: {ans}')

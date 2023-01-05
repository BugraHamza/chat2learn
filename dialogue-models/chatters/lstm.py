from typing import List

import torch
from torch import nn
from torchtext.data import get_tokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

from base_utils import BaseChatter, beam_search


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
                            dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.vocab_size)
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

    def detokenize(self, tokens) -> str:
        tokens = self.vocab.lookup_tokens(tokens.tolist())
        tokens = [token for token in tokens if token not in self.special_tokens.values()]

        return self.detokenizer.detokenize(tokens)

    def chat(self, text: str, states=None) -> str:
        tokenized_text = self.tokenize(text)
        tokens, states = beam_search(self.model, tokenized_text, num_beams=3, max_new_token=30,
                                     bos_token_id=self.vocab[self.special_tokens['bos_token']],
                                     eos_token_id=self.vocab[self.special_tokens['eos_token']],
                                     states=states)
        return self.detokenize(tokens), states


if __name__ == '__main__':
    lstm_chatter = LSTMChatter(model_path='../saved models/lstm-models/Model_50.pt',
                               tokenizer_path='../saved models/lstm-models/lstm-tokenizer.pth')

    states = None
    i = 0
    while True:
        sent = input("You: ")
        ans, states = lstm_chatter.chat(sent, states=states)
        print(f'Lissy: {ans}')

        i += 1
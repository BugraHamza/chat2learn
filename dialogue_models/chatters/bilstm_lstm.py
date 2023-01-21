from typing import List

import torch
from base_utils import BaseChatter, beam_search
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data import get_tokenizer


class EncoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, n_layers=32, n_hidden=128, dropout_rate=0.2):
        super(EncoderModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate

        self.embed = nn.Embedding(self.vocab_size, embedding_dim=self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.n_hidden, num_layers=self.n_layers, bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, lengths=None):
        out = self.dropout(self.embed(x))

        if lengths is not None:
            out = pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)
        _, states = self.lstm(out)

        return states


class DecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, n_layers=32, n_hidden=128, dropout_rate=0.2):
        super(DecoderModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate

        self.embed = nn.Embedding(self.vocab_size, embedding_dim=self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.n_hidden, num_layers=self.n_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, states, lengths=None):
        out = self.dropout(self.embed(x))
        if lengths is not None:
            out = pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)
        out, states = self.lstm(out, states)
        out = self.fc(out)
        out = self.softmax(out)

        return out, states


class BiLSTMChatter(BaseChatter):
    def __init__(self, encoder_model_path, decoder_model_path, tokenizer_path):
        self.tokenizer = get_tokenizer('spacy')
        self.detokenizer = TreebankWordDetokenizer()
        self.vocab = torch.load(tokenizer_path)

        self.encoder_model = torch.load(encoder_model_path, map_location='cpu')
        self.decoder_model = torch.load(decoder_model_path, map_location='cpu')
        self.encoder_model.eval(), self.decoder_model.eval()

        self.special_tokens = {'bos_token': '|BOS|', 'pad_token': '|PAD|',
                               'eos_token': '|EOS|', 'unk_token': '|UNK|'}

    def tokenize(self, text: str) -> List[str]:
        text = ' '.join([self.special_tokens['bos_token'], text, self.special_tokens['eos_token']])
        text = self.vocab(self.tokenizer(text))
        return torch.tensor(text)

    def detokenize(self, tokens) -> str:
        tokens = self.vocab.lookup_tokens(tokens.tolist())
        tokens = [token for token in tokens if token not in self.special_tokens.values()]

        return self.detokenizer.detokenize(tokens)

    def chat(self, text: str) -> str:
        tokenized_question = self.tokenize(text)
        encoder_states = self.encoder_model(tokenized_question)

        tokenized_bos = self.vocab[self.special_tokens['bos_token']]
        tokenized_bos = torch.tensor([tokenized_bos])

        tokens, _ = beam_search(self.decoder_model, tokenized_bos, num_beams=3, max_new_token=30,
                                eos_token_id=self.vocab[self.special_tokens['eos_token']],
                                states=encoder_states)
        return self.detokenize(tokens), encoder_states


if __name__ == '__main__':
    bilstm_chatter = BiLSTMChatter(encoder_model_path='../saved trainers/bilstm_lstm-trainers/encoder_model8.pt',
                                   decoder_model_path='../saved trainers/bilstm_lstm-trainers/decoder_model8.pt',
                                   tokenizer_path= '../saved trainers/bilstm_lstm-trainers/bilstm_tokenizer.pth')

    states = None
    while True:
        sent = input("You: ")
        #start = time()
        ans, states = bilstm_chatter.chat(sent)
        #print("Time: ", time() - start)
        print(f'Billy Lake: {ans}')

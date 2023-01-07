import re
from typing import List, Tuple, Union, Any

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, Dataset


class BaseChatter:
    def __init__(self):
        pass

    def tokenize(self, text: str) -> List[str]:
        pass

    def detokenize(self, tokens: List[str]) -> str:
        pass

    def chat(self, text: str) -> str:
        pass


class BaseModel:
    def __init__(self):
        pass

    def fit(self, dataloader, **kwargs):
        raise NotImplementedError('fit not implemented')

    def predict(self, x):
        raise NotImplementedError('predict not implemented')

    def evaluate(self, dataloader, **kwargs):
        raise NotImplementedError('eval not implemented')

    def save(self, path):
        raise NotImplementedError('save not implemented')

    def load(self, path):
        raise NotImplementedError('load not implemented')


class BaseTokenizer:
    def __init__(self, split_regex: str = r'\s+', **kwargs):
        self.split_regex: str = split_regex

        self.special_tokens: dict = kwargs['special_tokens'] if 'special_tokens' in kwargs.keys() else {}
        self.vocab2idx: dict = {}
        self.idx2vocab: dict = {}

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        tok_texts: List = []

        # for each text in texts list
        for text in texts:
            # split text into tokens
            text = re.split(self.split_regex, text)

            # for each token in text,
            # if token is in vocab, keep token
            # else, pass
            tok_text = []
            for word in text:
                if word in self.vocab2idx.keys():
                    tok_text.append(word)

            # add tok_text to tok_texts
            tok_texts.append(tok_text)

        return tok_texts

    def _detokenize(self, decoded_texts: List[List[str]], include_special_tokens: bool = True) -> List[str]:
        # join tokens into text
        # if include_special_tokens is False, remove the tokens that starts with < and ends with >
        texts = []
        for text in decoded_texts:
            if include_special_tokens:
                texts.append(' '.join(text))
            else:
                texts.append(' '.join([word for word in text if word not in self.special_tokens.values()]))
        return texts

    def encode(self, texts, max_length=None) -> List[List[int]]:
        if isinstance(texts, str):
            texts = [texts]

        if max_length is None:
            max_length = max([len(re.split(self.split_regex, text)) for text in texts])

        # takes list of string and encode them into list of list of int
        tokenized_texts = self._tokenize(texts=texts)

        encoded_texts = []
        for text in tokenized_texts:
            enc_text = []
            for word in text:
                if word in self.vocab2idx.keys():
                    enc_text.append(self.vocab2idx[word])
                elif self.vocab_path is not None and 'unk' in self.special_tokens.keys():
                    enc_text.append(self.vocab2idx[self.special_tokens['unk']])

            encoded_texts.append(
                enc_text[:max_length] + (max_length - len(enc_text)) * [self.vocab2idx[self.special_tokens['pad']]])

        return encoded_texts

    def decode(self, tokens: List[List[int]], special_tokens: bool = True) -> List[str]:
        # takes list of list of int and decode them into list of string
        decoded_texts = [list(map(lambda x: self.idx2vocab[x], text)) for text in tokens]
        return self._detokenize(decoded_texts=decoded_texts, include_special_tokens=special_tokens)

    def fit(self, texts: List[str]):
        # get max length for tokenization
        max_length = max([len(re.split(self.split_regex, text)) for text in texts])

        # get tokenized texts
        tokenized_texts = self._tokenize(texts=texts, max_length=max_length)

        # for each token in tokenized texts, add to vocab
        for text in tokenized_texts:
            for token in text:
                if token not in self.vocab2idx.keys():
                    self.vocab2idx[token] = len(self.vocab2idx)
                    self.idx2vocab[len(self.idx2vocab)] = token

        # add special tokens to vocab
        for token in self.special_tokens.values():
            if token not in self.vocab2idx.keys():
                self.vocab2idx[token] = len(self.vocab2idx)
                self.idx2vocab[len(self.idx2vocab)] = token

    def save(self, path):
        with open(path, 'w') as f:
            for word in self.vocab2idx.keys():
                f.write(f'{word}\n')

        print(f'[INFO]: Vocab saved to {path}')

    def load(self, path):
        print(f'[INFO]: Loading vocab from {path}')
        with open(path, 'r') as f:
            for idx, word in enumerate(f):
                self.vocab2idx[word.strip()] = idx
                self.idx2vocab[idx] = word.strip()

        print('[INFO]: Use special_tokens argument to include special tokens in decoding')

    def __len__(self):
        return len(self.vocab2idx)


def beam_search(model, tokenized_sent, num_beams, max_new_token, eos_token_id, **kwargs):
    # generate function for beam search decoding
    # model: model to generate
    # tokenized_sent: tokenized input sentence
    # num_beams: number of beams
    # max_new_token: maximum number of new tokens to generate
    # eos_token_id: end of sentence token_id for early stopping
    # **kwargs: additional arguments for model forward pass

    # get first beams
    out, states = model(tokenized_sent, kwargs.get('states'))
    top_preds = out[-1, :].topk(num_beams, dim=-1)

    # add predictions to beams
    beams = [(pred.unsqueeze(0), states, score.item()) for pred, score in zip(top_preds.indices, top_preds.values)]

    for i in range(max_new_token - 1):
        new_beams = []
        for beam in beams:
            if beam[0][-1] == eos_token_id:
                new_beams.append(beam)
                continue

            out, states = model(beam[0][-1:], beam[1])
            top_preds = out[-1, :].topk(num_beams, dim=-1)

            for pred, score in zip(top_preds.indices, top_preds.values):
                new_beams.append((torch.cat((beam[0], pred.unsqueeze(0)), dim=0), states, beam[2] + score.item()))

        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:num_beams]

        if beams[0][0][-1] == eos_token_id:
            break

    # return the best beam and its states
    return beams[0][0], beams[0][1]

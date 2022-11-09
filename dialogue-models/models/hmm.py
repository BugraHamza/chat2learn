import os
from typing import Union, List
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm, trange

from base_utils import BaseModel, BaseTokenizer, DataPreparer


class HiddenMarkovTokenizer(BaseTokenizer):
    def __init__(self, vocab_path: Union[None|str] = None, split_regex: str = r'\s+', **kwargs):
        super(HiddenMarkovTokenizer, self).__init__(split_regex, **kwargs)
        self.vocab_path: str = vocab_path

        # if no vocab file is provided, warn user
        if self.vocab_path is None:
            print('[INFO]: No vocab path specified, tokenizer can only be used to fit a vocab')
        elif os.path.exists(self.vocab_path):
            print('[INFO]: Loading vocab from', self.vocab_path)
            with open(self.vocab_path, 'r') as f:
                for idx, word in enumerate(f):
                    self.vocab2idx[word.strip()] = idx
                    self.idx2vocab[idx] = word.strip()
        else:
            raise ValueError(f'Vocab path {self.vocab_path} does not exist')


class HiddenMarkovModel(BaseModel):
    def __init__(self, vocab_size: int, bos_idx: int, **kwargs):
        if vocab_size < 1:
            raise ValueError('Vocab size must be greater than 0')

        self.vocab_size = vocab_size
        self.bos_idx = bos_idx

        if 'initial_probs' in kwargs.keys() and 'transition_probs' in kwargs.keys() and 'emission_probs' in kwargs.keys():
            self.initial_probs = kwargs['initial_probs']
            self.transition_probs = kwargs['transition_probs']
            self.emission_probs = kwargs['emission_probs']
        else:
            self.initial_probs = np.zeros(self.vocab_size)
            self.transition_probs = np.zeros((self.vocab_size, self.vocab_size))
            self.emission_probs = np.zeros((self.vocab_size, self.vocab_size))

    def fit(self, train_loader, method: str = 'frequency'):
        # this method calculates the probabilities for initial, transition, and emission matrices
        # initial probabilities are the probabilities of P(w|<BOS>)
        # transition probabilities are the probabilities of P(w|w-1)
        # emission probabilities are the probabilities of P(person1_w|person2_w)

        for (inputs, targets) in tqdm(train_loader):
            bs = len(inputs)
            for i in range(bs):
                # get input and target
                inp, trg = inputs[i], targets[i]

                # the tokens from person1 updates the emission matrix
                for prev_w, next_w in torch.stack((inp[1:], trg[1:]), dim=1):
                    self.emission_probs[prev_w, next_w] += 1

                # the tokens from person2 updates the transition matrix
                for per1, per2 in torch.stack((inp[:-1], inp[1:]), dim=1):
                    self.transition_probs[per1, per2] += 1

        # calculate the probabilities
        self.transition_probs = self.transition_probs / self.transition_probs.sum(axis=1, keepdims=True)
        self.emission_probs = self.emission_probs / self.emission_probs.sum(axis=1, keepdims=True)

        # if nan values are present, replace them with 0
        self.transition_probs[np.isnan(self.transition_probs)] = 0.
        self.emission_probs[np.isnan(self.emission_probs)] = 0.

        # get initial probabilities
        self.initial_probs = self.transition_probs[self.bos_idx, :]

    def predict(self, x: List[int]) -> List[int]:
        # this method predicts the most likely sequence of tokens given the input sequence
        # using the Viterbi algorithm

        # get empty tables for backtracking and probabilities
        table1 = np.zeros((self.vocab_size, self.vocab_size))
        table2 = np.zeros((self.vocab_size, self.vocab_size), dtype=int)

        # initialize the first row of the table
        table1[:, 0] = self.initial_probs * self.emission_probs[:, bos_idx]
        table2[:, 0] = 0

        # fill the rest of the table
        # for i in range(self.vocab_size):
        for j in trange(1, len(x)):
            #cell_vector = table1[:, j - 1] * self.transition_probs[:, i] * self.emission_probs[i, x[j]]
            cell_vector = table1[:, j - 1] * self.transition_probs * self.emission_probs[:, x[j]]
            table1[:, j] = np.max(cell_vector, axis=-1)
            table2[:, j] = np.argmax(cell_vector, axis=-1)

        # get the most likely sequence of tokens
        most_likely_sequence = [np.argmax(table1[:, -1])]
        for i in range(len(x) - 1, 0, -1):
            most_likely_sequence.append(table2[most_likely_sequence[-1], i])

        return most_likely_sequence[::-1]

    def evaluate(self, dataloader, **kwargs):
        pass

    def save(self, save_dir=f'.hmm_{datetime.timestamp(datetime.now())}'):
        # create save_dir
        os.makedirs(save_dir, exist_ok=True)

        # save initial, transition, and emission probabilities
        np.save(os.path.join(save_dir, 'initial_probs.npy'), self.initial_probs)
        np.save(os.path.join(save_dir, 'transition_probs.npy'), self.transition_probs)
        np.save(os.path.join(save_dir, 'emission_probs.npy'), self.emission_probs)

        print(f'[INFO]: Model saved to {save_dir}')

    def load(self, load_dir):
        # if load_dir does not exist, use the last created hmm model
        if not os.path.exists(load_dir):
            load_dir = sorted(glob.glob('.hmm_*'))[-1]
            print(f'[INFO]: Load dir does not exist, using {load_dir}')

        # load initial, transition, and emission probabilities
        self.initial_probs = np.load(os.path.join(load_dir, 'initial_probs.npy'))
        self.transition_probs = np.load(os.path.join(load_dir, 'transition_probs.npy'))
        self.emission_probs = np.load(os.path.join(load_dir, 'emission_probs.npy'))

        print(f'[INFO]: Model loaded from {load_dir}')


if __name__ == '__main__':
    # initialize tokenizer
    tokenizer = HiddenMarkovTokenizer(vocab_path='vocab_files/dialogsum_vocab.txt',
                                      special_tokens={'bos': '<BOS>', 'eos': '<EOS>',
                                                      'pad': '<PAD>', 'unk': '<UNK>'})

    # initialize data preparer for seq2seq training
    data_prep = DataPreparer(tokenizer, 'knkarthick/dialogsum', 'seq2seq_generation',
                             special_tokens=tokenizer.special_tokens['pad'])

    # get train and test dataloaders
    train_dataloader = data_prep('train', 'dialogue', batch_size=1, shuffle=True)
    test_dataloader = data_prep('validation', 'dialogue', batch_size=1, shuffle=True)

    # get bos index
    bos_idx = tokenizer.vocab2idx[tokenizer.special_tokens['bos']]

    # initialize model
    model = HiddenMarkovModel(vocab_size=len(tokenizer.vocab2idx), bos_idx=bos_idx)

    # fit model
    # model.fit(train_dataloader)

    # save model
    # model.save()

    # get evaluation results
    # model.eval(test_dataloader)

    # predict
    sent = 'Hi, how are you today?'
    print(sent)
    x = tokenizer.encode(sent)[0]
    y = model.predict(x)

    y_dec = tokenizer.decode([y])
    print(y_dec)

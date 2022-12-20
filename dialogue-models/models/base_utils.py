import re
from typing import List, Tuple, Union, Any

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


class DataPreparer:
    def __init__(self, tokenizer: BaseTokenizer, dataset_name: str, task_name: str, **kwargs):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.task_name = task_name

        if self.task_name not in ['seq2seq_generation', 'next_word_prediction', 'gpt2_lm_modeling']:
            raise ValueError(f'Invalid task_name: {self.task_name}. Must be one of [seq2seq_generation, next_word_prediction]')

        self.N = kwargs['N'] if 'N' in kwargs else 3
        self.is_lowercase = kwargs['is_lowercase'] if 'is_lowercase' in kwargs else False

        if 'regex_split' in kwargs:
            self.regex_split = kwargs['regex_split']
        elif self.is_lowercase:
            self.regex_split = '[\s]*#person\d#: '
        else:
            self.regex_split = '[\s]*#Person\d#: '

    def get_dialogs(self, split=None, dialog_column: str = 'dialogue') -> List[Tuple[str, str]]:
        dataset = load_dataset(self.dataset_name, split=split)
        dialogs = dataset[dialog_column]
        if self.is_lowercase:
            dialogs = [dialog.lower() for dialog in dialogs]
        return dialogs

    def get_sentence_pairs(self, dialogs: List[str]) -> List[Tuple[str, str]]:
        # get the sentence pairs for training
        # each sentence pair is a tuple of (input, target)
        sentence_pairs = []
        for dialog in dialogs:
            # split dialog into sentences
            splitted_dialogs = re.split(self.regex_split, dialog)[1:]
            for sent1, sent2 in zip(splitted_dialogs[:-1], splitted_dialogs[1:]):
                sentence_pairs.append((sent1, sent2))

        return sentence_pairs

    def get_ngram_pairs(self, sent_pairs: List[str]) -> List[Tuple[List[str], List[str]]]:
        # get the ngram pairs for next word prediction training
        # for each ngram pair, the input is the first N words and the target is the last word
        ngram_pairs = []
        for sent1, sent2 in sent_pairs:
            # merge sentences
            sent = f'{self.tokenizer.special_tokens["bos"]} {sent1} {self.tokenizer.special_tokens["pad"]} {sent2} {self.tokenizer.special_tokens["eos"]}'
            words = sent.split()
            for i in range(len(words) - self.N):
                ngram_pairs.append((' '.join(words[i:i + self.N]), words[i + self.N]))

        return ngram_pairs

    def tokenize(self, inputs: List[Tuple[List[str], List[str]]]) -> List[Tuple[List[int], List[int]]]:
        if self.tokenizer is None:
            raise ValueError('Tokenizer is not initialized')

        if self.task_name == 'seq2seq_generation':
            sentences = [sent for sent, _ in inputs]
            sentences.append(inputs[-1][1])  # the last answer sentence
            max_length = max([len(sent) for sent in sentences]) + 2  # +2 for <bos> and <eos>

            x, y = zip(*inputs)
            x = self.tokenizer.encode(x, max_length=max_length)
            y = self.tokenizer.encode(y, max_length=max_length)

            return list(zip(x, y))

        elif self.task_name == 'next_word_prediction':
            max_length = 1

            x, y = zip(*inputs)
            x = self.tokenizer.encode(x, max_length=self.N)
            y = self.tokenizer.encode(y, max_length=1)

            return list(zip(x, y))

        elif self.task_name == 'gpt2_lm_modeling':
            sentences = []
            for sent1, sent2 in inputs:
                gpt_sent = f'{self.tokenizer.special_tokens["bos_token"]}{sent1}{self.tokenizer.special_tokens["pad_token"]}{sent2}{self.tokenizer.special_tokens["eos_token"]}'
                sentences.append(gpt_sent)

            return self.tokenizer.encode(sentences)

    def __call__(self, split=None, dialog_column: str = 'dialogue',
                 batch_size: int = 32, shuffle: bool = True):

        # get tokenized_inputs
        dialogs = self.get_dialogs(split=split, dialog_column=dialog_column)
        pairs = self.get_sentence_pairs(dialogs)

        # if task_name is next_word_prediction, get ngram pairs
        if self.task_name == 'next_word_prediction':
            pairs = self.get_ngram_pairs(pairs)

        tokenized_inputs = self.tokenize(pairs)

        if self.task_name == 'seq2seq_generation' or self.task_name == 'next_word_prediction':
            # get inputs and labels
            inputs, labels = zip(*tokenized_inputs)

            # convert lists to tensors
            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)

            # create dataset
            dataset = TensorDataset(inputs, labels)

            # create a torch dataloader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        elif self.task_name == 'gpt2_lm_modeling':
            # create a custom dataset for dict input
            gpt2_dataset = GPT2Dataset(tokenized_inputs)

            # create dataset from tokenized_inputs
            dataloader = DataLoader(gpt2_dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader


class GPT2Dataset(Dataset):
    def __init__(self, tokenized_inputs):
        super(GPT2Dataset, self).__init__()
        self.tokenized_inputs = tokenized_inputs

    def __getitem__(self, item):
        return {k: v[item] for k, v in self.tokenized_inputs.items()}

    def __len__(self):
        return len(self.tokenized_inputs['input_ids'])

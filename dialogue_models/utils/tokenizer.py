import pickle
import torch
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer


class CustomTokenizer:
    def __init__(self, tokenize_fn=None, detokenize_fn=None, bos_token=None,
                 eos_token=None, pad_token=None, unk_token=None, ):
        self.w2i = {}
        self.i2w = {}

        self._tokenizer = tokenize_fn if tokenize_fn is not None else lambda x: x.split()
        self._detokenizer = detokenize_fn if detokenize_fn is not None else lambda x: ' '.join(x)

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

    def tokenize(self, x, y=None, **kwargs):
        tokenized_x = self._tokenizer(x)
        if y is not None:
            tokenized_y = self._tokenizer(y)
            tokenized_x = tokenized_x + [self.pad_token] + tokenized_y

        if self.bos_token is not None:
            tokenized_x = [self.bos_token] + tokenized_x

        if self.eos_token is not None:
            tokenized_x = tokenized_x + [self.eos_token]

        tokenized = self._pad(tokenized_x, kwargs.get('max_len', -1))
        tokenized = [self.w2i.get(token, self.w2i[self.unk_token]) for token in tokenized]
        return torch.tensor(tokenized, device=kwargs.get('device', 'cpu'))

    def detokenize(self, x, skip_special_tokens=False):
        if isinstance(x, torch.Tensor):
            x = x.tolist()
            if skip_special_tokens:
                x = [tok for tok in x if tok not in [self.w2i[self.bos_token], self.w2i[self.eos_token],
                                                        self.w2i[self.pad_token], self.w2i[self.unk_token]]]
        return self._detokenizer(map(self.i2w.get, x))

    def __len__(self):
        return len(self.w2i)

    def _add_token(self, token):
        if token not in self.w2i:
            self.w2i[token] = len(self.w2i)
            self.i2w[len(self.i2w)] = token

    def _pad(self, x, max_len):
        if max_len == -1:
            return x
        if len(x) > max_len:
            return x[:max_len]
        return x + [self.pad_token] * (max_len - len(x))

    def fit(self, texts):
        # add special tokens (bos, eos, pad, and unk)
        if self.bos_token is not None:
            self._add_token(self.bos_token)
        if self.eos_token is not None:
            self._add_token(self.eos_token)
        if self.pad_token is not None:
            self._add_token(self.pad_token)
        if self.unk_token is not None:
            self._add_token(self.unk_token)

        # fit on the texts given
        for i, text in enumerate(texts):
            for token in self._tokenizer(text):
                self._add_token(token)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pretrained(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print('Tokenizer not found at {}'.format(path))
            return None


if __name__ == '__main__':
    tokenize_fn = TreebankWordTokenizer().tokenize
    detokenize_fn = TreebankWordDetokenizer().detokenize

    tokenizer = CustomTokenizer(tokenize_fn=tokenize_fn, detokenize_fn=detokenize_fn)
    tokenizer.fit(texts=['hello world', "what's up", 'how are you doing?'])

    print(tokenizer.w2i)
    print(tokenizer.tokenize('hello world'))
    print(tokenizer.detokenize([0, 1, 2, 7, 4, 5]))

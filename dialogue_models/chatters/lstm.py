from typing import List

import torch
from base_utils import BaseChatter, beam_search
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext.data import get_tokenizer


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
        print('INPUT LENGTH: ', len(text))
        return torch.tensor(text)

    def detokenize(self, tokens) -> str:
        tokens = self.vocab.lookup_tokens(tokens.tolist())
        tokens = [token for token in tokens if token not in self.special_tokens.values()]

        return self.detokenizer.detokenize(tokens)

    def chat(self, text: str, states=None) -> str:
        tokenized_text = self.tokenize(text)
        tokens, states = beam_search(self.model, tokenized_text, num_beams=3, max_new_token=30,
                                     eos_token_id=self.vocab[self.special_tokens['eos_token']],
                                     states=states)
        return self.detokenize(tokens), states


if __name__ == '__main__':
    from time import time
    lstm_chatter = LSTMChatter(model_path='../saved trainers/lstm-trainers/lstm_model12.pt',
                               tokenizer_path='../saved trainers/lstm-trainers/lstm-tokenizer.pth')

    states = None
    while True:
        sent = input("You: ")
        start = time()
        ans, states = lstm_chatter.chat(sent, states=states)
        print("Time: ", time() - start)
        print(f'Lissy: {ans}')

import re
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.tokenize.treebank import TreebankWordDetokenizer
from base_utils import BaseModel, BaseChatter

class HiddenMarkovModel(BaseModel):
    def __init__(self, tokenizer_name, max_len, special_tokens, **kwargs):
        super().__init__()
        self.max_len = max_len
        self.special_tokens = special_tokens
        self.tokenizer = get_tokenizer(tokenizer_name)

        self.model = None
        self.vocab = None

    def _create_tokenized_pairs(self, x):
        sentences = re.split(r'[\s]*#Person\d#: ', x['dialogue'])[1:]

        sentences1, sentences2 = [], []
        for sent1, sent2 in zip(sentences[:-1], sentences[1:]):
            sent1 = ' '.join([self.special_tokens['bos_token'], sent1, self.special_tokens['eos_token']])
            sent2 = ' '.join([sent2, self.special_tokens['eos_token']])

            sent1 = self.tokenizer(sent1)[:self.max_len]
            sent2 = self.tokenizer(sent2)[:self.max_len]

            sent1 += [special_tokens['pad_token']] * (self.max_len - len(sent1))
            sent2 += [special_tokens['pad_token']] * (self.max_len - len(sent2))

            if sent1 not in sentences1 and sent2 not in sentences2:
                sentences1.append(sent1)
                sentences2.append(sent2)

        return {'sent1': sentences1, 'sent2': sentences2}

    def get_vocab(self, dataset):
        tokenized_dataset = dataset.map(self._create_tokenized_pairs)

        flatten_sent1 = [sent for sents in tokenized_dataset['sent1'] for sent in sents]
        flatten_sent2 = [sents[-1] for sents in tokenized_dataset['sent2']]
        flatten_sents = flatten_sent1 + flatten_sent2

        self.vocab = build_vocab_from_iterator(flatten_sents, min_freq=5, specials=list(self.special_tokens.values()))
        self.vocab.set_default_index(self.vocab[self.special_tokens['unk_token']])

    def _get_pairs(self, dataset):
        dataset = dataset.map(self._create_tokenized_pairs)
        questions = [self.vocab(sent) for sents in dataset['sent1'] for sent in sents]
        answers = [self.vocab(sent) for sents in dataset['sent2'] for sent in sents]

        return [[(word_q, word_a) for word_q, word_a in zip(question, answer)] for question, answer in
                zip(questions, answers)]

    def fit(self, dataloader, **kwargs):
        val_dataloader = kwargs['val_dataloader'] if 'val_dataloader' in kwargs else None
        train_pairs = self._get_pairs(dataloader)
        val_pairs = self._get_pairs(val_dataloader) if val_dataloader is not None else None
        self.model = HiddenMarkovModelTagger.train(train_pairs, test_sequences=val_pairs, max_iterations=1000000000)

    def predict(self, x: str):
        x = ' '.join([self.special_tokens['bos_token'], x, self.special_tokens['eos_token'], self.special_tokens['pad_token']])
        x = self.vocab(self.tokenizer(x))
        return self.model.best_path(x)


class HMMChatter(BaseChatter):
    def __init__(self, tokenizer_name, max_len, special_tokens):
        super().__init__()
        train_dataset = load_dataset('knkarthick/dialogsum', split='train')
        val_dataset = load_dataset('knkarthick/dialogsum', split='validation')

        self.model = HiddenMarkovModel(tokenizer_name=tokenizer_name, max_len=max_len, special_tokens=special_tokens)
        self.model.get_vocab(train_dataset)

        self.model.fit(train_dataset, val_dataloader=val_dataset)
        self.detokenizer = TreebankWordDetokenizer()

    def chat(self, text: str) -> str:
        pred = self.model.predict(text)
        # tokens = self.model.vocab.lookup_tokens(pred)
        filtered_tokens = list(filter(lambda x: x not in self.model.special_tokens.values(), tokens))
        return self.detokenizer.detokenize(filtered_tokens)


if __name__ == '__main__':
    max_length = 50
    special_tokens = {'bos_token': '|BOS|', 'eos_token': '|EOS|', 'pad_token': '|PAD|', 'unk_token': '|UNK|'}
    lst = time.time()
    hmm_chatter = HMMChatter(tokenizer_name='spacy', max_len=max_length, special_tokens=special_tokens)
    lst = time.time() - lst
    print(f'Loaded in {lst} seconds')
    while True:
        text = input('You: ')
        ist = time.time()
        print('Hans Markow:', hmm_chatter.chat(text))
        ist = time.time() - ist
        print(f'Inference time: {ist:.2f} seconds')

from typing import List, Union

import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM

from base_utils import BaseTokenizer, BaseModel, BaseChatter, DataPreparer


class GPTTokenizer(BaseTokenizer):
    def __init__(self, model_name: str = 'gpt2', **kwargs):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.special_tokens = kwargs['special_tokens'] if 'special_tokens' in kwargs.keys() else {}
        self.tokenizer.add_special_tokens(self.special_tokens)

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenizer.tokenize(text) for text in texts]

    def encode(self, texts, max_length=None) -> List[List[int]]:
        return self.tokenizer(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

    def decode(self, tokens):
        if isinstance(tokens[0], list):
            return self.tokenizer.batch_decode(tokens)
        else:
            return self.tokenizer.decode(tokens)


class GPTModel(nn.Module, BaseModel):
    def __init__(self, **kwargs):
        super(GPTModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer_length = kwargs['tokenizer_length'] if 'tokenizer_length' in kwargs.keys() else self.model.lm_head.out_features

        # extend output layer length
        self.model.resize_token_embeddings(self.tokenizer_length)

    def fit(self, dataloader, **kwargs):
        epochs = kwargs['epochs'] if 'epochs' in kwargs.keys() else 1
        optimizer = kwargs['optimizer'] if 'optimizer' in kwargs.keys() else torch.optim.Adam(self.parameters())
        device = kwargs['device'] if 'device' in kwargs.keys() else 'cpu'

        # move model to device
        # self.to(device)

        for epoch in range(epochs):
            for idx, batch in enumerate(dataloader, start=1):
                optimizer.zero_grad()
                loss = self.model(**batch, labels=batch['input_ids']).loss
                loss.backward()
                optimizer.step()

                print(f'Epoch: {epoch + 1}/{epochs}, Batch: {idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

                if idx % 5 == 0:
                    break

    def predict(self, x):
        return self.model.generate(**x, max_length=100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, )

    def evaluate(self, dataloader, **kwargs):
        device = kwargs['device'] if 'device' in kwargs.keys() else 'cpu'

        # move model to device
        self.to(device)

        for idx, batch in enumerate(dataloader, start=1):
            loss = self.model(**batch, labels=batch['input_ids']).loss

            print(f'Batch: {idx}/{len(dataloader)} - Loss: {loss.item():.4f}')

            if idx == 3:
                break

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f'[INFO]: Model saved to {path}')

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f'[INFO]: Model loaded from {path}')


class GPTChatter(BaseChatter):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>',
                                           'unk_token': '<UNK>', 'pad_token': '<PAD>'})

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

    def chat(self, text: str):
        text = self.tokenizer.bos_token + text + self.tokenizer.pad_token
        tok_text = self.tokenizer(text, max_length=50, return_tensors='pt')
        gpt_output = self.model.generate(tok_text['input_ids'], max_length=50, num_beams=5, temperature=0.7,
                                         no_repeat_ngram_size=2, early_stopping=True,
                                         do_sample=True, num_return_sequences=1)

        return self.tokenizer.decode(gpt_output[0], skip_special_tokens=True)


if __name__ == '__main__':
    galip = GPTChatter(model_path='../saved models/gpt_model')

    while True:
        sent = input("You: ")
        ans = galip.chat(sent)
        print(ans)

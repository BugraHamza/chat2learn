from typing import List, Union

import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from base_utils import BaseTokenizer, BaseModel, DataPreparer


class GPTTokenizer(BaseTokenizer):
    def __init__(self, model_name: str = 'gpt2', **kwargs):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.special_tokens = kwargs['special_tokens'] if 'special_tokens' in kwargs.keys() else {}
        self.tokenizer.add_special_tokens(self.special_tokens)

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenizer.tokenize(text) for text in texts]

    def encode(self, texts: Union[str | List[str]], max_length: Union[None | int] = None) -> List[List[int]]:
        return self.tokenizer(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

    def decode(self, tokens: Union[List[int] | List[List[int]]]) -> Union[str | List[str]]:
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


if __name__ == '__main__':
    # load tokenizer
    tokenizer = GPTTokenizer(model_name='gpt2', special_tokens={'bos_token': '<bos>',
                                                                'eos_token': '<eos>',
                                                                'pad_token': '<pad>'})

    # load model
    model = GPTModel(tokenizer_length=len(tokenizer.tokenizer))

    # initialize data preparer for seq2seq training
    data_prep = DataPreparer(tokenizer, 'knkarthick/dialogsum', 'gpt2_lm_modeling')

    # get train and test dataloaders
    train_dataloader = data_prep('train', 'dialogue', batch_size=4, shuffle=True)
    test_dataloader = data_prep('validation', 'dialogue', batch_size=4, shuffle=True)

    # fit model
    model.fit(train_dataloader, epochs=1, optimizer=torch.optim.Adam(model.model.parameters()))

    # evaluate model
    model.evaluate(test_dataloader)

    # predict
    sent = 'What will you do today?'
    enc_sent = tokenizer.encode(sent)
    pred = model.predict(enc_sent)

    print(f'Input: {sent}')
    print(f'Output: {tokenizer.tokenizer.decode(pred[0])}')

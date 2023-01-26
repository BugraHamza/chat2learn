from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm, trange
from .util_func import get_metric


class BaseChatter:
    def __init__(self):
        pass

    def tokenize(self, text: str) -> List[str]:
        pass

    def detokenize(self, tokens: List[str]) -> str:
        pass

    def chat(self, text: str) -> str:
        pass


class BaseTrainer:
    def __init__(self, model, criterion, metrics=[], tokenizer=None, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.metrics = {metric_name: get_metric(metric_name) for metric_name in metrics}
        self.tokenizer = tokenizer if metrics != [] else None
        self.device = device

    def model_forward(self, x):
        raise NotImplementedError('model_forward is not implemented and should be implemented in the child class')

    def fit(self, dataloader, epochs, **kwargs):
        optimizer_fn = kwargs['optimizer']
        learning_rate = kwargs.get('learning_rate', 0.001)
        optimizer = optimizer_fn(self.model.parameters(), lr=learning_rate)

        history = {'loss': []}
        for metric_name in self.metrics.keys():
            history[metric_name] = []

        self.model.train()
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f'Epoch #{epoch+1}/{epochs}', leave=False)
            for batch in pbar:
                if len(batch) == 1:
                    x = batch[0][:, :-1]
                    y = batch[0][:, 1:]
                elif len(batch) == 2:
                    x, y = batch
                else:
                    raise ValueError('batch should have 1 or 2 elements')

                optimizer.zero_grad()
                y_pred = self.model_forward(x)
                loss = self.criterion(y_pred.moveaxis(1, -1), y)
                loss.backward()
                optimizer.step()

                # tokenize y and y_pred to get metrics
                if self.tokenizer is not None:
                    y_pred = y_pred.argmax(dim=-1)
                    y = [self.tokenizer.detokenize(sent, skip_special_tokens=True) for sent in y]
                    y_pred = [self.tokenizer.detokenize(sent, skip_special_tokens=True) for sent in y_pred]

                history['loss'].append(loss.item())

                for metric_name, metric in self.metrics.items():
                    history[metric_name].append(metric(pred=y_pred, target=y))

                # update progress bar postfix using the mean
                postfix_dict = {'loss': np.mean(history['loss']), **{metric_name: np.mean(history[metric_name]) for metric_name in self.metrics.keys()}}
                pbar.set_postfix(postfix_dict)

        if kwargs.get('reduction') == 'mean':
            for metric_name, metric_result in self.metrics.items():
                history[metric_name] = sum(metric_result) / len(metric_result)
            history['loss'] = sum(history['loss']) / len(history['loss'])
        elif kwargs.get('reduction') == 'sum':
            for metric_name, metric_result in self.metrics.items():
                history[metric_name] = sum(metric_result)
            history['loss'] = sum(history['loss'])

        return history

    def evaluate(self, dataloader, **kwargs):

        history = {'loss': []}
        for metric_name in self.metrics.keys():
            history[metric_name] = []

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Evaluating', leave=False)
            for batch in pbar:
                if len(batch) == 1:
                    x = batch[0][:, :-1]
                    y = batch[0][:, 1:]
                elif len(batch) == 2:
                    x, y = batch
                else:
                    raise ValueError('batch should have 1 or 2 elements')

                y_pred = self.model_forward(x)
                loss = self.criterion(y_pred.moveaxis(1, -1), y)
                history['loss'].append(loss.item())

                # tokenize y and y_pred to get metrics
                if self.tokenizer is not None:
                    y_pred = y_pred.argmax(dim=-1)
                    y = self.tokenizer.detokenize(y)
                    y_pred = self.tokenizer.detokenize(y_pred)

                for metric_name, metric in self.metrics.items():
                    history[metric_name].append(metric(predictions=y_pred, references=y))

                # update progress bar postfix using the mean
                postfix_dict = {'loss': np.mean(history['loss']), **{metric_name: np.mean(history[metric_name]) for metric_name in self.metrics.keys()}}
                pbar.set_postfix(postfix_dict)

        if kwargs.get('reduction') == 'mean':
            for metric_name in self.metrics.keys():
                history[metric_name] = sum(history[metric_name]) / len(history[metric_name])
            history['loss'] = sum(history['loss']) / len(history['loss'])
        elif kwargs.get('reduction') == 'sum':
            for metric_name in self.metrics.keys():
                history[metric_name] = sum(history[metric_name])
            history['loss'] = sum(history['loss'])

        return history

    def save(self, path):
        raise NotImplementedError('save is not implemented and should be implemented in the child class')

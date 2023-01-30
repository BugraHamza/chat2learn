from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm, trange
from .metrics import get_metric


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

    def _get_optimizer(self, optimizer_fn, learning_rate):
        return optimizer_fn(self.model.parameters(), lr=learning_rate)

    def _get_history(self):
        history = {'loss': []}
        for metric_name in self.metrics.keys():
            history[metric_name] = []
        return history

    def _reduce_history(self, history, reduction_method: str = 'mean'):
        reduced_history = {}
        if reduction_method == 'mean':
            for metric_name in history.keys():
                reduced_history[metric_name] = np.mean(history[metric_name])
        elif reduction_method == 'sum':
            for metric_name in history.keys():
                reduced_history[metric_name] = np.sum(history[metric_name])
        elif reduction_method == 'last':
            for metric_name in history.keys():
                reduced_history[metric_name] = history[metric_name][-1]
        else:
            reduced_history = history
        return reduced_history

    def _trainer_loop(self, dataloader, is_training: bool = True, keep_history: bool = True, **kwargs):
        optimizer = self._get_optimizer(**kwargs) if is_training else None
        history = self._get_history() if keep_history else None

        self.model.train() if is_training else self.model.eval()
        with torch.set_grad_enabled(is_training):
            pbar = tqdm(dataloader, desc='Training' if is_training else 'Evaluating', leave=False)
            for batch in pbar:
                if len(batch) == 1:
                    x = batch[0][:, :-1]
                    y = batch[0][:, 1:]
                elif len(batch) == 2:
                    x, y = batch
                else:
                    raise ValueError('batch should have 1 or 2 elements')

                if is_training:
                    optimizer.zero_grad()
                y_pred = self.model_forward(x)
                loss = self.criterion(y_pred.moveaxis(1, -1), y)
                if is_training:
                    loss.backward()
                    optimizer.step()

                # tokenize y and y_pred to get metrics
                if self.tokenizer is not None:
                    y_pred = y_pred.argmax(dim=-1)
                    y = [self.tokenizer.detokenize(sent, skip_special_tokens=True) for sent in y]
                    y_pred = [self.tokenizer.detokenize(sent, skip_special_tokens=True) for sent in y_pred]

                if keep_history:
                    history['loss'].append(loss.item())

                    for metric_name, metric in self.metrics.items():
                        history[metric_name].append(metric(pred=y_pred, target=y))

                    # update progress bar postfix using the mean
                    postfix_dict = self._reduce_history(history, kwargs.get('reduction_method', 'mean')) # {'loss': np.mean(history['loss']), **{metric_name: np.mean(history[metric_name]) for metric_name in self.metrics.keys()}}
                    pbar.set_postfix(postfix_dict)

        if keep_history:
            return self._reduce_history(history, reduction_method=kwargs.get('reduction_method', 'mean'))

    def train(self, dataloader, **kwargs):
        return self._trainer_loop(dataloader, is_training=True, **kwargs)

    def evaluate(self, dataloader, **kwargs):
        return self._trainer_loop(dataloader, is_training=False, **kwargs)

    def save(self, path):
        raise NotImplementedError('save is not implemented and should be implemented in the child class')

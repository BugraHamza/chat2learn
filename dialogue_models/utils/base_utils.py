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
    def __init__(self, model, criterion, metric_name=None, tokenizer=None, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.metric_name = metric_name
        self.metric = get_metric(metric_name) if metric_name is not None else None
        self.tokenizer = tokenizer if metric_name is not None else None
        self.device = device

    def model_forward(self, x):
        raise NotImplementedError('model_forward is not implemented and should be implemented in the child class')

    def _get_optimizer(self, optimizer_fn, learning_rate):
        return optimizer_fn(self.model.parameters(), lr=learning_rate)

    def _trainer_loop(self, dataloader, is_training: bool = True, **kwargs):
        optimizer = self._get_optimizer(kwargs['optimizer'], kwargs['learning_rate']) if is_training else None

        losses = []
        metrics = []

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

                # add loss to losses
                losses.append(loss.item())

                # tokenize y and y_pred to add metric to metrics
                if self.tokenizer is not None:
                    y_pred = y_pred.argmax(dim=-1)
                    y = [self.tokenizer.detokenize(sent, skip_special_tokens=True) for sent in y]
                    y_pred = [self.tokenizer.detokenize(sent, skip_special_tokens=True) for sent in y_pred]

                    metrics.append(self.metric(pred=y_pred, target=y))

                    # update progress bar postfix using the mean
                    postfix_dict = {'loss': np.mean(losses), self.metric_name: np.mean(metrics)}
                    pbar.set_postfix(postfix_dict)

        return np.mean(losses), np.mean(metrics)

    def train(self, dataloader, **kwargs):
        return self._trainer_loop(dataloader, is_training=True, **kwargs)

    def evaluate(self, dataloader, **kwargs):
        return self._trainer_loop(dataloader, is_training=False, **kwargs)

    def save(self, path):
        raise NotImplementedError('save is not implemented and should be implemented in the child class')

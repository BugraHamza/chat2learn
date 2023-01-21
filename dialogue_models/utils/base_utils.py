from typing import List

import torch


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
    def __init__(self, model):
        self.model = model

    def model_forward(self, x):
        raise NotImplementedError('model_forward is not implemented and should be implemented in the child class')

    def fit(self, dataloader, epochs, **kwargs):
        optimizer = kwargs['optimizer']
        criterion = kwargs['criterion']
        metrics = kwargs['metrics']

        history = {'loss': []}
        for metric_name in metrics.keys():
            history[metric_name] = []

        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                optimizer.zero_grad()
                y_pred = self.model_forward(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                history['loss'].append(loss.item())
                for metric_name, metric in metrics.items():
                    history[metric_name].append(metric(y_pred, y))

        if kwargs.get('reduction') == 'mean':
            for metric_name in metrics.keys():
                history[metric_name] = sum(history[metric_name]) / len(history[metric_name])
            history['loss'] = sum(history['loss']) / len(history['loss'])
        elif kwargs.get('reduction') == 'sum':
            for metric_name in metrics.keys():
                history[metric_name] = sum(history[metric_name])
            history['loss'] = sum(history['loss'])

        return history

    def evaluate(self, dataloader, **kwargs):
        criterion = kwargs['criterion']
        metrics = kwargs['metrics']

        history = {'loss': []}
        for metric_name in metrics.keys():
            history[metric_name] = []

        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                y_pred = self.model_forward(x)
                loss = criterion(y_pred, y)
                history['loss'].append(loss.item())
                for metric_name, metric in metrics.items():
                    history[metric_name].append(metric(y_pred, y))

        if kwargs.get('reduction') == 'mean':
            for metric_name in metrics.keys():
                history[metric_name] = sum(history[metric_name]) / len(history[metric_name])
            history['loss'] = sum(history['loss']) / len(history['loss'])
        elif kwargs.get('reduction') == 'sum':
            for metric_name in metrics.keys():
                history[metric_name] = sum(history[metric_name])
            history['loss'] = sum(history['loss'])

        return history

    def save(self, path):
        raise NotImplementedError('save is not implemented and should be implemented in the child class')


def beam_search(model, tokenized_sent, num_beams, max_new_token, eos_token_id, **kwargs):
    # generate function for beam search decoding
    # model: model to generate
    # tokenized_sent: tokenized input sentence
    # num_beams: number of beams
    # max_new_token: maximum number of new tokens to generate
    # eos_token_id: end of sentence token_id for early stopping
    # **kwargs: additional arguments for model forward pass

    # get first beams
    out, states = model(tokenized_sent, kwargs.get('states'))
    top_preds = out[-1, :].topk(num_beams, dim=-1)

    # add predictions to beams
    beams = [(pred.unsqueeze(0), states, score.item()) for pred, score in zip(top_preds.indices, top_preds.values)]

    for i in range(max_new_token - 1):
        new_beams = []
        for beam in beams:
            if beam[0][-1] == eos_token_id:
                new_beams.append(beam)
                continue

            out, states = model(beam[0][-1:], beam[1])
            top_preds = out[-1, :].topk(num_beams, dim=-1)

            for pred, score in zip(top_preds.indices, top_preds.values):
                new_beams.append((torch.cat((beam[0], pred.unsqueeze(0)), dim=0), states, beam[2] + score.item()))

        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:num_beams]

        if beams[0][0][-1] == eos_token_id:
            break

    # return the best beam and its states
    return beams[0][0], beams[0][1]

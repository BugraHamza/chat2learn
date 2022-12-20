import random
import re

import numpy as np
import optuna
from datasets import load_dataset
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def create_sentence_pairs(x, tokenizer):
    sentences = re.split(r'[\s]*#Person\d#: ', x['dialogue'])[1:]
    return {
        'sentence_pairs': [' '.join([tokenizer.bos_token, sent1, tokenizer.pad_token, sent2, tokenizer.eos_token]) for
                           sent1, sent2 in zip(sentences[:-1], sentences[1:])]}

def prepare_model_and_tokenizer(path, device):
    # get gpt2 tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.add_special_tokens(
        {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'unk_token': '<UNK>', 'pad_token': '<PAD>'})

    # get gpt2 model and resize the output shape based on the tokens added
    model = AutoModelForCausalLM.from_pretrained(path)
    model.resize_token_embeddings(len(tokenizer))

    return model.to(device), tokenizer


class GPTDataset(Dataset):
    def __init__(self, sentence_pairs, tokenizer, max_len, device):
        self.sentence_pairs = [pair for pair_ls in sentence_pairs for pair in pair_ls]
        random.shuffle(self.sentence_pairs)
        self.sentence_pairs = self.sentence_pairs

        self.tokenized_pairs = tokenizer(self.sentence_pairs, max_length=max_len, padding='max_length', truncation=True,
                                         return_tensors='pt')
        self.tokenized_inputs = self.tokenized_pairs['input_ids']
        self.tokenized_mask = self.tokenized_pairs['attention_mask']

        self.device = device

    def __getitem__(self, item):
        return {'input_ids': self.tokenized_inputs[item].to(self.device),
                'attention_mask': self.tokenized_mask[item].to(self.device)}

    def __len__(self):
        return len(self.sentence_pairs)


def train_step(model, optimizer, train_loader):
    model.train()

    train_losses = []
    pbar = tqdm(train_loader)
    for idx, x in enumerate(pbar):
        optimizer.zero_grad()
        loss = model(**x, labels=x['input_ids']).loss
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        pbar.set_description(f'Loss: {np.mean(train_losses):.3f}')

    return np.mean(train_losses)

def eval_step(model, val_loader):
    model.eval()

    val_losses = []
    pbar = tqdm(val_loader)
    for x in pbar:
        loss = model(**x, labels=x['input_ids']).loss
        val_losses.append(loss.item())
        pbar.set_description(f'Validation Loss: {np.mean(val_losses):.3f}')

    return np.mean(val_losses)

def train(max_len, epochs, bs, lr, device):
    model_path = 'gpt2'
    model, tokenizer = prepare_model_and_tokenizer(model_path, device)

    train_dataset = load_dataset('knkarthick/dialogsum', split='train')
    val_dataset = load_dataset('knkarthick/dialogsum', split='validation')

    train_dataset = train_dataset.map(lambda x: create_sentence_pairs(x, tokenizer))
    val_dataset = val_dataset.map(lambda x: create_sentence_pairs(x, tokenizer))

    gpt_train = GPTDataset(train_dataset['sentence_pairs'], tokenizer, max_len, device)
    gpt_valid = GPTDataset(val_dataset['sentence_pairs'], tokenizer, max_len, device)

    train_loader = DataLoader(gpt_train, batch_size=bs, shuffle=True)
    val_loader = DataLoader(gpt_valid, batch_size=bs, shuffle=True)

    optimizer = Adam(model.parameters(), lr=lr)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loss = train_step(model, optimizer, train_loader)
        val_loss = eval_step(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        plt.plot(train_losses, 'b', label='Training Loss')
        plt.plot(val_losses, 'r', label='Validation Loss')

        model.save_pretrained(f'gpt_model_{epoch}')

    plt.legend()

    # return last validation loss
    return val_loss


def objective(trial):
    params = {'bs': trial.suggest_int('bs', 2, 6),
              'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True)}

    print(params)
    return train(max_len=100, epochs=1, device='cuda', **params)

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=3)

    print(study.best_params)

if __name__ == '__main__':
    main()

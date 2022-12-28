import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from base_utils import BaseChatter


class Seq2SeqModel:
    def __init__(self, bert_model, gpt_model, bert_tokenizer, gpt_tokenizer):
        self.bert_model = bert_model
        self.gpt_model = gpt_model
        self.bert_tokenizer = bert_tokenizer
        self.gpt_tokenizer = gpt_tokenizer

    def train_step(self, dataloader, optimizer):
        self.bert_model.train(), self.gpt_model.train()
        losses = []

        pbar = tqdm(dataloader)
        for i, (x, y) in enumerate(pbar):
            optimizer.zero_grad()
            bert_out = self.bert_model(**x)
            gpt_out = self.gpt_model(**y, labels=y['input_ids'],
                                     encoder_hidden_states=bert_out.last_hidden_state.to(self.gpt_model.device))
            loss = gpt_out.loss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_description(f'Batch Loss: {np.mean(losses):.5f}')

        return np.mean(losses)

    def eval_step(self, dataloader):
        self.bert_model.eval(), self.gpt_model.eval()
        losses = []

        pbar = tqdm(dataloader)
        for i, (x, y) in enumerate(pbar):
            bert_out = self.bert_model(**x)
            gpt_out = self.gpt_model(**y, labels=y['input_ids'],
                                     encoder_hidden_states=bert_out.last_hidden_state.to(self.gpt_model.device))
            loss = gpt_out.loss

            losses.append(loss.item())
            pbar.set_description(f'Batch Loss: {np.mean(losses):.5f}')

        return np.mean(losses)

    def answer(self, sent):
        with torch.no_grad():
            tokenized_sent = self.bert_tokenizer(sent, padding='max_length',
                                                 truncation=True, return_tensors='pt')
            tokenized_ans = self.gpt_tokenizer(self.gpt_tokenizer.bos_token, return_tensors='pt')

            tokenized_sent = {k: v.to(self.bert_model.device) for k, v in tokenized_sent.items()}
            tokenized_ans = {k: v.to(self.gpt_model.device) for k, v in tokenized_ans.items()}

            bert_out = self.bert_model(**tokenized_sent)
            gpt_out = self.gpt_model.generate(**tokenized_ans, max_new_tokens=30, temperature=0.8, num_beams=7,
                                              no_repeat_ngram_size=3, early_stopping=True, do_sample=True,
                                              num_return_sequences=1, top_k=50, top_p=0.95,
                                              bos_token_id=self.gpt_tokenizer.bos_token_id,
                                              pad_token_id=self.gpt_tokenizer.pad_token_id,
                                              eos_token_id=self.gpt_tokenizer.eos_token_id,
                                              encoder_hidden_states=bert_out.last_hidden_state)

            return self.gpt_tokenizer.decode(gpt_out[0])

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.bert_model.save_pretrained(os.path.join(model_dir, 'bert_encoder'))
        self.gpt_model.save_pretrained(os.path.join(model_dir, 'gpt_decoder'))


class BertGptChatter(BaseChatter):
    def __init__(self, model_path: str):
        self.model_max_length = 50

        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', model_max_length=self.model_max_length)
        self.gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2', model_max_length=self.model_max_length)
        self.gpt_tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>',
                                               'unk_token': '<UNK>', 'pad_token': '<PAD>'})

        self.bert_encoder = AutoModel.from_pretrained(os.path.join(model_path, 'bert_encoder'))
        self.gpt_decoder = AutoModelForCausalLM.from_pretrained(os.path.join(model_path, 'gpt_decoder'))

        self.gpt_decoder.resize_token_embeddings(len(self.gpt_tokenizer))
        self.gpt_decoder.config.add_cross_attention = True
        self.gpt_decoder.config.is_decoder = True

        self.seq2seq_model = Seq2SeqModel(bert_model=self.bert_encoder, gpt_model=self.gpt_decoder,
                                          bert_tokenizer=self.bert_tokenizer, gpt_tokenizer=self.gpt_tokenizer)

    def chat(self, text: str):
        return self.seq2seq_model.answer(text)


if __name__ == '__main__':
    bert_gpt_model = BertGptChatter(model_path='../saved models/bert_gpt_model')

    while True:
        sent = input("You: ")
        ans = bert_gpt_model.chat(sent)
        print('Berry Gabe: ', ans)

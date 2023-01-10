import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_utils import BaseChatter
import pandas as pd

class GPTChatter(BaseChatter):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
        self.tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>',
                                           'unk_token': '<UNK>', 'pad_token': '<PAD>'})

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

    def chat(self, text: str):
        text = self.tokenizer.bos_token + text + self.tokenizer.pad_token
        tok_text = self.tokenizer(text, return_tensors='pt')
        text_len=  len(tok_text['input_ids'][0])
        gpt_output = self.model.generate(**tok_text, max_new_tokens=50, num_beams=1, do_sample=False,)
        generated_text = gpt_output[0][text_len:]
        generated_text = self.tokenizer.decode(generated_text, skip_special_tokens=True)
        return generated_text


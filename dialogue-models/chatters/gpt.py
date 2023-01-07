import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from base_utils import BaseChatter
import pandas as pd

class GPTChatter(BaseChatter):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
        self.tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>',
                                           'unk_token': '<UNK>', 'pad_token': '<PAD>'})

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

    def chat(self, text: str, hidden_states: torch.Tensor):
        text_len = 1 + len(text)
        text = self.tokenizer.bos_token + text
        tok_text = self.tokenizer(text, return_tensors='pt')
        gpt_output = self.model.generate(**tok_text, max_new_tokens=50, num_beams=5, temperature=0.8,
                                    no_repeat_ngram_size=3, early_stopping=True, do_sample=True,
                                    num_return_sequences=1, top_k=80, top_p=0.7,
                                    bos_token_id=self.tokenizer.bos_token_id,
                                    pad_token_id=self.tokenizer.pad_token_id, 
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    #decoder_start_token_id=self.tokenizer.pad_token_id,
                                    output_hidden_states=True, return_dict_in_generate=True,
                                    encoder_hidden_states=hidden_states)

        generated_text = self.tokenizer.decode(gpt_output['sequences'][0], skip_special_tokens=True)[text_len:]
        hidden_states = gpt_output['hidden_states']

        return generated_text, hidden_states

sentence_df = pd.read_excel('~/Downloads/Rouge Sentences.xlsx')
    
print(sentence_df)
if __name__ == '__main__':
    galip = GPTChatter(model_path='../trained_models/gpt_epoch_4')

    for index, sent in enumerate(sentence_df['Sentences']):   
        hidden_states = None
        print(sent)
        
        ans, hidden_states = galip.chat(sent, hidden_states=hidden_states)
        sentence_df.at[index,"GPT Sentence" ] = ans

    sentence_df.to_csv('~/Downloads/Rouge Sentences1.csv', index=False)
    print(sentence_df) 

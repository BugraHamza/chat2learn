from base_utils import BaseChatter
from transformers import AutoTokenizer, AutoModelForCausalLM


class GPTChatter(BaseChatter):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
        self.tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>',
                                           'unk_token': '<UNK>', 'pad_token': '<PAD>'})

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

    def chat(self, text: str):
        text_len = 1 + len(text)
        text = self.tokenizer.bos_token + text
        tok_text = self.tokenizer(text, return_tensors='pt')
        print('INPUT LENGTH: ', len(tok_text['input_ids'][0]))
        gpt_output = self.model.generate(**tok_text, max_new_tokens=50, num_beams=1, do_sample=False,)

        generated_text = self.tokenizer.decode(gpt_output[0], skip_special_tokens=True)[text_len:]
        return generated_text


if __name__ == '__main__':
    from time import time
    galip = GPTChatter(model_path='../saved trainers/gpt2-trainers/gpt_epoch_4')

    while True:
        sent = input('You: ')
        start = time()
        ans = galip.chat(sent)
        print('Time: ', time() - start)
        print('Galip: ', ans)

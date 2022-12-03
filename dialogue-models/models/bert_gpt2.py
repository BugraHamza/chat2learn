from transformers import AutoTokenizer, EncoderDecoderModel
from base_utils import BaseChatter


class BertGPTChatter(BaseChatter):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.encoder_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=128)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained('gpt2', model_max_length=128)
        self.decoder_tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'})

        self.model = EncoderDecoderModel.from_pretrained(self.model_path, use_auth_token=True)

    def tokenize(self, text: str):
        tok_text = self.encoder_tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')

        return {'input_ids': tok_text['input_ids'], 'attention_mask': tok_text['attention_mask']}

    def decode(self, text: str):
        return self.decoder_tokenizer.decode(text, padding='max_length', truncation=True, skip_special_tokens=True)

    def chat(self, text: str):
        text = self.tokenize(text)
        gpt_output = self.model.generate(**text, max_new_tokens=50, num_beams=7, temperature=0.8,
                                    no_repeat_ngram_size=3, early_stopping=True, do_sample=True,
                                    num_return_sequences=1, top_k=50, top_p=0.95,
                                    bos_token_id=self.encoder_tokenizer.bos_token_id,
                                    pad_token_id=self.encoder_tokenizer.pad_token_id,
                                    eos_token_id=self.encoder_tokenizer.eos_token_id,
                                    decoder_start_token_id=self.decoder_tokenizer.pad_token_id)
        print(gpt_output)
        return self.decode(gpt_output[0])


if __name__ == '__main__':
    galip = BertGPTChatter(model_path='Quimba/bert_gpt2_dialog_generation/checkpoint-1000')

    while True:
        sent = input("You: ")
        ans = galip.chat(sent)
        print('Benny Gwayne: ', ans)

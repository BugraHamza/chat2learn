import pandas as pd
from nltk import TreebankWordTokenizer
from nltk import TreebankWordDetokenizer

from dialogue_models.utils.tokenizer import CustomTokenizer


if __name__ == '__main__':
    tok_fn = TreebankWordTokenizer().tokenize
    detok_fn = TreebankWordDetokenizer().detokenize
    tokenizer = CustomTokenizer(tokenize_fn=tok_fn, detokenize_fn=detok_fn,
                                bos_token='<bos>', eos_token='<eos>',
                                pad_token='<pad>', unk_token='<unk>')

    data = pd.read_csv('processed_data/train_pairs.csv')
    all_sents = data['sent1'].tolist() + data['sent2'].tolist()
    tokenizer.fit(all_sents)
    tokenizer.save('processed_data/tokenizer.pkl')
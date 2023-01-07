from flask import Flask,request
from response import GrammerErrorResponse,BotMessageResponse
from GrammerChecker import check_grammer
import time 
from chatters.gpt import GPTChatter
from chatters.hmm import HMMChatter
from chatters.lstm import LstmModel, LSTMChatter
print("[INFO] GPT Model Loading...")
gpt_chatter = GPTChatter(model_path='../dialogue-models/trained_models/gpt_epoch_4')
print("[INFO] GPT Model Loaded...")


# print("[INFO] HMM Model Loading...")
# hmm_chatter = HMMChatter(tokenizer_name='spacy', max_len=50, special_tokens={'bos_token': '|BOS|', 'eos_token': '|EOS|', 'pad_token': '|PAD|', 'unk_token': '|UNK|'})
# print("[INFO] HMM Model Loaded...")

app = Flask(__name__)
def get_model_message(modelId,message):
    if modelId == 2:
        return gpt_chatter.chat(message, hidden_states=None)
        
    else:
        lstm_chatter = LSTMChatter(model_path='/Users/sefagokceoglu/workspace/c2l/chat2learn/dialogue-models/trained_models/lstm/lstm_model12.pt',
                                tokenizer_path='/Users/sefagokceoglu/workspace/c2l/chat2learn/dialogue-models/trained_models/lstm/lstm-tokenizer.pth')
        return gpt_chatter.chat(message, hidden_states=None)

 


@app.route('/chat/<int:modelId>',methods=['POST'])
def _chat(modelId):
    data = request.get_json()
    message = data['message']
    print("[INFO] ModelId: ",modelId)
    print("[INFO] Message: ",message)

    ans, hidden_states = get_model_message(modelId,message)
    
    baseResponse = BotMessageResponse(None, None)

    baseResponse.responseText= ans.strip()

    return baseResponse.to_json()
 
@app.route('/check',methods=['POST'])
def _check():
    data = request.get_json()
    message = data['message']

    grammerResponse = check_grammer(message)

    baseResponse = GrammerErrorResponse(None,None, [],0)
    baseResponse.taggedCorrectText= grammerResponse.taggedText
    baseResponse.correctText= grammerResponse.correctedSentence
    baseResponse.errorTypes = grammerResponse.errorTypes
    baseResponse.score = grammerResponse.score

    return baseResponse.to_json()
 
# main driver function
if __name__ == "__main__":
    print("[INFO] Starting Flask Server...")
    app.run(port=9090)
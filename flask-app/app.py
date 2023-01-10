from flask import Flask,request
from response import GrammerErrorResponse,BotMessageResponse
from GrammerChecker import check_grammer
import time 
import chatters
from chatters.gpt import GPTChatter
from chatters.hmm import hmm_chatter
from chatters.bert_gpt import BertGptChatter
from chatters.lstm import LSTMChatter, LstmModel


print("[INFO] LSTM Model Loading...")
lstm_chatter = LSTMChatter(model_path='/Users/sefagokceoglu/workspace/c2l/chat2learn/dialogue-models/trained_models/lstm/lstm_model12.pt',
                               tokenizer_path='/Users/sefagokceoglu/workspace/c2l/chat2learn/dialogue-models/trained_models/lstm/lstm-tokenizer.pth')
print("[INFO] LSTM Model Loaded...")

print("[INFO] BERT + GPT Model Loading...")
bert_gpt_chatter = BertGptChatter(model_path='/Users/sefagokceoglu/workspace/c2l/chat2learn/dialogue-models/trained_models/bert_gpt2_epoch_6')
print("[INFO] BERT + GPT Model Loaded...")

print("[INFO] GPT Model Loading...")
gpt_chatter = GPTChatter(model_path='../dialogue-models/trained_models/gpt_epoch_4')
print("[INFO] GPT Model Loaded...")


print("[INFO] HMM Model Loading...")
hmm_chatter.chat("We should hangout more often.")
print("[INFO] HMM Model Loaded... Time :" )


print("[INFO] APP Starting...")
app = Flask(__name__)
def get_model_message(modelId,message):
    if modelId == 1:
        print("[INFO] Sending Message to HMM Model -> ",message)
        return hmm_chatter.chat(message)
    if modelId == 2:
        print("[INFO] Sending Message to GPT2 Model -> ",message)
        return gpt_chatter.chat(message)
    if modelId == 3:
        print("[INFO] Sending Message to LSTM Model -> ",message)
        ans, state = lstm_chatter.chat(message)
        return ans
    if modelId == 4:
        print("[INFO] Sending Message to BERT + GPT Model -> ",message)
        return bert_gpt_chatter.chat(message)

 


@app.route('/chat/<int:modelId>',methods=['POST'])
def _chat(modelId):
    data = request.get_json()
    message = data['message']

    ans = get_model_message(modelId,message)
    
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
    app.run(port=9090)
# Import flask module
from flask import Flask,request
from response import GrammerErrorResponse,BotMessageResponse
from GrammerChecker import check_grammer
import time 
from models.gpt import GPTChatter

app = Flask(__name__)
 

@app.route('/chat/<int:modelId>',methods=['POST'])
def _chat(modelId):
    data = request.get_json()
    message = data['message']
    st = time.time()
    galip = GPTChatter(model_path='../dialogue-models/saved models/gpt_model')
    et = time.time()
    print("Time taken to load model: ", et-st)
    ans, hidden_states = galip.chat(message, hidden_states=None)
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
    app.run(port=9090,threaded=True)
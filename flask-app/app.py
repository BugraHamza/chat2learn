# Import flask module
from flask import Flask,request
from GrammerChecker import check_grammer
app = Flask(__name__)
class BaseResponse:
    def __init__(self, correctText, responseMessage, error_types):
        self.correctText = correctText
        self.responseMessage = responseMessage
        self.errorTypes = error_types
 
    def to_json(self):
        return {
            "correctText": self.correctText,
            "responseMessage": self.responseMessage,
            "errorTypes": self.errorTypes
        }
 
@app.route('/<int:modelId>',methods=['POST'])
def index(modelId):
    data = request.get_json()
    message = data['message']
    baseResponse = BaseResponse(None, None, [])
    grammerResponse = check_grammer(message)
    baseResponse.correctText= grammerResponse.correctText
    baseResponse.errorTypes = grammerResponse.errorTypes



    return baseResponse.to_json()
 
# main driver function
if __name__ == "__main__":
    app.run(port=9090)
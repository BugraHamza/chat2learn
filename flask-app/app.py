# Import flask module
from flask import Flask,request,jsonify
 
app = Flask(__name__)
 
@app.route('/<int:modelId>',methods=['POST'])
def index(modelId):
    data = request.get_json()
    message = data['message']

    response = {
        "correctText":"correct text",
        "responseMessage":"response message"
    }
    return jsonify(response)
 
# main driver function
if __name__ == "__main__":
    app.run(port=9090)
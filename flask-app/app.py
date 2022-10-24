# Import flask module
from flask import Flask
 
app = Flask(__name__)
 
@app.route('/<int:modelId>')
def index(modelId):
    stas = "Hello World from Flask with modelId: " + str(modelId)
    return stas
 
# main driver function
if __name__ == "__main__":
    app.run()
from flask import Flask, render_template, request
from FaceDetection import FaceDetection

app = Flask(__name__)

@app.route('/')
@app.route("/index")
def index():
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def run_script():
    # output=request.form.to_dict()
    # name=output["flag"]
    if request.method == 'POST':
        a = request.files['fileup']
        # a=str(a.filename)
        return FaceDetection.func(a.filename)
    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True, port= 5000)
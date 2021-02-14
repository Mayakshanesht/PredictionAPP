import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('ProfitPredictor.pkl', 'rb'))
ohe = pickle.load(open('StateEncoder.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    rdSpend = float(request.form['rdSpend'])
    admSpend = float(request.form['admSpend'])
    markSpend = float(request.form['markSpend'])
    state = request.form['state']
    stateEncoded = ohe.transform(np.array([[state]]))
    finalFeatures = np.concatenate((stateEncoded,np.array([[rdSpend,admSpend,markSpend]])) , axis = 1)
    prediction = model.predict(finalFeatures)

    

    return render_template('index.html', prediction_text='Expected Profit from the Startup is  $ {}'.format(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True)
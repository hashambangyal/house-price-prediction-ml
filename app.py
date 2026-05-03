from flask import Flask , request , jsonify
import joblib 
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#  loading the model and columns 
model = joblib.load('houe_price_predict.pkl')
columns = joblib.load('column.pkl')

@app.route('/')
def home():
    return " House Prediction API running "

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    df = pd.DataFrame([data])

    df = df.reindex(columns=columns , fill_value=0)

    pred = model.predict(df)
    price =  np.expm1(pred)
    print(price)

    return jsonify({
        "Predictiedprice": float(price[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
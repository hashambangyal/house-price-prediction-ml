from flask import Flask, request, jsonify, render_template
import joblib 
import pandas as pd
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load model and columns
model = joblib.load('houe_price_predict.pkl')
columns = joblib.load('column.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = df.reindex(columns=columns, fill_value=0)

    pred = model.predict(df)
    price = np.expm1(pred)

    return jsonify({"PredictedPrice": float(price[0])})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
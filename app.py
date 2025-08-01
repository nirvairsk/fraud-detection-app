from flask import Flask, request, jsonify
import os
import joblib
import numpy as np

app=Flask(__name__)

model_path=os.path.join(os.path.dirname(__file__),"model","model.pkl")
model=joblib.load(model_path)

@app.route('/predict',methods=['POST'])
def predict():
    data=request.get_json()

    if 'features' not in data:
        return jsonify({"error": "Missing 'features' in request"}),400
    
    features=data['features']

    try:
        input_array= np.array(features).reshape(1,-1)
    except Exception as e:
        return jsonify({"error": f"Invalid error format : {str(e)}"}),400

    try:
        prediction=model.predict(input_array)[0]
        return jsonify({"Prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}),500

@app.route('/', methods=['GET'])
def home():
    return "<h1>Fraud Detection App is Running</h1><p>Use POST /predict to send your data.</p>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)


            

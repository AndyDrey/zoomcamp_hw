from flask import Flask
from flask import request, jsonify
import pickle

app = Flask('predictor')

with open("/app/model2.bin", "rb") as m_in, open("/app/dv.bin", "rb") as d_in:
        model = pickle.load(m_in)
        dv = pickle.load(d_in)

@app.route('/predict', methods=["POST"])
def predict():
        client = request.get_json()
        X_val = dv.transform([client])

        y_pred = model.predict_proba(X_val)[0,1]

        result = {
            "credit_probability" : float(y_pred)
        }

        return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
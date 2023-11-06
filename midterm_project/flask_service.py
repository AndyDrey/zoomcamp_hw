from flask import Flask
from flask import request, jsonify
import pickle
import xgboost as xgb

app = Flask('predictor')

with open("xgb_model.bin", "rb") as m_in:
        dv, model = pickle.load(m_in)
        features = dv.get_feature_names_out()
        features= [i.replace("=<", "_") for i in features]


@app.route('/predict', methods=["POST"])
def predict():
        client = request.get_json()
        print("1")
        X_val = dv.transform([client])
        print("2")
        print(dv.get_feature_names_out())
        dval = xgb.DMatrix(X_val, feature_names=dv.get_feature_names_out().tolist())
        print("3")
        y_pred = model.predict(dval)
        print("4")
        result = {
            "stroke_probability" : float(y_pred)
        }

        return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
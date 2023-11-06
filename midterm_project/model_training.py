import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text
from sklearn.metrics import mean_squared_error, mutual_info_score
import xgboost as xgb
import pickle

# PARAMS
model_name = 'xgb_model.bin'

xgb_params = {
    'eta': 0.1, 
    'max_depth': 2,
    'min_child_weight': 5,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

to_remove = ["blood_pressure_levels", "cholesterol_levels", "symptoms"]
def data_preparation(df):
    print("Data preparation started")
    del df["Patient ID"]
    del df["Patient Name"]
    
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df['symptoms'] = df['symptoms'].fillna("No symptoms")
    df['diagnosis'] = (df['diagnosis'] == "Stroke").astype(int)
    tmp = pd.DataFrame(df["blood_pressure_levels"].str.split('/').tolist(), columns=["systolic", "diastolic"])
    tmp["systolic"] = tmp["systolic"].astype(int)
    tmp["diastolic"] = tmp["diastolic"].astype(int)
    df_with_features = df.join(pd.DataFrame(tmp))
    l = [(int(items[0].split(" ")[1]), int(items[1].split(" ")[1])) for items in df["cholesterol_levels"].str.split(', ')]
    df_with_features = df_with_features.join(pd.DataFrame(l, columns=["hdl", "ldl"]))

    unique_symptoms = set(symptom for symptoms in df['symptoms'].tolist() for symptom in symptoms.split(", "))
    vectorizer = CountVectorizer(vocabulary=unique_symptoms, binary=True, lowercase=True)
    symptoms_bow = vectorizer.fit_transform(df['symptoms'])
    df_with_features = df_with_features.join(pd.DataFrame(symptoms_bow.toarray(), columns=vectorizer.get_feature_names_out()))

    df_with_features.columns = df_with_features.columns.str.lower().str.replace(" ", "_")
    df_with_features.columns

    del df_with_features["blood_pressure_levels"]
    del df_with_features["cholesterol_levels"]
    del df_with_features["symptoms"]
    print("Data preparation finished")
    return df_with_features

def train():
    print("Training...")
    df = pd.read_csv("stroke_prediction_dataset.csv")
    df_with_features = data_preparation(df)

    df_full_train, df_test = train_test_split(df_with_features, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    y_full_train = df_full_train["diagnosis"].values
    y_test = df_test["diagnosis"].values
    y_train = df_train["diagnosis"].values
    y_val = df_val["diagnosis"].values

    del df_full_train["diagnosis"]
    del df_test["diagnosis"]
    del df_train["diagnosis"]
    del df_val["diagnosis"]

    dv = DictVectorizer(sparse=True)
    train_dict = df_train.to_dict(orient="records")
    val_dict = df_val.to_dict(orient="records")
    full_train_dict = df_full_train.to_dict(orient="records")
    X_train = dv.fit_transform(train_dict)
    X_val = dv.transform(val_dict)
    X_full_train = dv.transform(full_train_dict)

    features = dv.get_feature_names_out()
    features= [i.replace("=<", "_") for i in features]
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, y_val, feature_names=features)

    model = xgb.train(xgb_params, dtrain, num_boost_round=100)

    return dv, model

print("Model learning started!")
dv, model = train()
f_out = open(model_name, 'wb') 
pickle.dump((dv, model), f_out)
f_out.close()
print(f'The model is saved to {model_name}')

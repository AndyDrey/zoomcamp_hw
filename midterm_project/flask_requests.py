import requests

url = "http://localhost:9696/predict"
client = {
    "age": 56, 
    "gender": "Male", 
    "hypertension": 0,
    "heart_disease": 1, 
    "marital_status": "Married", 
    "work_type": "Self-employed",
    "residence_type": "Rural", 
    "average_glucose_level": 130.91, 
    "body_mass_index_(bmi)": 22.37,
    "smoking_status": "Non-smoker", 
    "alcohol_intake": "Social Drinker", 
    "physical_activity": "Moderate",
    "stroke_history": 0, 
    "family_history_of_stroke": "Yes", 
    "dietary_habits": "Vegan",
    "stress_levels": 3.48, 
    "blood_pressure_levels": "140/108", 
    "cholesterol_levels": "HDL: 68, LDL: 133",
    "symptoms": "Difficulty Speaking, Headache"}
print(requests.post(url, json=client).json())

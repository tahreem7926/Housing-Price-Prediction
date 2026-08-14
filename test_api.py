"""
Quick sanity check. Run the API first (uvicorn api:app --reload),
then in another terminal: python test_api.py
"""
import requests

payload = {
    "bedrooms": 4,
    "bathrooms": 3,
    "area_value": 10,
    "area_unit": "Marla",
    "location": "F-7, Islamabad",
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.status_code)
print(response.json())

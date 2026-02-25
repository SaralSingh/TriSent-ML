#ml_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="TriSent ML API")

# Load trained artifacts
vectorizer = joblib.load("models/vectorizer.joblib")
model = joblib.load("models/sentiment_model.joblib")

class TextInput(BaseModel):
    text: str

def format_response(label):
    return {
        "positive": {"label": "positive", "emoji": "😊"},
        "negative": {"label": "negative", "emoji": "😞"},
        "mixed": {"label": "mixed", "emoji": "😐"},
    }[label]

@app.post("/predict")
def predict_sentiment(input: TextInput):
    pred = model.predict(
        vectorizer.transform([input.text])
    )[0]

    return format_response(pred)
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
import numpy as np
import joblib
from .schemas import ReviewIn, PredictionOut

app = FastAPI(title='Sentiment API')
_model = None
_vectorizer = None

@app.on_event('startup')
def load_model():
    global _model, _vectorizer
    payload = joblib.load(app.state.model_path)
    _model = payload['model']
    _vectorizer = payload['vectorizer']

@app.post('/predict', response_model=PredictionOut)
def predict(payload: ReviewIn):
    text = payload.text
    if not text:
        raise HTTPException(status_code=400, detail='Empty text')
    vec = _vectorizer.transform([text])
    proba = _model.predict_proba(vec)[0]
    classes = _model.classes_.tolist()
    idx = int(np.argmax(proba))
    return PredictionOut(label=classes[idx], score=float(proba[idx]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    app.state.model_path = args.model
    uvicorn.run('src.api.app:app', host=args.host, port=args.port, reload=False)

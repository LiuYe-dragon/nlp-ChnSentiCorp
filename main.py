from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_text

app = FastAPI(title="BERT Chinese Text Classification API")

# 输入数据格式
class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "BERT API running successfully"}

@app.post("/predict")
def predict(input_data: TextInput):
    text = input_data.text
    label, prob = predict_text(text)
    return {"text": text, "label": label, "probability": prob}

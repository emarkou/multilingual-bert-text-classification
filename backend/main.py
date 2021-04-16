import asyncio
import time
import uvicorn
from fastapi import FastAPI
import inference
import os
import torch
import config
from transformers import BertForSequenceClassification
from pydantic import BaseModel

class document(BaseModel):
    value: str

app = FastAPI()

# @app.on_event("startup")
# async def load_model():
#     path_to_model = os.path.join(config.MODEL_PATH, 'pytorch_model.bin')
#     model = torch.load(path_to_model, map_location=torch.device('cpu'))


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/predict")
async def get_doc(doc: document):
    start = time.time()
    # find way to load at startup
    # path_to_model = os.path.join(config.MODEL_PATH, 'pytorch_model.bin')
    model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH)
    # model = torch.load(path_to_model, map_location=torch.device('cpu'))
    assigned_class = inference.inference(doc.value, model)
    return {"doc": doc, "time": time.time() - start, "assigned_class": assigned_class}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)


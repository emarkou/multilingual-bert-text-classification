import time
import uvicorn
from fastapi import FastAPI
import inference
import config
from transformers import BertForSequenceClassification
from pydantic import BaseModel

class document(BaseModel):
    value: str

app = FastAPI()

model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/predict")
async def get_doc(doc: document):
    start = time.time()
    assigned_class = inference.inference(doc.value, model)
    return {"doc": doc, "time": time.time() - start, "assigned_class": assigned_class}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)


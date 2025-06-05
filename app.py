from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from preprocess import preprocess_image  # make sure this still works for (1,28,28,1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your CNN model and class labels
cnn_model = keras.models.load_model('cnn_classifier_model.keras')
classes = ['flower', 'hat', 'bicycle', 'cat', 'tree', 'fish', 'candle', 'star', 'face', 'house']

# Load LLM
llm_model_name = "declare-lab/flan-alpaca-large"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = None


def load_model():
    global model
    if model is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)


# Story generation function
def generate_story(obj_class):
    prompt = f"Write a short and simple story for a 3-year-old child about a {obj_class}. Use easy words and make it fun."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=80,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        top_k=50,
        repetition_penalty=1.1,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_image(contents)  # Output: (1, 28, 28, 1)
    load_model()
    prediction = cnn_model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_label = classes[predicted_class_idx]

    story = generate_story(predicted_label)

    return JSONResponse(content={"prediction": predicted_label, "story": story})

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io
import base64
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from preprocess import preprocess_image
from models import CVAE_Encoder, CVAE_Decoder  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load classifier model
cnn_model = keras.models.load_model('cnn_classifier_model.keras')

# Define class names
classes = ['flower', 'hat', 'bicycle', 'cat', 'tree', 'fish', 'candle', 'star', 'face', 'house']

# Build and load encoder/decoder
latent_dim = 16 


encoder = keras.models.load_model('vae_encoder.keras', custom_objects={'CVAE_Encoder': CVAE_Encoder})
decoder = keras.models.load_model('vae_decoder.keras', custom_objects={'CVAE_Decoder': CVAE_Decoder})

# Load LLM
llm_model_name = "declare-lab/flan-alpaca-large"

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

def generate_story(obj_class):
    prompt = f"Write a short and simple story for a 3-year-old child about a {obj_class}. Use easy words and make it fun."

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
    **inputs,
    max_length=80,
    do_sample=True,           # Enables sampling
    top_p=0.95,               # Nucleus sampling
    temperature=0.8,          # Controls randomness (lower = more conservative)
    top_k=50,                 # Optional: adds randomness by sampling from top-k tokens
    repetition_penalty=1.1,   # Penalize repetition
    num_return_sequences=1    # Only 1 story
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Force build
dummy_noise = tf.random.normal((1, 16))
dummy_label = tf.one_hot([0], depth=10)
_ = decoder((dummy_noise, dummy_label))

decoder.summary()


def sample(mu, logvar):
    """
    Sample z from a normal distribution using the reparameterization trick:
    z = mu + sigma * epsilon
    where sigma = exp(0.5 * logvar)
    """
    eps = tf.random.normal(shape=tf.shape(mu))  # ε ~ N(0,1)
    return mu + tf.exp(0.5 * logvar) * eps

# Helper to convert array to base64 image
def array_to_base64(img_array):
    img_array = np.squeeze(img_array)
    img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    base64_encoded = base64.b64encode(byte_im).decode('utf-8')
    return base64_encoded



def visualize_debug(img_tensor, title="Image"):
    img = np.squeeze(img_tensor.numpy())  # Convert tensor -> numpy and remove batch/channel dims if needed
    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis('off')
    plt.show()



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    import matplotlib.pyplot as plt

    # Step 1. Read and preprocess uploaded image
    contents = await file.read()
    img_array = preprocess_image(contents)  # Shape (1,28,28,1)

    # Step 2. Predict class using CNN
    prediction = cnn_model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_label = classes[predicted_class_idx]

    # Step 3. Prepare one-hot label for CVAE
    label_onehot = tf.one_hot(predicted_class_idx, depth=len(classes))
    label_onehot = tf.expand_dims(label_onehot, axis=0)  # Shape (1, num_classes)

    # Step 4. Encoder-Decoder reconstruction
    # 🧠 Group image and label into one input
    mu, logvar = encoder((img_array, label_onehot))
    z = sample(mu, logvar)
    reconstructed = decoder((z, label_onehot))


    # Step 5. Random noise generation from label
    random_noise = tf.random.normal(shape=(1, latent_dim))
    random_generated = decoder((random_noise, label_onehot))

     # Generate story
    story = generate_story(predicted_label)


    # Step 7. Return only prediction and story to frontend (not images)
    return JSONResponse(content={"prediction": predicted_label, "story": story})





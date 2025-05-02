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


from preprocess import preprocess_image
from models import CVAE_Encoder, CVAE_Decoder  # 🧡 your custom models

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
latent_dim = 16  # or whatever you used

# encoder = CVAE_Encoder()
# encoder.build(input_shape=(None, 28, 28, 1))  # make sure it matches
# encoder.load_weights('encoder.weights.h5')

# decoder = CVAE_Decoder()
# decoder.build(input_shape=(None, latent_dim))
# decoder.load_weights('decoder.weights.h5')


encoder = keras.models.load_model('vae_encoder.keras', custom_objects={'CVAE_Encoder': CVAE_Encoder})
decoder = keras.models.load_model('vae_decoder.keras', custom_objects={'CVAE_Decoder': CVAE_Decoder})

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


    # Step 6. Plot all three images side by side
    # fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # axs[0].imshow(img_array[0, :, :, 0], cmap='gray')
    # axs[0].set_title("1. Input Sketch")
    # axs[0].axis('off')

    # axs[1].imshow(reconstructed[0, :, :, 0], cmap='gray')
    # axs[1].set_title("2. Reconstructed Sketch")
    # axs[1].axis('off')

    # axs[2].imshow(random_generated[0, :, :, 0], cmap='gray')
    # axs[2].set_title(f"3. Random {predicted_label} Sketch")
    # axs[2].axis('off')

    # plt.tight_layout()
    # plt.show()

    # print(decoder.summary())
    print(tf.reduce_mean(decoder((tf.random.normal((1, latent_dim)), tf.one_hot([0], depth=len(classes))))))


    # Step 7. Return only prediction to frontend (not images)
    return JSONResponse(content={"prediction": predicted_label})




# 🆕 New generate route (returns base64 images)
# @app.post("/generate")
# async def generate(file: UploadFile = File(...), label: str = Form(...)):
#     contents = await file.read()
#     img_array = preprocess_image(contents)  # (1, 28, 28, 1)

#     # Reconstruct user sketch
#     mu, logvar = encoder((img_array, "label_onehot"))
#     eps = tf.random.normal(shape=tf.shape(mu))
#     z = mu + tf.exp(0.5 * logvar) * eps
#     reconstructed = decoder(z)

#     # Generate a fresh random image from label
#     label_idx = classes.index(label.lower())
#     label_one_hot = tf.one_hot(label_idx, depth=len(classes))
     
#      # ➡️ Expand dims to add batch
#     label_one_hot = tf.expand_dims(label_one_hot, axis=0)   
    
#     label_dense = tf.keras.layers.Dense(latent_dim)(label_one_hot)
#     noise = tf.random.normal(shape=(1, latent_dim))
#     latent_input = noise + label_dense
#     generated = decoder(latent_input)

#     # Convert both images to base64
#     reconstructed_b64 = array_to_base64(reconstructed)
#     generated_b64 = array_to_base64(generated)

#     # visualize_debug(reconstructed, title="Reconstructed Image")
#     # visualize_debug(generated, title="Generated Image") 

#     return JSONResponse(content={
#         "reconstructed_image": reconstructed_b64,
#         "generated_image": generated_b64
#     })

import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable

latent_dim = 16  # or whatever you used, keep it same everywhere
image_shape = (28, 28, 1)  # assuming your input images are 28x28 grayscale

@register_keras_serializable()
class CVAE_Encoder(Model):
    def __init__(self):
        super().__init__()
        # Define your layers
        self.label_dense = layers.Dense(28*28, activation='relu')
        self.reshape = layers.Reshape((28, 28, 1))
        self.conv1 = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')
        self.conv2 = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.fc_mu = layers.Dense(latent_dim)
        self.fc_logvar = layers.Dense(latent_dim)

    def call(self, inputs):
        x, label = inputs  # 🧡 unpack here
        label_info = self.label_dense(label)
        label_img = self.reshape(label_info)
        x = tf.concat([x, label_img], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


@register_keras_serializable()
class CVAE_Decoder(Model):
    def __init__(self):
        super().__init__()
        self.fc = layers.Dense(7*7*64, activation='relu')
        self.reshape = layers.Reshape((7, 7, 64))
        self.deconv1 = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')
        self.deconv2 = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')
        self.deconv3 = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')
        self.output_layer = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')

    def call(self, inputs):
        z, label = inputs  # 🧡 Unpack inside!
        z_cond = tf.concat([z, label], axis=-1)
        x = self.fc(z_cond)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.output_layer(x)
        return x

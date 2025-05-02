import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable

@register_keras_serializable()
class CVAE_Encoder(Model):
    def __init__(self, latent_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim  # 🧡 save hyperparameters

        self.label_dense = layers.Dense(28*28, activation='relu')
        self.reshape = layers.Reshape((28, 28, 1))
        self.conv1 = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')
        self.conv2 = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.fc_mu = layers.Dense(latent_dim)
        self.fc_logvar = layers.Dense(latent_dim)

    def call(self, inputs):
        x, label = inputs
        label_info = self.label_dense(label)
        label_img = self.reshape(label_info)
        x = tf.concat([x, label_img], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class CVAE_Decoder(Model):
    def __init__(self, latent_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim  # 🧡 Save hyperparameters

        self.fc = layers.Dense(7*7*64, activation='relu')
        self.reshape = layers.Reshape((7, 7, 64))
        self.deconv1 = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')
        self.deconv2 = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')
        self.deconv3 = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')
        self.output_layer = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')

    def call(self, inputs):
        z, label = inputs
        z_cond = tf.concat([z, label], axis=-1)
        x = self.fc(z_cond)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.output_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

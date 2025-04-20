import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom")
def angular_loss(y_true, y_pred):
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    cos_loss = 1 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(y_true, y_pred)
    return 0.7 * cos_loss + 0.3 * mse_loss

class DenseBlock(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_norms1 = [layers.BatchNormalization() for _ in range(4)]
        self.activations1 = [layers.LeakyReLU(0.1) for _ in range(4)]
        self.convs1 = [layers.Conv2D(32, 1, padding='same', kernel_initializer='he_normal') for _ in range(4)]
        self.batch_norms2 = [layers.BatchNormalization() for _ in range(4)]
        self.activations2 = [layers.LeakyReLU(0.1) for _ in range(4)]
        self.convs2 = [layers.Conv2D(8, 3, padding='same', kernel_initializer='he_normal') for _ in range(4)]

    def call(self, x, training=False):
        concatenated = [x]
        for i in range(4):
            x = tf.concat(concatenated, axis=-1)
            x = self.batch_norms1[i](x, training=training)
            x = self.activations1[i](x)
            x = self.convs1[i](x)
            x = self.batch_norms2[i](x, training=training)
            x = self.activations2[i](x)
            x = self.convs2[i](x)
            concatenated.append(x)
        return tf.concat(concatenated, axis=-1)

class TransitionLayer(layers.Layer):
    def __init__(self, output_filters, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(output_filters, 1, kernel_initializer="he_normal")
        self.bn = layers.BatchNormalization()
        self.pool = layers.AveragePooling2D(2)
        self.act = layers.LeakyReLU(0.1)

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.pool(x)
        x = self.act(x)
        return x

class HourglassModule(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(32, 3, strides=1, padding='same', kernel_initializer="he_normal")
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.LeakyReLU(0.1)
        self.conv2 = layers.Conv2D(32, 3, strides=2, padding='same', kernel_initializer="he_normal")
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.LeakyReLU(0.1)
        self.deconv = layers.Conv2DTranspose(32, 3, strides=2, padding='same', kernel_initializer="he_normal")
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.LeakyReLU(0.1)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.deconv(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        return x

@register_keras_serializable(package="Custom", name="DPG")
class DPG(tf.keras.Model):
    def __init__(self, input_shape=(80, 120, 3), **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = input_shape
        
        self.conv1 = layers.Conv2D(32, 7, strides=2, padding="same", kernel_initializer="he_normal")
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.LeakyReLU(0.1)

        self.hourglass_modules = [HourglassModule() for _ in range(3)]
        self.dense_blocks = []
        self.transition_layers = []
        
        current_filters = 32
        for i in range(4):
            self.dense_blocks.append(DenseBlock())
            if i < 3:
                current_filters += 32  
                transition_filters = int(current_filters * 0.8) 
                self.transition_layers.append(TransitionLayer(transition_filters))
                current_filters = transition_filters

        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.global_max_pool = layers.GlobalMaxPooling2D()

        self.output_layers = models.Sequential([
            layers.Dense(64, activation='swish', kernel_initializer="he_normal"),
            layers.Dropout(0.3),
            layers.Dense(3) 
        ])

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        for hg_module in self.hourglass_modules:
            residual = x
            x = hg_module(x, training=training)
            x = x + residual

        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x, training=training)
            if i < len(self.transition_layers):
                x = self.transition_layers[i](x, training=training)

        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        combined = tf.concat([avg_pool, max_pool], axis=1)

        return self.output_layers(combined)

    def build_graph(self):
        x = layers.Input(shape=self.input_shape_)
        return models.Model(inputs=[x], outputs=self.call(x))

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_
        })
        return config

def load_model(model_path="model_tf.keras"):
    return tf.keras.models.load_model(
        model_path, 
        custom_objects={
            'Custom>DPG': DPG,
            'Custom>angular_loss': angular_loss
        }
    )

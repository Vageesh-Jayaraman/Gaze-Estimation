import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom", name="angular_loss")
def angular_loss(y_true, y_pred):
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    cos_loss = 1 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))
    
    mse = tf.keras.losses.MeanSquaredError()  
    mse_loss = mse(y_true, y_pred) 
    
    return 0.7 * cos_loss + 0.3 * mse_loss

@register_keras_serializable(package="Custom", name="DPG")
class DPG(tf.keras.Model):
    def __init__(self, input_shape=(80, 120, 3), num_modules=3, num_feature_maps=32,
                 growth_rate=8, compression_factor=0.8, num_dense_blocks=4, **kwargs):
        super(DPG, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.num_modules = num_modules
        self.num_feature_maps = num_feature_maps
        self.growth_rate = growth_rate
        self.compression_factor = compression_factor
        self.num_dense_blocks = num_dense_blocks

        # Initial convolution layer with proper initialization
        self.initial_conv = layers.Conv2D(
            filters=num_feature_maps,
            kernel_size=7,
            strides=2,
            padding="same",
            kernel_initializer="he_normal"
        )
        self.initial_bn = layers.BatchNormalization()
        self.initial_activation = layers.LeakyReLU(0.1)

        self.hourglass_modules = [self._build_hourglass_module() for _ in range(num_modules)]

        self.dense_blocks = [self._build_dense_block() for _ in range(num_dense_blocks)]
        self.transition_layers = [self._build_transition_layer() for _ in range(num_dense_blocks - 1)]

        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.global_max_pool = layers.GlobalMaxPooling2D()
        
        self.output_layers = models.Sequential([
            layers.Dense(64, activation='swish', kernel_initializer="he_normal"),
            layers.Dropout(0.3),
            layers.Dense(3)  
        ])

    def call(self, inputs, training=False):
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_activation(x)

        for hg_module in self.hourglass_modules:
            residual = x
            x = hg_module(x, training=training)
            x = x + residual  

        for i in range(self.num_dense_blocks):
            x = self.dense_blocks[i](x, training=training)
            if i < self.num_dense_blocks - 1:
                x = self.transition_layers[i](x, training=training)

        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        combined = tf.concat([avg_pool, max_pool], axis=1)
        
        return self.output_layers(combined)

    def _build_hourglass_module(self):
        return models.Sequential([
            layers.Conv2D(self.num_feature_maps, 3, padding='same', 
                        kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.Conv2D(self.num_feature_maps, 3, strides=2, padding='same',
                        kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.Conv2DTranspose(self.num_feature_maps, 3, strides=2, 
                                 padding='same', kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1)
        ])

    def _build_dense_block(self):
        return models.Sequential([
            layers.Conv2D(4 * self.growth_rate, 1, activation='linear',
                        kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.Conv2D(self.growth_rate, 3, padding='same', activation='linear',
                        kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1)
        ])

    def _build_transition_layer(self):
        return models.Sequential([
            layers.Conv2D(int(self.num_feature_maps * self.compression_factor), 1,
                        kernel_initializer="he_normal"),
            layers.AveragePooling2D(2),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'num_modules': self.num_modules,
            'num_feature_maps': self.num_feature_maps,
            'growth_rate': self.growth_rate,
            'compression_factor': self.compression_factor,
            'num_dense_blocks': self.num_dense_blocks,
        })
        return config

def load_model(model_path="best_model.keras"):
    return tf.keras.models.load_model(
        model_path, 
        custom_objects={
            'Custom>DPG': DPG,
            'Custom>angular_loss': angular_loss
        }
    )

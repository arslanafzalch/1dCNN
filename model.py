from tensorflow import keras
from tensorflow.keras import layers, models
from scipy import signal
import tensorflow as tf

class BandpassFilterLayer(layers.Layer):
    def __init__(self, lowcut, highcut, fs, order=4, **kwargs):
        super(BandpassFilterLayer, self).__init__(**kwargs)
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

    def build(self, input_shape):
        super(BandpassFilterLayer, self).build(input_shape)

    def butterworth_bandpass_filter(self, x):
        def _butterworth_bandpass_filter(x):
            nyquist = 0.5 * self.fs
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            b, a = signal.butter(self.order, [low, high], btype='band')
            return signal.filtfilt(b, a, x)

        return tf.py_function(_butterworth_bandpass_filter, [x], tf.float32)

    def call(self, inputs):
        outputs = tf.map_fn(self.butterworth_bandpass_filter, inputs)
        # Set the output shape to be the same as the input shape
        outputs.set_shape(inputs.shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class MyModel:
    def __init__(self, input_shape=(100, 6), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def inverted_residual_block(self, x, filters, kernel_size, strides):
        # Depthwise separable convolution
        x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Pointwise convolution
        x = layers.Conv2D(filters, kernel_size=1, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Skip connection if input and output shapes are the same, and strides are 1
        if strides == 1 and x.shape[-1] == filters:
            return layers.Add()([x, x])  # Change 'inputs' to 'x'
        return x

    def build_feature_extractor(self):
        inputs = keras.Input(shape=self.input_shape)

        # Apply Butterworth bandpass filters and process each channel
        processed_channels = []
        for i in range(self.input_shape[-1]):
            channel_input = layers.Lambda(lambda x: x[:, :, i])(inputs)
            # Butterworth bandpass filtering using custom layer
            lowpass = BandpassFilterLayer(0.22, 8, 100)(channel_input)
            mediumpass = BandpassFilterLayer(8, 32, 100)(channel_input)
            highpass = BandpassFilterLayer(32, 100, 100)(channel_input)

            lowpass_expanded = tf.keras.backend.expand_dims(lowpass, axis=-1)
            mediumpass_expanded = tf.keras.backend.expand_dims(mediumpass, axis=-1)
            highpass_expanded = tf.keras.backend.expand_dims(highpass, axis=-1)
            channel_input_expanded = tf.keras.backend.expand_dims(channel_input, axis=-1)
            
            concatenated_channel = layers.Concatenate()([lowpass_expanded, mediumpass_expanded, highpass_expanded, channel_input_expanded])


            # Concatenate the filtered signals and raw channel input
            # concatenated_channel = layers.Concatenate(axis=-1)([lowpass, mediumpass, highpass, channel_input])

            
            # Reshape to 100 x 4
            # reshaped_channel = layers.Reshape((100, 4, 1))(concatenated_channel)

            # Detrend each 100x4 stream
            detrended_channel = layers.TimeDistributed(layers.Lambda(lambda x: x - keras.backend.mean(x, axis=1)))(concatenated_channel)

            # Apply MbConv1
            x = self.inverted_residual_block(detrended_channel, filters=32, kernel_size=(3, 3), strides=1)

            # Apply MbConv6
            x = self.inverted_residual_block(x, filters=64, kernel_size=(3, 3), strides=1)

            processed_channels.append(x)

        # Concatenate the processed channels
        x = layers.Concatenate(axis=-1)(processed_channels)  # Concatenate along the last axis
        print(x.shape)
        # Additional processing
        x = layers.SeparableConv1D(128, 3, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Flatten()(x)

        return keras.Model(inputs=inputs, outputs=x)

    def build_classifier(self, feature_extractor):
        x = layers.Dense(80, activation='relu')(feature_extractor.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(40, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(20, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(10, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return keras.Model(inputs=feature_extractor.input, outputs=outputs)

    def build_model(self):
        feature_extractor = self.build_feature_extractor()
        classifier = self.build_classifier(feature_extractor)
        return classifier

    def compile_model(self):
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"]
        )

    def summary(self):
        self.model.summary()


# Instantiate the model
my_model = MyModel()

# Compile and display the model summary
my_model.compile_model()
my_model.summary()


from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class CharacterEmbeddingLayer(Layer):
    def __init__(self, output_dim,  **kwargs):
        # output_dim = embedding_dimension
        self.output_dim = output_dim
        super(CharacterEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[3], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(CharacterEmbeddingLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(CharacterEmbeddingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

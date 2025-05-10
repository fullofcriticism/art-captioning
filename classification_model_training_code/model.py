import tensorflow as tf
from keras.applications import EfficientNetV2L
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from keras.layers import BatchNormalization

def create_model(base_model, input_shape, output_shape, base_trainable=False):
    base_model.trainable = base_trainable
    model = Sequential([
        tf.keras.Input(shape=input_shape), 
        base_model, 
        BatchNormalization(), 
        Dense(512, activation='relu'), 
        Dropout(0.5),
        Dense(256, activation='relu'), 
        Dropout(0.5), 
        Dense(output_shape, activation='sigmoid')
        ])
    
    model.compile(
        optimizer='adamW',
        loss=tf.losses.BinaryCrossentropy(),
        metrics=[
            tf.metrics.BinaryAccuracy(), 
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(), 
            tf.keras.metrics.AUC()
        ]
    )
    return model

def load_pretrained_model(pretrained_model_path=None, input_shape=None, output_shape=None):
    if pretrained_model_path:
        return tf.keras.models.load_model(pretrained_model_path)
    else:
        base_model = EfficientNetV2L(weights="imagenet", include_top=False, pooling="avg")
        return create_model(base_model, input_shape, output_shape)
import numpy as np
import tensorflow as tf
# import cv2
model = tf.keras.models.load_model("happy_vs_sadA.h5")
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#dictionary to label all traffic signs class.
#initialise GUI

def classify(image):
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    normalized = image/255.0
    reshaped = np.reshape(normalized, (1, 150, 150, 3))
    reshaped = np.vstack([reshaped])
    pred = model.predict(reshaped)
    if pred<=0.5:
        an='Happy'
    else:
        an='Sad'
    return an

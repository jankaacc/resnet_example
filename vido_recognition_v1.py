import cv2
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Input
from keras.applications.resnet50 import (preprocess_input,
					 decode_predictions,
                                         ResNet50)
from PIL import Image

from capture_manager import VideoCaptureManager, VideoCaptureRecognizer


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
K.tensorflow_backend.set_session(tf.Session(config=config))


class Resnet_model():

    def __init__(self, include_top=True,
                 weights='imagenet', input_shape=(224, 224, 3)):
	
	self.img_input = Input(shape=input_shape)
	self.resnet_model = ResNet50(include_top=include_top, weights=weights,
                                     input_tensor=self.img_input)
        
    def make_prediction(self, img_array, verbouse_level, top):
        predictions = self.resnet_model.predict(img_array)
        if verbouse_level == 1:
            print('Predicted:', decode_predictions(predictions, top=top)[0])
            return predictions
        else:
            return predictions
        

if __name__ == '__main__':
    
    with VideoCaptureManager(device_nr=0, output_size=(224,224), cnn_model=Resnet_model()) as cap:     
         cap.capture()

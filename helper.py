# import cv2
import os
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models,utils
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras import utils
current_path = os.getcwd()
dog_breeds_category_path = os.path.join(current_path, 'static/dog_breeds_category.pickle')
predictor_model = load_model(r'static/dogbreed.h5')
with open(dog_breeds_category_path, 'rb') as handle:
    dog_breeds = pickle.load(handle)

# feature_extractor = load_model(r'static\feature_extractor.h5')
from keras.applications.resnet_v2 import ResNet50V2 , preprocess_input as resnet_preprocess
from keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.models import Model

input_shape = (331,331,3)
input_layer = Input(shape=input_shape)


#first extractor inception_resnet
preprocessor_resnet = Lambda(resnet_preprocess)(input_layer)
inception_resnet = ResNet50V2(weights = 'imagenet',
                                     include_top = False,input_shape = input_shape,pooling ='avg')(preprocessor_resnet)

preprocessor_densenet = Lambda(densenet_preprocess)(input_layer)
densenet = DenseNet121(weights = 'imagenet',
                                     include_top = False,input_shape = input_shape,pooling ='avg')(preprocessor_densenet)


merge = concatenate([inception_resnet,densenet])
feature_extractor = Model(inputs = input_layer, outputs = merge)


def predictor(img_path): # here image is file name 
    # base_path = os.path.join(current_path, 'static\images\cache')
    # path = os.path.join(base_path,image_name)
    img = load_img(img_path, target_size=(331,331))
    # print(path)
    # img = cv2.resize(img,(331,331))
    img = img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    features = feature_extractor.predict(img)
    prediction = predictor_model.predict(features)*100
    prediction = pd.DataFrame(np.round(prediction,1),columns = dog_breeds).transpose()
    prediction.columns = ['values']
    prediction  = prediction.nlargest(5, 'values')
    prediction = prediction.reset_index()
    prediction.columns = ['name', 'values']
    return(prediction)

    


# print(predictor('samoyed_puppy_dog_pictures.jpg'))



# img = cv2.imread(r'C:\Users\Abhishek\Desktop\dog_breed classifier\static\images\sample.jpg')
# img = cv2.resize(img,(331,331))
# print(img.shape)

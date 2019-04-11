from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import cv2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import skimage.io as ski
from sklearn.metrics import accuracy_score
import numpy as np

# Template for python script Inference

# JSON File Contents:
# {
#     "data": <It can be an image URL or text>
#     "model_path": <absolute path to model file>,
#     "secondary_model_path": <absolute path to 2nd model file>,
#     "label_path": <absolute path to label/class file>,
# }

import argparse, json
# from helper import getImage, deleteFile

# Parse JSON data file
ap = argparse.ArgumentParser()
ap.add_argument("-j", "--json", required=True,help="path to JSON data file")
args = vars(ap.parse_args())

with open(args["json"]) as f:
    data = json.load(f)

#  Input Variable (Example Variable, add/remove according to usage)
# DATA = data['data']
PATH_TO_FILE = data['data'] # input image
PATH_TO_MODEL = data["model_path"] # model
PATH_TO_LABELS = data["label_path"] # class.json
RETURN_IMAGE = data['return_image']

with open(PATH_TO_LABELS, 'r') as g:
    inference_json = json.load(g)

labeled_class = inference_json['class']
print(labeled_class)
task = inference_json['task']

# img_path = "normal.png"

def keras_inference(input_image, model_type, labels, return_image):

    """
    Getting a tensor from a given path.
    """
    # Loading the image
    img = image.load_img(input_image, target_size=(50, 50))
    # Converting the image to numpy array
    x = image.img_to_array(img)   
    # convert 3D tensor to 4D tensor with shape (1, 512, 512, 3)
    x =  np.expand_dims(x, axis=0)

    image_to_predict = x.astype('float32')/255
    
    # image_to_plot = path_to_tensor(input_image)

    # model's weight for localization
    model = load_model(model_type)
    prediction = model.predict(image_to_predict)
    # print("X shape : ", x.shape)
    # prediction_final = "Not_cancer: " + str(np.round(prediction[0][0]*100, decimals = 2)) + "%" + \
    #                     " | Cancer: " + str(np.round(prediction[0][1]*100, decimals = 2)) + "%"
    print("Prediction : ",prediction[0])
    print("Argmax : ", np.argmax(prediction[0]))
    confidence = np.max(prediction[0]) * 100
    classify = labeled_class[int(np.argmax(prediction[0]))]
    print("classify :", classify)
    output = {
            "label": "{}".format(task),
            "type" : "classification",
            "output" : {
                "confidence" : "{0:.2f}".format(round(confidence,2)),
                "results" : classify,
                "image" : return_image
            }
        } 
    
    return output
# print(prediction)
# print(prediction_final)

print(keras_inference(PATH_TO_FILE, PATH_TO_MODEL, PATH_TO_LABELS, RETURN_IMAGE), end='')







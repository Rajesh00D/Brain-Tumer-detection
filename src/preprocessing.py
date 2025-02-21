import os
import pandas as pd
import numpy as np
import cv2
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from pymongo import MongoClient


file_p = []
labels = []

#This function loads training image file paths from a list of directories and associates them with class labels.
def load_images(path_list,class_labels):
    
    for i, dir_list in enumerate(path_list):
     for filename in os.listdir(dir_list):
                fpath = os.path.join(dir_list, filename)
                file_p.append(fpath)
                labels.append(class_labels[i])

    filepath = pd.Series(file_p, name="filepaths")    
    Labels = pd.Series(labels, name="labels")
    data = pd.concat([filepath, Labels], axis=1)
    data = pd.DataFrame(data)
    
    return data

#This function loads testing image file paths from a list of directories and associates them with class labels.
def load_eval_images(dir):
    file_p = []
    labels = [] 
    
    for i,filename in enumerate(os.listdir(dir)):
                fpath = os.path.join(dir, filename)
                file_p.append(fpath)
                if i==0:
                    labels.append('Healthy')
                else:
                    labels.append('Brain_Tumor')

    filepath = pd.Series(file_p, name="filepaths")    
    Labelss = pd.Series(labels, name="labels")
    data = pd.concat([filepath, Labelss], axis=1)
    data = pd.DataFrame(data)
    
    return data


#This function loads training image file paths from a database.

def load_from_database(url):
    try:
        client = MongoClient(url)
     
    except Exception as e:
        print(e)
         
    database = client['DL-Project']
    collection = database['Brain-Tumor-Type']
    
    cursor = collection.find()
    
    data = list(cursor)
    
    data = pd.DataFrame(data)
    
    return data


#This function loads testing image file paths from a database.

def load_test_from_database(url):
    try:
        client = MongoClient(url)
     
    except Exception as e:
        print(e)
         
    database = client['DL-Project']
    collection = database['Brain-Tumor-Type-Test']
    
    cursor = collection.find()
    
    data = list(cursor)
    
    data = pd.DataFrame(data)
    
    return data

#enhance_image function is used to enchance the input image
def enhance_image(image):

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    hue = image[:, :, 0]
    saturation = image[:, :, 1]
    value = image[:, :, 2]
    value = np.clip(value * 1.25, 0, 255)

    image[:, :, 2] = value

    return image


# This function preprocesses the images from the provided training and testing dataframes
def preprocessing_images(train_df,test_df):
    
    image_gen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=lambda image: enhance_image(image))
    
    
    train = image_gen.flow_from_dataframe(dataframe= train_df,x_col="filepaths",y_col="labels",
                                      target_size=(256,256),
                                      color_mode='rgb',
                                      class_mode="categorical", 
                                      batch_size=64,
                                      shuffle= True            
                                     )
    test = image_gen.flow_from_dataframe(dataframe= test_df,x_col="filepaths", y_col="labels",
                                     target_size=(256,256),
                                     color_mode='rgb',
                                     class_mode="categorical",
                                     batch_size=64,
                                     shuffle= True  
                                    )
   
    
    
    return  train,test
    
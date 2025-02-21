# from pymongo import MongoClient
# from dotenv import load_dotenv
# import os
# import pandas as pd

# load_dotenv()

# url = os.getenv('url')

# client = MongoClient(url)

# database = client['DL-Project']
# collection = database['Brain-Tumor-Type']


# glioma = "/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Training/glioma"
# meningoma = "/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Training/meningioma"
# notumor = "/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Training/notumor"
# pituitarey = "/home/shiva/Desktop/ML/Files/Brain_Tumor dataset/Training/pituitary"



# path_list = [glioma,meningoma,notumor,pituitarey]
# class_labels = ['glioma','meningioma','notumor','pituitary']

# # Brain_Tumor = "/home/shiva/Desktop/ML/Files/Data/Brain Tumor"
# # Healthy ="/home/shiva/Desktop/ML/Files/Data/Healthy"

# # path_list = [Brain_Tumor, Healthy]
# # class_labels = ['BrainTumor', 'Healthy']


# file_p = []
# labels = []


# for i, dir_list in enumerate(path_list):
#     for filename in os.listdir(dir_list):
#                 fpath = os.path.join(dir_list, filename)
#                 file_p.append(fpath)
#                 labels.append(class_labels[i])

# filepath = pd.Series(file_p, name="filepaths")    
# Labelss = pd.Series(labels, name="labels")
# data = pd.concat([filepath, Labelss], axis=1)
# data = pd.DataFrame(data)


# dict = data.to_dict(orient="records")
 
# result = collection.insert_many(dict)

# # print(result)


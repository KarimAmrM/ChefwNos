import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from torchvision import transforms
import torch
from image_loader import ImageLoader
from model import Resnet34
from sklearn.cluster import KMeans
from utils import divide_video, extract_features
import pickle as pkl


#if frames folder exists then delete it
#then create a new frames folder
current_dir = os.path.dirname(os.path.realpath(__file__))
frame_path = os.path.join(current_dir,"frames")
if os.path.exists(frame_path):
    #delete frames folder
    for file in os.listdir(frame_path):
        os.remove(os.path.join(frame_path,file))
    os.rmdir(frame_path)
os.makedirs(frame_path)



video_name = "recipe1.mp4"
extract_features(video_name)

model = Resnet34(output_layer = "layer4")
model.eval()

#define transforms needed for model 
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

#define image loader
#get current directory where main.py is located
current_dir = os.path.dirname(os.path.realpath(__file__))
frame_path = os.path.join(current_dir,"frames")
image_loader = ImageLoader(frame_path,preprocess)
image_loader.load_images()
image_loader.preprocess()

#get output from model
output = []
for image in image_loader.images_preprocessed:
    output.append(model(image))
    
#prepare output for kmeans
output = output[0].detach().numpy()
output = output.reshape(output.shape[0], -1)

kmean = KMeans(n_clusters=5, random_state=0).fit(output)

# for i in range(0, len(kmean.labels_)):
#     #display images in cluster and annotate cluster number in title
#     plt.imshow(image_loader.images[i])
#     plt.title("Cluster: " + str(kmean.labels_[i]))
#     plt.show()
    
scenes = divide_video(image_loader.images,image_loader.image_names,kmean.labels_)
print(scenes)

path = os.path.join(current_dir,"scenes")
if not os.path.exists(path):
    os.makedirs(path)

pkl.dump(scenes,open(os.path.join(path,"scenes.pkl"),"wb"))
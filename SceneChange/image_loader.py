import os
from PIL import Image
from torchvision import transforms
import torch

class ImageLoader():
    def __init__(self,frames_path, transform = None):
        self.images = []
        self.folder_path = frames_path
        self.transform = transform
        self.images_preprocessed = []
        self.image_names = []
        
    def load_images(self):
        for filename in sorted(os.listdir(self.folder_path)):
            img = Image.open(os.path.join(self.folder_path,filename))
            self.images.append(img)
            
        image_names = [x.filename.split("\\")[-1] for x in self.images]
        #sort images_names by frame number in name frame#.jpg
        image_names = [int(x.split(".")[0].split("frame")[1]) for x in image_names]
        #sort images by name
        self.images = [x for _,x in sorted(zip(image_names,self.images))]
        self.image_names = [x for _,x in sorted(zip(image_names,image_names))]
    
    def preprocess(self):
        if self.transform != None:
            for img in self.images:
                self.images_preprocessed.append(self.transform(img))
        self.images_preprocessed = torch.stack(self.images_preprocessed)
        self.images_preprocessed = self.images_preprocessed.unsqueeze(0)
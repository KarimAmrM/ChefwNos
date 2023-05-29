import numpy as np
import pandas as pd
import time
import cv2
import os
from scipy.sparse import csc_matrix
from sklearn.cluster import Birch
from scipy.sparse.linalg import svds, eigs

def divide_video(images, images_name, kmean_labels):
    #divide video into scenes based on kmeans labels
    #when label changes scene changes and record the frame number from the image name
    scenes = []
    scene = []
    scene.append(images_name[0])
    for i in range(1,len(images_name)):
        if kmean_labels[i] != kmean_labels[i-1]:
            scene.append(images_name[i])
            scenes.append(scene)
            scene = []
        else:
            scene.append(images_name[i])
    scenes.append(scene)
    #get first frame of each scene
    first = [[x[0]] for x in scenes]
    #check that the first frame of each scene is at least 50 frames away from the previous scene
    #if not then merge the scenes
    for i in range(1,len(first)):
        if first[i][0] - first[i-1][0] < 200:
            scenes[i-1] = scenes[i-1] + scenes[i]
            scenes[i] = []
    scenes = [x for x in scenes if x != []]
    return scenes

#opens video specified and extract key frames in it
#
def get_histogram_features(video_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    frame_path = os.path.join(current_dir, video_name)
    cap = cv2.VideoCapture(frame_path)
    
    color_histograms = np.empty((0, 1944), int)
    frames = dict()
    frame_count = 0 
    
    #print with time that we started
    print("started analyzing {}".format(time.time()))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #since cv reads frame in bgr order so rearraning to get frames in rgb order
            frames[frame_count] = frame_rgb   #storing each frame (array) to frames , so that we can identify key frames later 
            #dividing a frame into 3*3 i.e 9 blocks
            height, width, channels = frame_rgb.shape
            if height % 3 == 0:
                h_chunk = int(height/3)
            else:
                h_chunk = int(height/3) + 1

            if width % 3 == 0:
                w_chunk = int(width/3)
            else:
                w_chunk = int(width/3) + 1
            h=0
            w= 0 
            feature_vector = []
            for a in range(1,4):
                h_window = h_chunk*a
                for b in range(1,4):
                    frame = frame_rgb[h : h_window, w : w_chunk*b , :]
                    hist = cv2.calcHist(frame, [0, 1, 2], None, [6, 6, 6], [0, 256, 0, 256, 0, 256])#finding histograms for each block  
                    hist1= hist.flatten()  #flatten the hist to one-dimensinal vector 
                    feature_vector += list(hist1)
                    w = w_chunk*b
                h = h_chunk*a
                w= 0

            color_histograms =np.vstack((color_histograms, feature_vector )) #appending each one-dimensinal vector to generate N*M matrix (where N is number of frames
            #and M is 1944) 
            frame_count+=1
        else:
            break

    print("--- %s seconds ---" % (time.time()))
    color_histograms = color_histograms.transpose() #transposing so that i will have all frames in columns i.e M*N dimensional matrix 
    print(color_histograms.shape)
    print(frame_count)
    
    histogram_matrix = csc_matrix(color_histograms, dtype=float)
    u, s, vt = svds(histogram_matrix, k = 63)
    vt = vt.transpose()
    projections = vt @ np.diag(s)
    return projections, frames

def dynamic_cluster(projections, frames):
    frame_in_cluster = dict() #to store frames in respective cluster
    for i in range(projections.shape[0]):
        frame_in_cluster[i] = np.empty((0,63), int)
        
    #adding first two projected frames in first cluster i.e Initializaton    
    frame_in_cluster[0] = np.vstack((frame_in_cluster[0], projections[0]))   
    frame_in_cluster[0] = np.vstack((frame_in_cluster[0], projections[1]))

    centroids_dict = dict() #to store centroids of each cluster
    for i in range(projections.shape[0]):
        centroids_dict[i] = np.empty((0,63), int)
        
    centroids_dict[0] = np.mean(frame_in_cluster[0], axis=0) #finding centroid of frame_in_cluster[0] cluster

    count = 0
    for i in range(2,projections.shape[0]):
        similarity = np.dot(projections[i], centroids_dict[count])/( (np.dot(projections[i],projections[i]) **.5) * (np.dot(centroids_dict[count], centroids_dict[count]) ** .5)) #cosine similarity
        #this metric is used to quantify how similar is one vector to other. The maximum value is 1 which indicates they are same
        #and if the value is 0 which indicates they are orthogonal nothing is common between them.
        #Here we want to find similarity between each projected frame and last cluster formed chronologically. 
        
        
        if similarity < 0.9: #if the projected frame and last cluster formed  are not similar upto 0.9 cosine value then 
                            #we assign this data point to newly created cluster and find centroid 
                            #We checked other thresholds also like 0.85, 0.875, 0.95, 0.98
                            #but 0.9 looks okay because as we go below then we get many key-frames for similar event and 
                            #as we go above we have lesser number of key-frames thus missed some events. So, 0.9 seems optimal.
                            
            count+=1         
            frame_in_cluster[count] = np.vstack((frame_in_cluster[count], projections[i])) 
            centroids_dict[count] = np.mean(frame_in_cluster[count], axis=0)   
        else:  #if they are similar then assign this data point to last cluster formed and update the centroid of the cluster
            frame_in_cluster[count] = np.vstack((frame_in_cluster[count], projections[i])) 
            centroids_dict[count] = np.mean(frame_in_cluster[count], axis=0)  
    b = []  #find the number of data points in each cluster formed.

    #We can assume that sparse clusters indicates 
    #transition between shots so we will ignore these frames which lies in such clusters and wherever the clusters are densely populated indicates they form shots
    #and we can take the last element of these shots to summarise that particular shot

    for i in range(projections.shape[0]):
        b.append(frame_in_cluster[i].shape[0])

    last = b.index(0)  #where we find 0 in b indicates that all required clusters have been formed , so we can delete these from frame_in_cluster
    b1=b[:last ] #The size of each cluster. 
    res = [idx for idx, val in enumerate(b1) if val >= 25] #so i am assuming any dense cluster with atleast 25 frames is eligible to 
    GG = frame_in_cluster #copying the elements of frame_in_cluster to GG, the purpose of  the below code is to label each cluster so later 
    #it would be easier to identify frames in each cluster
    for i in range(last):
        p1= np.repeat(i, b1[i]).reshape(b1[i],1)
        GG[i] = np.hstack((GG[i],p1))
        
    F=  np.empty((0,64), int) 
    for i in range(last):
        F = np.vstack((F,GG[i]))

    colnames = []
    for i in range(1, 65):
        col_name = "v" + str(i)
        colnames+= [col_name]
    df = pd.DataFrame(F, columns= colnames)
    df['v64']= df['v64'].astype(int)  #converting the cluster level from float type to integer type
    df1 =  df[df.v64.isin(res)] 
    new = df1.groupby('v64').tail(1)['v64'] #For each cluster /group take its last element which summarize the shot i.e key-frame
    new1 = new.index #finding key-frames (frame number so that we can go back get the original picture)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    for c in new1:
        frame_rgb1 = cv2.cvtColor(frames[c], cv2.COLOR_RGB2BGR) #since cv consider image in BGR order
        frame_num_chr = str(c)
        file_name = 'frame'+ frame_num_chr +'.png'
        final_path = os.path.join(current_dir, "frames")
        file_name = os.path.join(final_path, file_name)
        cv2.imwrite(file_name, frame_rgb1)

def extract_features(video_name):
    projections, frames = get_histogram_features(video_name)
    dynamic_cluster(projections, frames)
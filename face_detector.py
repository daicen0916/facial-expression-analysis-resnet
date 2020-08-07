# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:27:25 2020

@author: daicen
"""

from PIL import Image
from facenet_pytorch import MTCNN
import glob
import os
import torchvision.transforms.functional as TF
import torch

def detect_face(mtcnn,img,tensor_path,grayscale=False):
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        print(tensor_path)
        croped_face = TF.center_crop(img, (224,224))
    else:
        box = boxes[0]
        left,top,right,bottom = int(box[0]),int(box[1]),int(box[2]),int(box[3])
        width = abs(left-right)
        height = abs(top-bottom)
        croped_face = TF.resized_crop(img, top, left, height, width,(224,224))
    if grayscale:
        croped_face = TF.to_grayscale(croped_face)
    face_tensor = TF.to_tensor(croped_face)
    if not grayscale:
        norm_tensor = TF.normalize(face_tensor, mean=(0.485, 0.456, 0.406), 
                                   std =  (0.229, 0.224, 0.225))
    else:
        norm_tensor = TF.normalize(face_tensor, 0.5,0.5)
    torch.save(norm_tensor,tensor_path+'.pt')
#%%
path_list = glob.glob('Subset For Assignment SFEW/*/*.*')
if not os.path.exists('face_images'):
    os.mkdir('face_images')
if not os.path.exists('grayscale_faces'):    
    os.mkdir('grayscale_faces')
    
emotion_list = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

for emotion in emotion_list:
    os.mkdir('face_images/'+emotion)
    os.mkdir('grayscale_faces/'+emotion)
    
mtcnn = MTCNN(post_process=False, device='cuda:0')
for path in path_list:
    tensor_path1 = 'face_images'+path[path.find('\\'):-4]
    tensor_path2 = 'grayscale_faces'+path[path.find('\\'):-4]
    img = Image.open(path)
    detect_face(mtcnn, img, tensor_path1)
    detect_face(mtcnn, img, tensor_path2,True)


import cv2
#from skimage.io import imread, imsave
#from skimage.transform import resize
#import numpy as np
import os
from os.path import isfile, join

#org_path = "/home/compu/ymh/drawing/dataset/num_003/frame/"
pathIn= '/data/ymh/drawing/dataset/num_006/image_pseudo'
#pathIn= '/home/compu/ymh/drawing/dataset/num_004/image_semi/'
pathOut = '/data/ymh/drawing/dataset/num_006/org.mp4'
#pathOut = "/home/compu/ymh/drawing/save/num_004/Drawing/joint/semi.mp4"
fps = 29.97

width, height = 1280, 720

frame_array = []

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort()

#files2 = [f for f in os.listdir(org_path) if isfile(join(org_path, f))]
#files2.sort()

#for i in range(1):
for i in range(len(files)):
    print(i)
    #filename2 = org_path + files[i]
    #index = int(files[i].split("_")[1].split(".")[0])
    filename = os.path.join(pathIn, files[i])
    #print(filename)
    #reading each files
    #org = cv2.imread(filename2)
    #org = cv2.resize(org, (240,426))
    #print(org)
    #org = org * 255
    #org = org.astype('uint8')
    
    img = cv2.imread(filename)
    img = cv2.resize(img, (width, height))#, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    #print(img)
    #img = img * 255
    #img = img.astype('uint8')
    #print(img.shape)
    #height, width, layers = img.shape
    #size = (width,height)
    
    #inserting the frames into an image array
    
    #print(org.shape, img.shape)
    #frame = np.concatenate((org, img), axis=1)
    #frame = (org - img)
    #frame = np.abs(frame)
    #print(frame.shape)
    
    frame_array.append(img)
    #frame_array.append(frame)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'FMP4'), fps, (width, height))

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
    print(i)

out.release()

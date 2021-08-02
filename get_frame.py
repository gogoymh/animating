import cv2
import os

read_path = "/home/compu/ymh/drawing/dataset/num_006/yerin.mp4"
save_path = "/home/compu/ymh/drawing/dataset/num_006/yerin_frame/"

vidcap = cv2.VideoCapture(read_path)
success, image = vidcap.read()
count = 0
while success:
  cv2.imwrite(os.path.join(save_path, "frame_%07d.png" % count), image)
  print("Frame %d is saved." % count)
  success, image = vidcap.read()
  #print('Read a new frame: ', success)
  count += 1
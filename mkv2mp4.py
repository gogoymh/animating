import cv2
import os

read_path = "/home/compu/ymh/drawing/save/num_003/Drawing/character/refine.mkv"
save_path = "/home/compu/ymh/drawing/save/num_003/Drawing/character/refine.mp4"

fps = 30

vidcap = cv2.VideoCapture(read_path)
success, image = vidcap.read()
count = 0

frame_array = []

cnt = 0
while success:
    frame_array.append(image)
    
    success, image = vidcap.read()
    print('working: %d' % cnt)
    cnt += 1
    
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'FMP4'), fps, (1920, 1080))

cnt = 0
for i in range(len(frame_array)):
    out.write(frame_array[i])
    print('working: %d' % cnt)
    cnt += 1

out.release()

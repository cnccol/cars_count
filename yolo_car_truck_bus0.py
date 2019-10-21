from __future__ import division
import os
import pickle
import time
import datetime
import numpy as np
import pandas as pd 
import cv2
import imutils
import torch 
import torch.nn as nn
from torch.autograd import Variable
from util import load_classes, write_results
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pdb

# working directory helping variable
cwd = os.getcwd()
font = cv2.FONT_HERSHEY_DUPLEX

# params
confidence = 0.7 #yolo
nms_thesh = 0.4 #yolo
yolo_input_height = str(320) #yolo
min_area = 24000 #motion
delta_threshold = 12 #motion

############################################################
##### FOR YOLO #############################################
############################################################

CUDA = torch.cuda.is_available()
num_classes = 80
# bbox_attrs = 5 + num_classes
print("Loading network ...")
model = Darknet(cwd + "/cfg/yolov3.cfg")
model.load_weights(cwd + "/yolov3.weights")
classes = load_classes(cwd + '/data/coco.names')
print("Network successfully loaded")
model.net_info["height"] = yolo_input_height
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32
if CUDA:
    model.cuda()
    
############################################################
############################################################
############################################################

# initialize cv2 video capture objects and warm-up.
cap = cv2.VideoCapture('videos/test_video.avi')
time.sleep(2.0)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
F_AREA = W*H
print('W:', W)
print('H:', H)
print('FPS:', FPS)

# main
frame_number = 1
prev_frame = None
last_veh_thresh = 70

# results dictionary structure.
results = {"frame_number":[], "vehicle_type":[], "rects":[]}
last_veh_dict = {'car': -last_veh_thresh, 'bus': -last_veh_thresh, 
                 'truck': -last_veh_thresh}
counts_dict = {'car': 0, 'bus': 0, 'truck': 0}

while True:

    ret, frame = cap.read()
    if not ret:
        break

    ############################################################
    ##### FOR YOLO #############################################
    ############################################################
    orig_im = frame.copy()
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    im_dim = torch.FloatTensor(dim).repeat(1,2)
    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()
    t2 = time.time()    
    with torch.no_grad():   
        output = model(Variable(img), CUDA)
    print("YOLO forward time:", time.time() - t2)    
    output = write_results(output, confidence, num_classes, nms = True, 
                        nms_conf = nms_thesh)
    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor
    for ii in range(output.shape[0]):
        output[ii, [1,3]] = torch.clamp(output[ii, [1,3]], 0.0, im_dim[ii,0])
        output[ii, [2,4]] = torch.clamp(output[ii, [2,4]], 0.0, im_dim[ii,1])

    for thing in output:
        cls = int(thing[-1])
        label = "{0}".format(classes[cls])
        bbox = thing[1:5].int()
        
        if ((label == "car") or (label == "bus") or (label == "truck")
            ) and (bbox.prod().item() != 0):
            vleft, vtop, vright, vbottom = bbox
            vleft, vtop, vright, vbottom = (vleft.item(), vtop.item(), 
                                            vright.item(), vbottom.item())
            area = (vright-vleft) * (vbottom-vtop)
            if area > (F_AREA*0.33):
                cv2.rectangle(frame, (vleft, vtop), (vright, vbottom), 
                            (0,100,0), 2, 8)
                cv2.putText(frame, label, (vleft, vbottom), font, 2.0, (0, 0, 255), 1)
                results["frame_number"].append(frame_number)
                results["vehicle_type"].append(label)
                results["rects"].append((vleft, vtop, vright, vbottom))

                if frame_number - last_veh_dict[label] >= last_veh_thresh:
                    counts_dict[label] += 1
                    cv2.imwrite('imgs/' + label + '/' + str(counts_dict[label]) + '.jpg',
                                frame)
                last_veh_dict[label] = frame_number

    ############################################################
    ############################################################
    ############################################################
    
    frame_number += 1
        
cap.release()
cv2.destroyAllWindows()

df1 = pd.DataFrame([counts_dict])
df2 = pd.DataFrame(results)

df1.to_csv("counts.csv")
df2.to_csv("results.csv")
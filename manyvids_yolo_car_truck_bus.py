from __future__ import division
import os
import pathlib
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

# params
confidence = 0.6 #yolo
nms_thesh = 0.4 #yolo
yolo_input_height = str(480) #yolo
last_veh_thresh = 36
min_passing_veh_obs = 7
min_area_as_frame_fraction = 0.12

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
for input_video in os.listdir(cwd+"/videos/converted/"):
    input_video = input_video[:-4]
    pathlib.Path(cwd+'/imgs/'+input_video+'/car').mkdir(parents=True, exist_ok=True)
    pathlib.Path(cwd+'/imgs/'+input_video+'/bus').mkdir(parents=True, exist_ok=True)
    pathlib.Path(cwd+'/imgs/'+input_video+'/truck').mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture('videos/converted/'+input_video+'.avi')

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    FRAME_AREA = W*H
    print('W:', W)
    print('H:', H)
    print('FPS:', FPS)

    # main
    frame_number = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    # results dictionary structure.
    results = {"frame_number":[], "vehicle_type":[], "rect_area":[]}
    last_veh_frame = -last_veh_thresh
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
        with torch.no_grad():   
            output = model(Variable(img), CUDA)
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
                if area > (FRAME_AREA*min_area_as_frame_fraction):
                    cv2.rectangle(frame, (vleft, vtop), (vright, vbottom), 
                                (0,100,0), 2, 8)
                    cv2.putText(frame, label, (vleft, vbottom), font, 2.0, (0, 0, 255), 1)

                    if frame_number - last_veh_frame >= last_veh_thresh:
                        analysing_passing_veh = True
                        passing_veh_labels = [label]
                    else:
                        if analysing_passing_veh:
                            if len(passing_veh_labels) >= min_passing_veh_obs:
                                labels_mode = max(passing_veh_labels, key=passing_veh_labels.count)
                                counts_dict[labels_mode] += 1
                                cv2.imwrite(
                                    'imgs/'+input_video+'/'+labels_mode+'/'+str(frame_number)+'.jpg',
                                    frame)
                                results["frame_number"].append(frame_number)
                                results["vehicle_type"].append(labels_mode)
                                results["rect_area"].append(area)
                                analysing_passing_veh = False
                            else:
                                passing_veh_labels.append(label) 
                    last_veh_frame = frame_number

        ############################################################
        ############################################################
        ############################################################
        
        counts_text = str(counts_dict)
        cv2.putText(frame, counts_text, (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        # small_frame = imutils.resize(frame, height=480)

        # cv2.imshow('procesado', small_frame)    
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break

        frame_number += 1
            
    cap.release()
    # cv2.destroyAllWindows()

    df1 = pd.DataFrame([counts_dict])
    df2 = pd.DataFrame(results)

    df1.to_csv(cwd+"csvs/counts/"+input_video+"_counts.csv")
    df2.to_csv(cwd+"csvs/results/"input_video+"_results.csv")
    print('Done with', input_video)

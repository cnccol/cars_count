from __future__ import division
import os
import re
import pathlib
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
from similarity.normalized_levenshtein import NormalizedLevenshtein
from openalpr import Alpr

cwd = os.getcwd() #global

see = True
write_video = False

def create_dirs(input_video_, cwd_=cwd):
    pathlib.Path(cwd_+'/imgs/'+input_video_+'/car').mkdir(parents=True, exist_ok=True)
    pathlib.Path(cwd_+'/imgs/'+input_video_+'/bus').mkdir(parents=True, exist_ok=True)
    pathlib.Path(cwd_+'/imgs/'+input_video_+'/truck').mkdir(parents=True, exist_ok=True)
    pathlib.Path(cwd_+'/output/videos').mkdir(parents=True, exist_ok=True)
    pathlib.Path(cwd_+'/output/csvs').mkdir(parents=True, exist_ok=True)

def cap_properties(cap_):
    W_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS_ = int(cap.get(cv2.CAP_PROP_FPS))
    FRAME_AREA_ = W_*H_
    return W_, H_, FPS_, FRAME_AREA_

def draw_plates(frame_, plate_, draw=False):
    if not draw:
        return None
    cv2.line(frame_,
             (plate_["coordinates"][0]["x"], plate_["coordinates"][0]["y"]),
             (plate_["coordinates"][1]["x"], plate_["coordinates"][1]["y"]),
             (255,0,0), 5)
    cv2.line(frame_,
             (plate_["coordinates"][1]["x"], plate_["coordinates"][1]["y"]),
             (plate_["coordinates"][2]["x"], plate_["coordinates"][2]["y"]),
             (255,0,0), 5)
    cv2.line(frame_,
             (plate_["coordinates"][2]["x"], plate_["coordinates"][2]["y"]),
             (plate_["coordinates"][3]["x"], plate_["coordinates"][3]["y"]),
             (255,0,0), 5)
    cv2.line(frame_,
             (plate_["coordinates"][3]["x"], plate_["coordinates"][3]["y"]),
             (plate_["coordinates"][0]["x"], plate_["coordinates"][0]["y"]),
             (255,0,0), 5)
    

# input objects
config_path = "/etc/openalpr/openalpr.conf" #alpr
runtime_data_path = "/usr/share/openalpr/runtime_data" #alpr
country = "us" #alpr
fourcc = cv2.VideoWriter_fourcc(*"XVID") #cv2
if write_video:
    out = cv2.VideoWriter(cwd_+"/output/videos/processed.avi", fourcc, FPS, (W, H))

# params to find
confidence = 0.65 #yolo
nms_thesh = 0.4 #yolo
yolo_input_height = str(480) #yolo

# params objects and variables
frame_fraction = 0.08 #area multiplier
font = cv2.FONT_HERSHEY_DUPLEX #font for cv2

# results dictionary structure
results = {"frame_number":[], "yolo_rect_area":[], "vehicle_type":[],
           "plate": [], "plate_confidence":[]}

############################################################
##### LOADING YOLO #########################################
############################################################

CUDA = torch.cuda.is_available()
num_classes = 80
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


alpr = None
try:
    alpr = Alpr(country, config_path, runtime_data_path)
    if not alpr.is_loaded():
        print("--- Error loading OpenAlpr")
        sys.exit(1)
    else:
        print("Using OpenAlpr "+alpr.get_version())
        alpr.set_top_n(7)
except:
    None
        
#total_frame_number = 1
input_video = "NVR_ch8_main_20190702060002_20190702060500"
# creating some dirs
create_dirs(input_video)
# cap and its properties
cap = cv2.VideoCapture(cwd+"/videos/converted/"+input_video+".avi")
W, H, FPS, FRAME_AREA = cap_properties(cap)
print("Processing "+input_video)

### main ###
frame_number = 1
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
    ############################################################
    ############################################################
    ############################################################
    
    for thing in output:
        cls = int(thing[-1])
        label = "{0}".format(classes[cls])
        bbox = thing[1:5].int()
        
        if ((label == "car") or (label == "bus") or (label == "truck")
            ) and (bbox.prod().item() != 0):
            vleft, vtop, vright, vbottom = bbox
            vleft, vtop, vright, vbottom = (vleft.item(), vtop.item(), 
                                            vright.item(), vbottom.item())
            
            yolo_area = (vright-vleft) * (vbottom-vtop) # area of the detection bbox
        
            if yolo_area > (FRAME_AREA*frame_fraction):
                
                results["frame_number"].append(frame_number)
                results["yolo_rect_area"].append(yolo_area)
                results["vehicle_type"].append(label)
                #results["total_frame_number"].append(total_frame_number)
                
                cv2.rectangle(frame, (vleft-20, vtop-20), (vright+20, vbottom+20), 
                              (0,100,0), 2, 8)
                cv2.putText(frame, label, (vleft, vbottom), font, 1.0, (0,100,0), 1)
                
                alpr_results = alpr.recognize_ndarray(frame[max(vtop-20,0):min(vbottom+20,frame.shape[0]), 
                                                            max(vleft-20,0):min(vright+20,frame.shape[1])])
                
                if alpr_results["results"]:
                    plate = alpr_results["results"][0]["plate"]
                    plate = plate[:3].replace("0","O") + plate[3:]
                    plate_confidence = alpr_results["results"][0]["confidence"]
                    # print(plate, plate_confidence)
                
                    if re.match(r"^[A-Z]{3}[0-9]{3}$", plate):
                        results["plate"].append(plate)
                        results["plate_confidence"].append(plate_confidence)
                        cv2.putText(frame, "placa: "+plate, (15, 10),
                                    font, 1.0, (1, 1, 255), 1)
                    else:
                        plate = np.nan
                        plate_confidence = np.nan
                        results["plate"].append(plate)
                        results["plate_confidence"].append(plate_confidence)
                else:
                    plate = np.nan
                    plate_confidence = np.nan
                    results["plate"].append(plate)
                    results["plate_confidence"].append(plate_confidence)
                    
                
    cv2.putText(frame, str(frame_number), (15,60), font, 1.0, (255,255,255), 1)
    
    if write_video:
        out.write(frame)
    if see:
        small_frame = imutils.resize(frame, height=480)
        cv2.imshow("processed", small_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    frame_number += 1
    # total_frame_number += 1
df = pd.DataFrame(results)
df.to_csv(cwd+"/output/csvs/r_"+input_video+".csv")
print("DONE "+input_video)
    
if alpr:
    alpr.unload()
    
cap.release()
cv2.destroyAllWindows()

if write_video:
    out.release()
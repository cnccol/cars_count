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

see = True
draw = False
write_video = False
save = False
use_alpr = False
debug = False
write_imgs = False

cwd = os.getcwd() #global
peaje = "Koran"

#############################################################################
def create_dirs(input_video_, cwd_=cwd, peaje_=peaje):
    pathlib.Path(cwd_+'/output/imgs/'+peaje_+"/"+input_video_).mkdir(parents=True, 
                                                                    exist_ok=True)
    pathlib.Path(cwd_+'/output/csvs/'+peaje_).mkdir(parents=True, 
                                                   exist_ok=True)
    pathlib.Path(cwd_+'/output/videos/'+peaje_).mkdir(parents=True, 
                                                     exist_ok=True)

def cap_properties(cap_):
    W_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS_ = int(cap.get(cv2.CAP_PROP_FPS))
    FRAME_AREA_ = W_*H_
    return W_, H_, FPS_, FRAME_AREA_

def draw_plates(frame_, plate_, draw=False):
    if not draw:
        None
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

def find_yolo_areas(output_):
    yolo_areas_ = []
    for thing in output_:
        bbox = thing[1:5].int()
        vleft, vtop, vright, vbottom = bbox
        vleft, vtop, vright, vbottom = (vleft.item(), vtop.item(), 
                                        vright.item(), vbottom.item())
        yolo_area_ = (vright-vleft) * (vbottom-vtop)
        yolo_areas_.append(yolo_area_)
    largest_area_index_ = yolo_areas_.index(max(yolo_areas_))
    return yolo_areas_, largest_area_index_

#############################################################################


# params to find
confidence = 0.60 #yolo
nms_thesh = 0.4 #yolo
yolo_input_height = str(480) #yolo

# input objects
config_path = "/etc/openalpr/openalpr.conf" #alpr
runtime_data_path = "/usr/share/openalpr/runtime_data" #alpr
country = "us" #alpr

# load openalpr
alpr = None
if use_alpr:
    try:
        alpr = Alpr(country, config_path, runtime_data_path)
        if not alpr.is_loaded():
            print("--- Error loading OpenAlpr")
            sys.exit(1)
        else:
            print("--- Using OpenAlpr "+alpr.get_version())
            alpr.set_top_n(7)
    except:
        None
else:
    print("--- Will not use OpenAlpr")

# for cv2 writer
fourcc = cv2.VideoWriter_fourcc(*"XVID") #cv2
#fourcc = cv2.VideoWriter_fourcc(*"H264") #cv2

if write_video:
    out = cv2.VideoWriter(cwd_+"/output/videos/processed.avi", fourcc, FPS, (W, H))

# params objects and variables
frame_fraction = 0.05 #area multiplier
font = cv2.FONT_HERSHEY_DUPLEX #font for cv2

# results

# load yolo
CUDA = torch.cuda.is_available()
num_classes = 80
print("--- Loading YOLO network ...")
model = Darknet(cwd + "/cfg/yolov3.cfg")
model.load_weights(cwd + "/yolov3.weights")
classes = load_classes(cwd + '/data/coco.names')
print("--- YOLO network successfully loaded")
model.net_info["height"] = yolo_input_height
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32
if CUDA:
    model.cuda()
        
# list of videos to process
video_list = sorted(os.listdir(cwd+"/videos/converted/"+peaje))
begin, end = 4, 8
print("    ")
print("--- Will process "+str(len(video_list[begin:end]))+" videos")

for input_video in video_list[begin:end]:
    print("  - "+input_video)
    
for input_video in video_list[begin:end]:
    tic = time.time()

    # results dictionary structure
    results = {"frame_number":[], "vehicle_type":[], "yolo_rect_area":[], 
               "yolo_centroid":[], "b_mean":[], "g_mean":[], "r_mean":[], 
               "yolo_confidence":[], "plate":[], "plate_confidence":[]}

    # video input format
    vid_format = input_video[-4:]
    
    # video input name
    input_video = input_video[:-4]

    # creating some dirs
    create_dirs(input_video)

    # cap and its properties
    cap = cv2.VideoCapture(cwd+"/videos/converted/"+peaje+"/"+input_video+vid_format)
    W, H, FPS, FRAME_AREA = cap_properties(cap)
    print("        --- VIDEO: "+input_video+vid_format)
    print("          --- FPS: "+str(FPS))
    print("          --- W: "+str(W))
    print("          --- H: "+str(H))

    ######## main ########
    H_resize = 720
    r = H/H_resize
    
    frame_number = 1
    while True:
        
        ret, frame = cap.read()

        if not ret:
            break

        if draw:
            cv2.putText(frame, str(frame_number), (15,60), font, 1.0, 
                        (255,255,255), 1)

        # for region of interest (roi)
        xroi_m = int(0)
        xroi_M = int(1000) #frame.shape[1]
        yroi_m = int(0)
        yroi_M = int(frame.shape[0]) #frame.shape[0]
        roi_frame = frame[yroi_m:yroi_M, xroi_m:xroi_M]
        if draw:
            cv2.rectangle(frame, (xroi_m, yroi_m), (xroi_M, yroi_M), 
                          (0,0,100), 2, 8)
            if debug:
                cv2.imshow("roi", roi_frame)

        # pass through yolo
        orig_im = roi_frame.copy()
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
        
        yolo_areas, lai = find_yolo_areas(output)
        
        if yolo_areas[lai]>(FRAME_AREA*frame_fraction):
            largest_thing = output[lai]
            bbox = largest_thing[1:5].int()
            cls = int(largest_thing[-1])
            label = "{0}".format(classes[cls])
            
            if ((label == "car") or (label == "bus") or (label == "truck")
                ) and (bbox.prod().item() != 0):
                results["frame_number"].append(frame_number)
                results["vehicle_type"].append(label)
                vleft, vtop, vright, vbottom = bbox
                vleft, vtop, vright, vbottom = (vleft.item()+xroi_m, vtop.item()+yroi_m, 
                                                vright.item()+xroi_m, vbottom.item()+yroi_m)
                if draw:
                    cv2.rectangle(frame, (vleft, vtop), (vright, vbottom), 
                                (0,100,0), 2, 8)
                    cv2.putText(frame, label, (vleft,vbottom), font, 1, 
                                (255,0,255), 1)
                centroid = ((vleft+vright)/2, (vtop+vbottom)/2)
                results["yolo_centroid"].append(centroid)
                yolo_area = yolo_areas[lai]
                results["yolo_rect_area"].append(yolo_area)
                yolo_confidence = largest_thing[6].item()
                results["yolo_confidence"].append(yolo_confidence)
                b_mean, g_mean, r_mean = frame[vtop:vbottom, vleft:vright,
                                            :].mean(axis=0).mean(axis=0)
                results["b_mean"].append(b_mean)
                results["g_mean"].append(g_mean)
                results["r_mean"].append(r_mean)
            
                if write_imgs:
                    small_frame_w = imutils.resize(frame, height=H_resize)
                    cv2.imwrite(cwd+"/output/imgs/"+peaje+"/"+input_video+"/"+str(frame_number)+".jpg",
                                small_frame_w)
                
                if use_alpr:
                    alpr_results = alpr.recognize_ndarray(frame[max(vtop-20,0):min(vbottom+20,frame.shape[0]), 
                                                                max(vleft-20,0):min(vright+20,frame.shape[1])]
                                                         )
                    if debug:
                        cv2.imshow("alpr_frame", frame[max(vtop-20,0):min(vbottom+20,frame.shape[0]), 
                                                       max(vleft-20,0):min(vright+20,frame.shape[1])])
                    if alpr_results["results"]:
                        plate = alpr_results["results"][0]["plate"]
                        plate = plate[:3].replace("0","O") + plate[3:]
                        plate_confidence = alpr_results["results"][0]["confidence"]
                        if re.match(r"^[A-Z]{3}[0-9]{3}$", plate):
                            if draw:
                                cv2.putText(frame, "placa: "+plate, (3, 100), font, 1.0, 
                                            (1, 1, 255), 1)
                            results["plate"].append(plate)
                            results["plate_confidence"].append(plate_confidence)
                        else:
                            plate = ""
                            plate_confidence = np.nan
                            results["plate"].append(plate)
                            results["plate_confidence"].append(plate_confidence)
                    else:
                        plate = ""
                        plate_confidence = np.nan
                        results["plate"].append(plate)
                        results["plate_confidence"].append(plate_confidence)
                        
                else:
                    results["plate"].append("")
                    results["plate_confidence"].append(np.nan)
                                
        if see:
            small_frame = imutils.resize(frame, height=H_resize)
            cv2.imshow("processed", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
        frame_number += 1
        
        if write_video:
            out.write(frame)
            
    ######## end ########

    if save:
        df = pd.DataFrame(results)
        df.to_csv(cwd+"/output/csvs/"+peaje+"/"+"r_"+input_video+".csv")
        print("        --- CSV SAVED")

    cap.release()
    cv2.destroyAllWindows()
    
    print("          --- DONE: "+input_video)
    print("          --- Processing took: "+str(time.time()-tic))
    
if write_video:
    out.release()

if use_alpr:
    if alpr:
        alpr.unload()
        print("--- OPENALPR UNLOADED" )

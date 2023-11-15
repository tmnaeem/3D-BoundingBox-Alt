import os
import cv2 
import time
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from time import process_time
import subprocess
import sys

# 3D bounding box import
from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, Model_mobilenet, Model_mobilenetv3, ClassAverages
from yolo.yolo import cv_Yolo
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg, mobilenet_v2, mobilenet_v3_small
# 3D bounding box import

# Download YOLOv7 code
path = "https://github.com/WongKinYiu/yolov7"
subprocess.run(["git", "clone", path])
os.chdir("yolov7")
sys.path.append(os.getcwd())
from pathlib import Path

from models.yolo import Model as md
from utils.general import check_requirements, set_logging
from utils.google_utils import attempt_download
from utils.torch_utils import select_device

def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def custom(path_or_model='path/to/model.pt', autoshape=True):
    """custom mode

    Arguments (3 options):
        path_or_model (str): 'path/to/model.pt'
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

    Returns:
        pytorch model
    """
    model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model

    hub_model = md(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    if autoshape:
        hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
    return hub_model.to(device)

def process_frame(img, model, device):
    """
    get_prediction
    parameters:
      - img_path - path of the input image
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
    
    """
    
    img_resized = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
            pred = model(img_resized)
            return pred
    
 # load torch
weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
if len(model_lst) == 0:
    print('No previous model found, please train first!')
    exit()
else:
    print('Using previous model %s'%model_lst[-1])
    # TODO: load bins from file or something
    # my_vgg = vgg.vgg19_bn(pretrained=True)
    # model = Model.Model(features=my_vgg.features, bins=2).cuda()
    # my_mobilenet = mobilenet_v2(pretrained=True)
    # model = Model_mobilenet.Model(features=my_mobilenet.features, bins=2).cuda()
    my_mobilenet = mobilenet_v3_small(pretrained=True)
    model = Model_mobilenetv3.Model(features=my_mobilenet.features, bins=2).cuda()
    checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

calib_file = "Z:/education/pytorch_training/3D-BoundingBox-bak/eval/video/2011_09_26/calib_cam_to_cam.txt"
averages = ClassAverages.ClassAverages()
# TODO: clean up how this is done. flag?
angle_bins = generate_bins(2)

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
modelv7 = custom(path_or_model=r'Z:\education\pytorch_training\yolov7-tiny.pt')  # custom example

if modelv7:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelv7.to(device)
    modelv7.eval()

video_source = cv2.VideoCapture('Z:/education/pytorch_training/3dbbtest.mp4')
threshold = 0.5
rect_th = 3
text_size = 3
text_th = 3
ori_scale_w = 320
ori_scale_h = 320

if not video_source.isOpened():
    print("Error opening video stream or file")

while video_source.isOpened():
    start_time = time.time()
    ret, frame = video_source.read()

    if not ret:
        break

    original_h, original_w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (ori_scale_w, ori_scale_h))
    scale_w, scale_h = original_w / ori_scale_w, original_h / ori_scale_h

    pred = process_frame(frame_resized, modelv7, device)

    if pred:
        pd_results = pred.pandas().xyxy[0]            
        labels, boxes, scores = pd_results['name'].to_numpy(), pd_results[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy(), pd_results['confidence'].to_numpy()

        for label, box, score in zip(labels, boxes, scores):
            if score < threshold:
                continue

            if not averages.recognized_class(label):
                continue

            detection_box_2d = [(int(box[0] * scale_w), int(box[1]* scale_h)), (int(box[2] * scale_w), int(box[3]* scale_h))]
            try:
                detectedObject = DetectedObject(frame, label, detection_box_2d, calib_file)
            except:
                continue
            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = detection_box_2d
            detected_class = label

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            start_time_3D = time.time()

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            location = plot_regressed_3d_bbox(frame, proj_matrix, box_2d, dim, alpha, theta_ray)
            print("3D_bounding generation: ", (time.time() - start_time_3D))
    #         xy = (int(box[0] * scale_w), int(box[1] * scale_h))
    #         x2y2 = (int(box[2] * scale_w), int(box[3] * scale_h))
    #         cv2.rectangle(frame, xy, x2y2, color=(0, 255, 0), thickness=rect_th)
    #         cv2.putText(frame, label, xy, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(-1)

    print("FPS: ", 1.0 / (time.time() - start_time))

video_source.release()
cv2.destroyAllWindows()
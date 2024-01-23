import tensorrt as trt

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
from line_profiler import LineProfiler

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

import argparse


parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--weight-type", default="vgg-19",
                    help="Specify the model weight type, default to VGG")

parser.add_argument("--trained-weight-path", required=True,
                    help="Provide the path to the model's weight (required)")

parser.add_argument("--default-objDec-model", default="Z:/education/pytorch_training/yolov7-tiny.pt",
                    help="Provide the path to object detection model, default to yolov7-tiny")

parser.add_argument("--video-source", default="Z:/education/pytorch_training/3dbbtest.mp4",
                    help="Provide the path to video source")

parser.add_argument("--to-format", default="onnx",
                    help="Provide the format you want to convert")

def convert_to_tensorrt():

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create a builder
    builder = trt.Builder(TRT_LOGGER)

    # Create a network definition with the explicit batch flag
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    # Create a parser to load the ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Load the ONNX file
    onnx_file_path = "Z:/education/pytorch_training/3D-BoundingBox-alt/to3Dmodel.onnx"
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    # Define optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name  # Adjust this according to your model

    # Set the dynamic range for the first dimension (vary)
    # Here, I'm assuming a range from 1 to a max batch size, e.g., 32
    # Adjust the range according to your requirements
    min_shape = (1, 3, 224, 224)
    opt_shape = (16, 3, 224, 224)    # Example optimum shape
    max_shape = (32, 3, 224, 224)    # Example maximum shape

    profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)

    # Specify other builder configurations
    # builder.max_batch_size = max_shape[0]  # Set max batch size to the maximum of your dynamic range
    config.max_workspace_size = 1 << 30  # 1GB

    # Engine building
    engine = builder.build_engine(network, config)
    print(engine)
    # Serialize the engine to a file
    trt_engine_path = "Z:/education/pytorch_training/3D-BoundingBox-alt/to3Dmodeld.trt"
    # Build the TensorRT engine
    with builder.build_engine(network, config) as engine, open(trt_engine_path, 'wb') as f:
        f.write(engine.serialize())

        return True

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

def process_frame(img, model):
    """
    get_prediction
    parameters:
      - img_path - path of the input image
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
    
    """
    
    with torch.no_grad():
        pred = model(img)
        return pred
    
def main_loop(frame, model, odModel, dimension_map_keys, threshold = 0.5, ori_scale_w = 320, ori_scale_h = 320):
    original_h, original_w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (ori_scale_w, ori_scale_h))
    scale_w, scale_h = original_w / ori_scale_w, original_h / ori_scale_h

    pred = process_frame(frame_resized, odModel)

    df_np = pred.pandas().xyxy[0].to_numpy()  

    # Now we will filter the numpy array based on the confidence and whether the name is in the 'dimension_map_keys'
    name_filter = np.isin(df_np[:, 6], dimension_map_keys)
    confidence_filter = df_np[:, 4] > threshold
    combined_filter = name_filter & confidence_filter

    # Apply the filters to the numpy array
    np_results_filtered = df_np[combined_filter]
    if len(np_results_filtered) == 0: return False

    # Apply scaling and convert to integer using numpy
    detection_box_2d_filtered = np.column_stack((
        (np_results_filtered[:, 0] * scale_w).astype(int),
        (np_results_filtered[:, 1] * scale_h).astype(int),
        (np_results_filtered[:, 2] * scale_w).astype(int),
        (np_results_filtered[:, 3] * scale_h).astype(int)
    ))  

    processed_crops = DetectedObject.batch_format_imgs(frame, detection_box_2d_filtered)
    # Move the batch tensor to the GPU if using CUDA
    input_tensor = torch.randn(1, 3, 224, 224).to('cuda')  # Make sure to use .to('cuda') only if you are using a GPU

    # Specify dynamic axes for batch size (the first dimension)
    dynamic_axes = {
        'input': {0: 'batch_size'},  # The batch size dimension is dynamic
        'output': {0: 'batch_size'}
    }

    # Export the model to ONNX
    # Ensure 'input_tensor' is a dummy input that reflects the shape and type of your actual model input
    torch.onnx.export(
        model, 
        input_tensor, 
        "Z:/education/pytorch_training/3D-BoundingBox-alt/to3Dmodel.onnx", 
        export_params=True, 
        opset_version=11,  # Use an opset version that is compatible with your model and TensorRT
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    return True

def main():
    # get parser
    FLAGS = parser.parse_args()
    weight_type = FLAGS.weight_type
    trained_weight_path = FLAGS.trained_weight_path
    default_objDec_model = FLAGS.default_objDec_model
    video_source = FLAGS.video_source
    to_format = FLAGS.to_format

    print(f"Using {weight_type}")
    if weight_type == "mobilenet_v2":
        my_mobilenet = mobilenet_v2(pretrained=True)
        model = Model_mobilenet.Model(features=my_mobilenet.features, bins=2).cuda()
    elif weight_type == "mobilenet_v3":
        my_mobilenet = mobilenet_v3_small(pretrained=True)
        model = Model_mobilenetv3.Model(features=my_mobilenet.features, bins=2).cuda()
    else:
        # TODO: load bins from file or something
        my_vgg = vgg.vgg19_bn(pretrained=True)
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
    
    checkpoint = torch.load(trained_weight_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    odModel = custom(path_or_model=default_objDec_model)  # custom example
    if odModel:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        odModel.to(device)
        odModel.eval()
    else :
        print("No object detection model")
        exit()
        
    video_source = cv2.VideoCapture(video_source)

    # The provided 'averages.dimension_map' is a dictionary with nested arrays.
    # We will need to extract the keys (names) to use them for filtering the 'name' column.
    # Extract the keys (names) for filtering
    averages = ClassAverages.ClassAverages()
    dimension_map_keys = list(averages.dimension_map.keys())

    if not video_source.isOpened() :
        print("Error opening video stream or file")
    else:
        breakLoop = False
        while video_source.isOpened():
            ret, frame = video_source.read()

            if not ret:
                break

            # Run the main loop
            convertToTensorrt = to_format == 'tensorrt'
            if convertToTensorrt:
                breakLoop = convert_to_tensorrt()
            else:
                main_loop(frame, model, odModel, dimension_map_keys)

            if breakLoop:
                print("File should be generated already, proceed exit")
                break

if __name__ == '__main__':
    main()
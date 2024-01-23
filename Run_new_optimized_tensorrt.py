# TensorRT and CUDA imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime as ort

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

from models.yolo import Model as md
from utils.general import check_requirements, set_logging
from utils.google_utils import attempt_download
from utils.torch_utils import select_device
# Initialize LineProfiler
lp = LineProfiler()

import argparse


parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--use-format", default="onnx",
                    help="Provide the format you want to use")

# @lp
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
    
    with torch.no_grad():
        pred = model(img)
        return pred
    
# Load TensorRT Engine Function
def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def run_trt_inference(engine, input_data):
    # Create an execution context
    context = engine.create_execution_context()

    # Allocate buffers
    inputs, outputs, bindings, bindings_shape, stream = allocate_buffers(engine, input_data)
    
    # Ensure the input_tensor is on CPU before converting to numpy
    if input_data.is_cuda:
        input_data = input_data.cpu()

    np.copyto(inputs[0]['host'], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

    # Set the binding dimensions for the current batch size
    input_tensor_name = engine.get_tensor_name(0)  # Assuming the first binding is the input
    context.set_input_shape(input_tensor_name, input_data.shape)

    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from GPU
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    # Return the output
    new_output = []
    for output, shape in zip(outputs, bindings_shape[1:]):
        new_output.append(np.array(output['host']).reshape(shape))
    return tuple(new_output)
    
def allocate_buffers(engine, input_data):
    inputs, outputs, bindings, bindings_shape, stream = [], [], [], [], cuda.Stream()
    for binding_index in range(engine.num_bindings):
        # Use get_tensor_name instead of get_binding_name
        tensor_name = engine.get_tensor_name(binding_index)
        binding_shape = engine.get_tensor_shape(tensor_name)

        if binding_shape[0] == -1:  # Dynamic dimension
            binding_shape[0] = input_data.shape[0]  # Set dynamic dimension to the actual batch size

        size = trt.volume(binding_shape) * 1
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        bindings_shape.append(binding_shape)

        # Adjust the condition based on the correct method to check if it's an input tensor
        if str(engine.get_tensor_mode(tensor_name)) == 'TensorIOMode.INPUT':  # Replace with correct condition
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    return inputs, outputs, bindings, bindings_shape, stream


# Run Inference Using TensorRT Engine
def run_onnx_inference(engine, input_tensor):
    outputs = engine.run(None, {'input': input_tensor.cpu().numpy()})
    # Convert each numpy array in outputs to a PyTorch tensor and move it to GPU
    gpu_outputs = [torch.tensor(output).to('cuda') for output in outputs]

    return gpu_outputs
       
# The main loop to be profiled
# @lp
def main_loop(frame, averages, proj_matrix, model_engine, odModel, angle_bins, dimension_map_keys, use_format, threshold = 0.5, ori_scale_w = 320, ori_scale_h = 320):
    original_h, original_w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (ori_scale_w, ori_scale_h))
    scale_w, scale_h = original_w / ori_scale_w, original_h / ori_scale_h

    pred = process_frame(frame_resized, odModel, 'cuda')

    df_np = pred.pandas().xyxy[0].to_numpy()  

    # Now we will filter the numpy array based on the confidence and whether the name is in the 'dimension_map_keys'
    name_filter = np.isin(df_np[:, 6], dimension_map_keys)
    confidence_filter = df_np[:, 4] > threshold
    combined_filter = name_filter & confidence_filter

    # Apply the filters to the numpy array
    np_results_filtered = df_np[combined_filter]
    if len(np_results_filtered) == 0 : return

    # Apply scaling and convert to integer using numpy
    detection_box_2d_filtered = np.column_stack((
        (np_results_filtered[:, 0] * scale_w).astype(int),
        (np_results_filtered[:, 1] * scale_h).astype(int),
        (np_results_filtered[:, 2] * scale_w).astype(int),
        (np_results_filtered[:, 3] * scale_h).astype(int)
    ))  
    theta_ray = DetectedObject.vectorized_calc_theta_ray(frame.shape[1], detection_box_2d_filtered, proj_matrix)
    processed_crops = DetectedObject.batch_format_imgs(frame, detection_box_2d_filtered)
    # Move the batch tensor to the GPU if using CUDA
    input_tensor = processed_crops.to('cuda')  # Make sure to use .to('cuda') only if you are using a GPU

    # Feed the batch tensor into the model
    # The model should be designed to handle batch inputs
    [orient, conf, dim] = run_trt_inference(model_engine, input_tensor) if use_format == 'tensorrt' else run_onnx_inference(model_engine, input_tensor)
    
    if use_format == 'onnx':
        orient_np = orient.cpu().data.numpy()  # Shape will be (batch_size, ...)
        conf_np = conf.cpu().data.numpy()      # Shape will be (batch_size, ...)
        dim_np = dim.cpu().data.numpy()        # Shape will be (batch_size, ...)
    else:
        # Convert the entire batch of outputs from PyTorch tensors to NumPy arrays
        orient_np = orient
        conf_np = conf
        dim_np = dim 

    # Assuming 'detected_classes' is a list or array of class labels for each item in the batch
    # and 'dims' is a 2D NumPy array with the dimensions for each item in the batch
    average_dims = np.array([averages.get_item(cls) for cls in np_results_filtered[:, 6]])

    # Now 'average_dims' is a 2D array where each row corresponds to the average dimensions for the detected class of each item
    # You can then add these average dimensions to the 'dim' array directly
    dim_np += average_dims

    # Assuming 'conf' is a 2D NumPy array with shape (batch_size, num_classes)
    argmax = np.argmax(conf_np, axis=1)  # Compute argmax for each element in the batch along the class axis

    # Now, 'orient' is a 3D NumPy array with shape (batch_size, num_orientations, 2) where the last dimension contains [cos, sin]
    # We need to select the orientation for each item in the batch according to the argmax index
    orient = np.array([orient_np[i, argmax[i], :] for i in range(orient.shape[0])])

    # 'orient' is now a 2D array with shape (batch_size, 2)
    cos = orient[:, 0]
    sin = orient[:, 1]
    alpha = np.arctan2(sin, cos)  # This will give us the alpha value for each item in the batch

    # Assuming 'angle_bins' is a 1D NumPy array with shape (num_classes,)
    # We want to add the corresponding angle bin to each alpha value in the batch
    angle_bins = np.array(angle_bins)  # Make sure this is a NumPy array if it isn't already
    alpha += angle_bins[argmax]  # Add the angle bin corresponding to each argmax

    alpha -= np.pi  # Subtract pi from each alpha value
    nested_boxes = detection_box_2d_filtered.reshape(-1, 2, 2)
    for i in range(len(nested_boxes)):
        # Convert the flat representation of boxes into a nested list of coordinates
        single_box_2d = nested_boxes[i]
        single_dim = dim_np[i]
        single_alpha = alpha[i]
        single_theta_ray = theta_ray[i]
        plot_regressed_3d_bbox(frame, proj_matrix, single_box_2d, single_dim, single_alpha, single_theta_ray)
    

def main():

     # get parser
    FLAGS = parser.parse_args()
    use_format = FLAGS.use_format
    
    if use_format == 'tensorrt':
         # Load TensorRT engines
        # Create a TensorRT runtime object
        trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        model = 'Z:/education/pytorch_training/3D-BoundingBox-Alt/to3Dmodeld.trt'
        model_engine = load_engine(trt_runtime, model)
    else:
        model = 'Z:/education/pytorch_training/3D-BoundingBox-Alt/to3Dmodel.onnx'
        model_engine = ort.InferenceSession(model, providers=['CUDAExecutionProvider'])


    calib_file = "Z:/education/pytorch_training/3D-BoundingBox-bak/eval/video/2011_09_26/calib_cam_to_cam.txt"
    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)
    proj_matrix = get_P(calib_file)

    odModel = custom(path_or_model=r'Z:\education\pytorch_training\yolov7-tiny.pt')  # custom example

    if odModel:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        odModel.to(device)
        odModel.eval()

    video_source = cv2.VideoCapture('Z:/education/pytorch_training/3dbbtest.mp4')

    # The provided 'averages.dimension_map' is a dictionary with nested arrays.
    # We will need to extract the keys (names) to use them for filtering the 'name' column.
    # Extract the keys (names) for filtering
    dimension_map_keys = list(averages.dimension_map.keys())

    if not video_source.isOpened():
        print("Error opening video stream or file")

    while video_source.isOpened():
        start_time = time.time()
        ret, frame = video_source.read()

        if not ret:
            break

        # Run the main loop with profiling
        main_loop(frame, averages, proj_matrix, model_engine, odModel, angle_bins, dimension_map_keys, use_format)

        print("Per frame: ", (time.time() - start_time))
        print("FPS: ", 1.0 / (time.time() - start_time))

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(-1)

    video_source.release()
    cv2.destroyAllWindows()

    # Print the profiler results
    lp.print_stats()

if __name__ == '__main__':
    main()
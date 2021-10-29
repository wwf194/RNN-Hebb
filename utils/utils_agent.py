import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
#training parameters.

import numpy as np
import math
import cmath
import random

import copy

def save_dict(net, dict=None):
    return net.dict
def load_dict(net, f):
    net.dict=pickle.load(f)

def Getweight(net, name=""):
    try:
        if name in [""]:
            return net.weight_names
        else:
            names = name.split(".",1)
            return net.name_index[(names[0].split("_",1))[0]].Getweight(names[1])
    except Exception:
        return "error"

def Getweight_index(init_des=""):
    index = 1.00e-03
    ex_large_index = 1.00
    large_index = 0.10
    middle_index = 0.01
    if init_des in ["ex-large"]:
        index=ex_large_index
    if init_des in ["large"]:
        index=large_index
    if init_des in ["middle"]:
        index=middle_index
    return index

def Getei_mask(E_num, N_num, device=None):
    ei_mask = torch.zeros((N_num, N_num), device=device, requires_grad=False)
    for i in range(E_num):
        ei_mask[i][i] = 1.0
    for i in range(E_num, N_num):
        ei_mask[i][i] = -1.0
    return ei_mask

def Getmask(N_num, output_num, device=None):
    mask = torch.ones((N_num, output_num), device=device, requires_grad=False)
    return mask

def set_act_func(net, act_func_des="relu"):
    if(isinstance(act_func_des, str)):
        set_act_func_from_name(net, act_func_des)
    elif(isinstance(act_func_des, dict)):
        if(net.dict.get("type")!=None and act_func_des.get(net.type)!=None):
            set_act_func_from_name(net, act_func_des[net.type])
        else:
            set_act_func_from_name(net, act_func_des["default"])
def Getact_func_module(act_func_des):
    name=act_func_des
    if name=="relu":
        return nn.ReLU()
    elif name=="tanh":
        return nn.Tanh()
    elif name=="softplus":
        return nn.Softplus()
    elif name=="sigmoid":
        return nn.Sigmoid()

def Getact_func_from_name(name="relu", param="default"):
    if(name=="none"):
        return lambda x:x
    elif(name=="relu"):
        if(param=="default"):
            return lambda x:F.relu(x)
        else:
            return lambda x:param * F.relu(x)
    elif(name=="tanh"):
        if(param=="default"):
            return lambda x:torch.tanh(x)
        else:
            return lambda x:param * F.tanh(x)
    elif(name=="relu_tanh"):
        if(param=="default"):
            return lambda x:F.relu(torch.tanh(x))
        else:
            return lambda x:param * F.relu(torch.tanh(x))

def Getact_func(act_func_des):
    if(isinstance(act_func_des, list)):
        act_func_name = act_func_des[0]
        act_func_param = act_func_des[1]
    elif(isinstance(act_func_des, str)):
        act_func_name = act_func_des
        act_func_param = "default"
    elif(isinstance(act_func_des, dict)):
        act_func_name = act_func_des["name"]
        act_func_param = act_func_des["param"]
    return Getact_func_from_name(act_func_name, act_func_param)

import utils_torch

def LogSpatialActivity(activity, XYs, SpatialActivity):
    # activity: [BatchSize, StepNum, NeuronNum]
    # XYs: [BatchSize, StepNum, (x, y)]. Can be true XYs or predicted XYs
    BatchSize = activity.shape[0]
    StepNum = activity.shape[1]
    NeuronNum = activity.shape[2]
    XYNum = BatchSize * StepNum
    activity = activity.reshape(XYNum, NeuronNum)
    
    ResolutionX, ResolutionY = SpatialActivity.ResolutionX, SpatialActivity.ResolutionY
    XYs = activity.reshape(XYNum, 2)
    XYPixels = utils_torch.geometry2D.XYs2PixelIndices(
        XYs, SpatialActivity.BoundaryBox,
        ResolutionX, ResolutionY
    ) # [XYNum, 2]
    
    # Exclude-out-of boundarybox XYs
    Mask1 = XYPixels[0] >= 0
    Mask2 = XYPixels[0] < ResolutionX
    Mask3 = XYPixels[1] >= 0
    Mask4 = XYPixels[1] < ResolutionY
    InsideMasks = Mask1 * Mask2 * Mask3 * Mask4
    XYInsideNum = np.sum(InsideMasks)
    XYOutsideNum = XYNum - XYInsideNum
    utils_torch.AddLog(
        "LogSpatialActivity: %d/%d(%.3f) XYs are outside boundarybox."%(XYOutsideNum, XYNum, XYOutsideNum * 1.0 / XYNum),
        logger="LogSpatialActivity", FileOnly=True
    )
    InsideIndices = np.argwhere(InsideMasks)
    
    XYPixels = XYPixels[InsideIndices]
    activity = activity[InsideIndices]

    XYActivitySum = SpatialActivity.XYActivitySum
    XYActivitySquareSum = SpatialActivity.XYActivitySquareSum
    XYActivityCount = SpatialActivity.XYActivityCount
    for XYIndex in range(XYNum):
        # This for-loop might be very time-consuming, but there does not seem a way to do it using numpy array indexing.
        XY = XYPixels[XYIndex]
        X, Y = XY[0], XY[1]
        XYActivitySum[X, Y] += activity[XYIndex]
        XYActivitySquareSum[X, Y] += activity[XYIndex]
        XYActivityCount[X, Y] += 1

def InitSpatialActivity(BoundaryBox, Resolution, NeuronNum):
    SpatialActivity = utils_torch.PyObj()
    SpatialActivity.BoundaryBox = utils_torch.plot.CopyBoundaryBox(BoundaryBox)
    ResolutionX, ResolutionY = utils_torch.plot.ParseResolution(BoundaryBox.Width, Resolution.Height, Resolution)
    SpatialActivity.ResolutionX = ResolutionX
    SpatialActivity.ResolutionY = ResolutionY
    SpatialActivity.NeuronNum = NeuronNum

    SpatialActivity.XYActivitySum = np.zeros((ResolutionX, ResolutionY, NeuronNum), dtype=np.float32)
    SpatialActivity.XYActivitySquareSum = np.zeros((ResolutionX, ResolutionY, NeuronNum), dtype=np.float32)
    SpatialActivity.XYActivityCount = np.zeros((ResolutionX, ResolutionY), dtype=np.int32)

def ClearSpatialActivity(SpatialActivity):
    ResolutionX, ResolutionY, NeuronNum = SpatialActivity.ResolutionX, SpatialActivity.ResolutionY, SpatialActivity.NeuronNum
    SpatialActivity.XYActivitySum = np.zeros((ResolutionX, ResolutionY, NeuronNum), dtype=np.float32)
    SpatialActivity.XYActivitySquareSum = np.zeros((ResolutionX, ResolutionY, NeuronNum), dtype=np.float32)
    SpatialActivity.XYActivityCount = np.zeros((ResolutionX, ResolutionY), dtype=np.int32)

def CalculateSpatialActivity(SpatialActivity):
    SpatialActivity.XYActivityMean = SpatialActivity.XYActivitySum / SpatialActivity.XYActivityCount
    # DX = E(X^2) - (EX)^2
    SpatialActivity.XYActivityVar = SpatialActivity.XYActivitySquareSum / SpatialActivity.XYActivityCount - SpatialActivity.XYActivityMean ** 2
    SpatialActivity.XYActivityStd = SpatialActivity.XYActivityVar ** 0.5
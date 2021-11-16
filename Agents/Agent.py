# -*- coding: utf-8 -*-
#import tensorflow as tf

import os
import math
import random
import sys
from typing import DefaultDict
from utils.utils import GlobalParam
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import cv2 as cv

import utils
from utils_torch.attrs import *
from utils_torch.utils import NpArray2Tensor

class Agent(object):
    def __init__(self, param=None, data=None, **kw):
        utils_torch.model.InitForModel(
            self, param, data, 
            FullName="agent",
            ClassPath="Agents.Agent",
            **kw
        )
    def InitFromParam(self, IsLoad=False):
        utils_torch.model.InitFromParamForModel(self, IsLoad)
        param = self.param
        data = self.data
        cache = self.cache
        if cache.IsInit:
            utils_torch.AddLog("Agent: Initializing...")
        else:
            utils_torch.AddLog("Agent: Loading...")
        self.BuildModules()
        if cache.IsInit:
            EnsureAttrs(param, "InitTasks", default=[])
            utils_torch.model.DoTasksForModel(param.InitTasks, ObjCurrent=self.param, ObjRoot=utils_torch.GetGlobalParam())
        self.InitModules()
        self.ParseRouters()

        if cache.IsInit:
            utils_torch.AddLog("Agent: Initialized.")
        else:
            utils_torch.AddLog("Agent: Loaded.")
    def AddModule(self, name, module):
        setattr(self.cache.Modules, name, module)
    def SetModelInputOutput(self):
        self.SetModelInputOutputNum()
    def SetTrajectory2ModelInputMethod(self):
        param = self.param
        if param.Task in ["PredictXYs"]:
            self.Trajectory2ModelInputInit = self.Trajectory2ModelInputInitXY
        elif param.Task in ["PredictPlaceCellsActivity"]:
            self.Trajectory2ModelInputInit = self.Trajectory2ModelInputInitPlaceCells
        else:

            raise Exception(param.Task)
        EnsureAttrs(param, "Modules.model.Input.Type", default="dXY")
        if param.Modules.model.Input.Type in ["dXY"]:
            self.Trajectory2ModelInput = self.Trajectory2ModelInputdXY
        elif param.Modules.model.Input.Type in ["dLDirection"]:
            self.Trajectory2ModelInput = self.Trajectory2ModelInputdLDirection
        else:
            raise Exception()
    def SetTrajectory2ModelOutputMethod(self):
        param = self.param
        if param.Task in ["PredictXYs"]:
            self.Trajectory2ModelOutput = self.Trajectory2ModelOutputXYs
        elif param.Task in ["PredictPlaceCellsActivity"]:
            self.Trajectory2ModelOutput = self.Trajectory2ModelOutputPlaceCells
        else:
            raise Exception()
    def SetModelInputOutputNum(self):
        param = self.param
        # EnsureAttrs(param, "Task", default="PredictPlaceCellsActivity")
        #DatasetConfig = utils_torch.Datasets.DataSetType2InputOutputOutput(param.Modules.Dataset.Type)
        GlobalParam = utils_torch.GetGlobalParam()
        SetAttrs(param, "Modules.model.Neurons.Output.Num", value=GlobalParam.param.image.Output.Num)
        SetAttrs(param, "Modules.model.Neurons.Input.Num", value=GlobalParam.param.image.Input.Num)
    def ParseParam(self):
        utils_torch.parse.ParsePyObjStatic(self.param, ObjCurrent=self.param, ObjRoot=utils_torch.GetGlobalParam(), InPlace=True)
        #utils_torch.parse.ParsePyObjDynamic(self.param, ObjCurrent=self.param, ObjRoot=utils.GlobalParam, InPlace=True)
        return
    def InitBeforeEpoch(self):
        self.Dataset.ResetBatches()
    def forward(self, data):
        data.update(self.model.forward(data))
        return data
    def SetTensorLocation(self, Location="cpu", Recur=True):
        self.cache.TensorLocation = Location
        if Recur:
            for Name, Module in ListAttrsAndValues(self.cache.Modules):
                if hasattr(Module, "SetTensorLocation"):
                    Module.SetTensorLocation(Location)
    def GetTensorLocation(self):
        return self.cache.TensorLocation
    def SetLogger(self, logger):
        return utils_torch.model.SetLoggerForModel(self, logger)
    def GetLogger(self):
        return utils_torch.model.GetLoggerForModel(self)
    def Log(self, data, Name="Undefined"):
        return utils_torch.model.LogForModel(self, data, Name)
    def PlotTrueAndPredictedTrajectory(self, ):
        return
    # def TrainBatch(self, TrainParam, BatchParam, log):
    #     data = self.Modules.Dataset.GenerateBatch(BatchParam, log=log)
    #     self.Modules.model.Train(data, TrainParam, log=log)
    def ReportSelf(self):
        utils_torch.AddLog("AgentPoint2D: id:%d"%id(self))
        utils_torch.AddLog("AgentPoint2D2: id:%d"%id(utils_torch.GetGlobalParam().object.agent))
        utils_torch.AddLog("Weight.id: %d"%id(self.Modules.model.Modules.Recurrent.Modules.FiringRate2RecurrentInput.data.Weight))

__MainClass__ = Agent
utils_torch.model.SetMethodForModelClass(__MainClass__)
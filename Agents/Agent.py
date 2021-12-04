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

class Agent(utils_torch.module.AbstractModule):
    def __init__(self, param=None, data=None, **kw):
        utils_torch.transform.InitForModule(
            self, param, data,
            FullName="agent",
            ClassPath="Agents.Agent",
            **kw
        )
    def InitFromParam(self, IsLoad=False):
        utils_torch.transform.InitFromParamForModule(self, IsLoad)
        param = self.param
        data = self.data
        cache = self.cache
        if cache.IsInit:
            utils_torch.AddLog("Agent: Initializing...")
        else:
            utils_torch.AddLog("Agent: Loading...")
        self.BuildModules()

        self.Modules.dataset.InitFromParam()
        self.Modules.model.SetNeuronsNum(
            InputNum = self.Modules.dataset.GetInputOutputShape()[0],
            OutputNum = self.Modules.dataset.GetInputOutputShape()[1],
        )        

        self.Modules.model.InitFromParam()

        # self.InitModules()
        self.ParseRouters()

        if cache.IsInit:
            utils_torch.AddLog("Agent: Initialized.")
        else:
            utils_torch.AddLog("Agent: Loaded.")
    def AddModule(self, name, module):
        SetAttrs(self.param, "Modules.%s"%name, value=module)
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
        #DatasetConfig = utils_torch.dataset.DataSetType2InputOutputOutput(param.Modules.Dataset.Type)
        GlobalParam = utils_torch.GetGlobalParam()
        SetAttrs(param, "Modules.model.Neurons.Output.Num", value=GlobalParam.param.image.Output.Num)
        SetAttrs(param, "Modules.model.Neurons.Input.Num", value=GlobalParam.param.image.Input.Num)
    def ParseParam(self):
        utils_torch.parse.ParsePyObjStatic(self.param, ObjCurrent=self.param, ObjRoot=utils_torch.GetGlobalParam(), InPlace=True)
        #utils_torch.parse.ParsePyObjDynamic(self.param, ObjCurrent=self.param, ObjRoot=utils.GlobalParam, InPlace=True)
        return
    # def SetTensorLocation(self, Location="cpu", Recur=True):
    #     self.cache.TensorLocation = Location
    #     if Recur:
    #         for Name, Module in ListAttrsAndValues(self.cache.Modules):
    #             if hasattr(Module, "SetTensorLocation"):
    #                 Module.SetTensorLocation(Location)
    def GetTensorLocation(self):
        return self.cache.TensorLocation
    # def Log(self, data, Name="Undefined"):
    #     return utils_torch.transform.LogForModule(self, data, Name)
    def PlotTrueAndPredictedTrajectory(self, ):
        return
    def ReportSelf(self):
        utils_torch.AddLog("AgentPoint2D: id:%d"%id(self))
        utils_torch.AddLog("AgentPoint2D2: id:%d"%id(utils_torch.GetGlobalParam().object.agent))
        utils_torch.AddLog("Weight.id: %d"%id(self.Modules.model.Modules.Recurrent.Modules.FiringRate2RecurrentInput.data.Weight))
    def RegisterFlow(self, Name):
        setattr(self.cache.flows, Name)
    def RemoveFlow(self, Name):
        delattr(self.cache.flows, Name)
    def GetFlow(self, Name):
        return getattr(self.cache.flows, Name)
    def CreateTrainFlow(self, FlowName="Default", BatchParam=None, log=None):
        self.Modules.dataset.CreateFlow(FlowName, BatchParam, Type="Train")
        return
    def ResetTrainFlow(self, FlowName="Default", BatchParam=None, OptimizeParam=None, log=None):
        cache = self.cache
        flow = self.GetFlow(FlowName)
        self.Modules.dataset.ResetFlow(flow)
        return flow
        # "BeforeEpoch":{ // Things to do before new epoch.
        #     "In": ["BatchParam", "OptimizeParam", "log"],
        #     "Out": ["BatchNum"],
        #     "Routings":[
        #         "Name=Train |--> &^object.image.ClearFlow",
        #         "BatchParam=%BatchParam, Name=Train, Type=Train |--> &^object.image.CreateFlow",
        #         "Name=Train |--> &^object.image.GetBatchNum |--> BatchNum",
        #     ]
        # },
    def RunTrainBatch(self, FlowName="Default", BatchParam=None, OptimizeParam=None, log=None):
        flow = self.GetFlow(FlowName)
        BatchData = self.Modules.dataset.GetBatch(flow)
        self.Modules.model.Dynamics.Optimize(
            BatchData, OptimizeParam, log=log
        )
        return flow
    def AfterTrain(self, FlowName="Default"):
        self.RemoveFlow(self, FlowName)
    def RunTestBatchRandom(self, BatchParam, TrainParam, log):
        # Run 1 test batch. Samples are randomly selected from test set.
        BatchData = self.Modules.dataset.GetBatchRandom(BatchParam, Type="Test")
        self.Modules.model.Dynamics.TestBatch(
            BatchData["Input"], BatchData["Output"], TrainParam, log=log
        )
        # "TestBatch":{
        #     "In":["BatchParam", "TrainParam", "log"],
        #     "Out":[],
        #     "Routings": [
        #         "Name=Test |--> &^object.image.GetBatch |--> DataBatch",
        #         "DataBatch, Name=Input |--> &FilterFromDict |--> ModelInput",
        #         "DataBatch, Name=Output |--> &FilterFromDict |--> ModelOutputTarget",
        #         "ModelInput, ModelOutputTarget, TrainParam, log |--> &model.Dynamics.TestBatch",
        #     ],
        # },
    def CreateTestFlow(self, FlowName="Default", BatchParam=None):
        flow = self.Modules.dataset.CreateFlow(FlowName, Type="Test")
        self.RegisterFlow(FlowName, flow)
    def RunTestBatch(self, FlowName="Default", BatchParam=None, TrainParam=None, log=None):
        # Run 1 test batch. Samples are randomly selected from test set.
        BatchData = self.Modules.dataset.GetBatch(BatchParam, Type="Test")
        self.Modules.model.Dynamics.TestBatch(
            BatchData["Input"], BatchData["Output"], TrainParam, log
        )
        # {
        #     "In":["BatchParam", "TrainParam", "log"],
        #     "Out":[],
        #     "Routings": [
        #         "Name=Test |--> &^object.image.GetBatch |--> DataBatch",
        #         "DataBatch, Name=Input |--> &FilterFromDict |--> ModelInput",
        #         "DataBatch, Name=Output |--> &FilterFromDict |--> ModelOutputTarget",
        #         "ModelInput, ModelOutputTarget, TrainParam, log |--> &model.Dynamics.TestBatch",
        #     ],
        # },
    def RemoveTestFlow(self, FlowName="Default"):
        self.RemoveFlow(FlowName)
    def SetTask(self, TaskName):
        SetAttrs(self.param, "Task", value=TaskName)

__MainClass__ = Agent
utils_torch.transform.SetMethodForTransformModule(__MainClass__)
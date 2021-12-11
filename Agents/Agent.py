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
from utils_torch.attr import *
from utils_torch.utils import NpArray2Tensor

class Agent(utils_torch.module.AbstractModuleWithParam):
    FullNameDefault = "agent"
    def __init__(self, param=None, data=None, **kw):
        return
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        data = self.data
        cache = self.cache

        cache.flows = utils_torch.EmptyPyObj()
        if cache.IsInit:
            utils_torch.AddLog("Agent: Initializing...")
        else:
            utils_torch.AddLog("Agent: Loading...")
        self.BuildModules()

        self.Modules.dataset.Build()
        self.Modules.model.SetNeuronsNum(
            InputNum = self.Modules.dataset.GetInputOutputShape()[0],
            OutputNum = self.Modules.dataset.GetInputOutputShape()[1],
        )        

        self.Modules.model.Build()

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
    def RegisterFlow(self, name, flow):
        setattr(self.cache.flows, name, flow)
    def RemoveFlow(self, Name):
        delattr(self.cache.flows, Name)
    def GetFlow(self, Name):
        return getattr(self.cache.flows, Name)
    def BeforeTrain(self, **kw):
        TensorLocation = kw['TensorLocation']
        self.SetTensorLocation(TensorLocation)
        self.Modules.model.SetTrainWeight() # SetTensorLocation --> SetTrainWeight
    def CreateTrainFlow(self, Name="Default", BatchParam=None, log=None):
        flow = self.Modules.dataset.CreateFlow(Name, BatchParam, Type="Train")
        self.RegisterFlow(Name, flow)
        return flow
    def ResetFlow(self, Name="Default", BatchParam=None, OptimizeParam=None, log=None):
        cache = self.cache
        flow = self.GetFlow(Name)
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
    def RunTrainBatch(self, Name="Default", OptimizeParam=None, log=None):
        flow = self.GetFlow(Name)
        BatchData = self.Modules.dataset.GetBatch(flow)
        self.Modules.model.Dynamics.RunTrainBatch(
            BatchData, OptimizeParam, log=log
        )
        return flow.IsEnd
    def AfterTrain(self, Name="Default"):
        self.RemoveFlow(self, Name)
    def RunTestBatchRandom(self, BatchParam, OptimizeParam, log):
        # Run 1 test batch. Samples are randomly selected from test set.
        BatchData = self.Modules.dataset.GetBatchRandom(BatchParam, Type="Test")
        self.Modules.model.Dynamics.TestBatch(
            BatchData["Input"], BatchData["Output"], OptimizeParam, log=log
        )
        # "TestBatch":{
        #     "In":["BatchParam", "OptimizeParam", "log"],
        #     "Out":[],
        #     "Routings": [
        #         "Name=Test |--> &^object.image.GetBatch |--> DataBatch",
        #         "DataBatch, Name=Input |--> &FilterFromDict |--> ModelInput",
        #         "DataBatch, Name=Output |--> &FilterFromDict |--> ModelOutputTarget",
        #         "ModelInput, ModelOutputTarget, OptimizeParam, log |--> &model.Dynamics.TestBatch",
        #     ],
        # },
    def CreateTestFlow(self, Name="Default", BatchParam=None):
        flow = self.Modules.dataset.CreateFlow(Name, BatchParam, Type='Test')
        self.RegisterFlow(Name, flow)
    def RunTestBatch(self, Name="Default", OptimizeParam=None, log=None):
        # Run 1 test batch. Samples are randomly selected from test set.
        BatchData = self.Modules.dataset.GetBatch(self.GetFlow(Name))
        self.Modules.model.Dynamics.RunTestBatch(
            BatchData, OptimizeParam, log
        )
        # {
        #     "In":["BatchParam", "OptimizeParam", "log"],
        #     "Out":[],
        #     "Routings": [
        #         "Name=Test |--> &^object.image.GetBatch |--> DataBatch",
        #         "DataBatch, Name=Input |--> &FilterFromDict |--> ModelInput",
        #         "DataBatch, Name=Output |--> &FilterFromDict |--> ModelOutputTarget",
        #         "ModelInput, ModelOutputTarget, OptimizeParam, log |--> &model.Dynamics.TestBatch",
        #     ],
        # },
    # def RemoveTestFlow(self, Name="Default"):
    #     self.RemoveFlow(Name)
    def SetTask(self, TaskName):
        SetAttrs(self.param, "Task", value=TaskName)

__MainClass__ = Agent
#utils_torch.transform.SetMethodForTransformModule(__MainClass__)
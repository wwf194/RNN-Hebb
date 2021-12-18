import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
import traceback

import utils_torch
from utils_torch.attr import *


import Agents
from utils_torch.log.AbstractLog import AbstractLogAlongEpochBatchTrain

import analysis
import config

class MainTasksForImageClassification(utils_torch.log.AbstractModuleAlongEpochBatchTrain):
    def __init__(self, TaskParam):
        param = self.param = utils_torch.PyObj(TaskParam)
        data  = self.data  = utils_torch.EmptyPyObj()
        cache = self.cache = utils_torch.EmptyPyObj()    
    
        self.ParseParam()
        utils_torch.ChangeMainSaveDir(
            utils_torch.RenameDirIfExists("./log/" + "%s-%s-%s/"%(param.Task.Dataset.Name, param.Optimize.Method, param.Model.Type))
        )
        if param.MainTask in ["Train"]:
            #SetAttrs(param, "Batch.Num", "Auto")
            param.Batch.Num = "Auto"
            cache.BatchParam = utils_torch.PyObj({
                "Batch.Size": param.Train.Batch.Size,
                "Batch.Num":  param.Train.Batch.Num
            })
            EnsureAttrs(param, "Agent.ParamFile", default="./param/agent.jsonc")
        else:
            raise Exception(param.MainTask)
   
    def ParseParam(self):
        param = self.param
        if not HasAttrs(param, "Model.ParamFile"):
            assert HasAttrs(param, "Model.Type")
            param.Model.ParamFile = self.ParseParamFileFromType(param.Model.Type)
        
        #assert HasAttrs(param.Task.Dataset.Name)
        if not HasAttrs(param, "Task.Dataset.ParamFile"):
            param.Task.Dataset.ParamFile = self.ParseParamFileFromType(param.Task.Dataset.Name)

    def DoTask(self):
        param = self.param
        MainTask = param.MainTask
        if MainTask in ["Train"]:
            self.InitObjects()
            self.Train()
        else:
            raise Exception(MainTask)
    def ParseParamFileFromType(self, Type):
        if Type in ["RNNLIF"]:
            ParamFile = "./param/RNNLIF.jsonc"
        elif Type in ["RNNLIF"]:
            ParamFile = "./param/MLP.jsonc"   
        elif Type in ["cifar10"]:
            ParamFile = "./param/cifar10.jsonc"
        elif Type in ["agent"]:
            ParamFile = "./param/agent.jsonc"
        else:
            raise Exception(Type)
        return ParamFile
    
    def InitObjects(self):
        # GlobalParam = utils_torch.GetGlobalParam()
        # Load param file for nemodel
        param = self.param
        cache = self.cache

        cache.analyzer = analysis.AnalysisForImageClassificationTask()
        cache.log = utils_torch.PyObj({
            "Train": utils_torch.log.LogAlongEpochBatchTrain().Build(IsLoad=False),
            "Test":  utils_torch.log.LogAlongEpochBatchTrain().Build(IsLoad=False)
        })

        cache.logTrain = cache.log.Train
        cache.logTest  = cache.log.Test
        cache.SetEpochBatchList = [
            cache.log.Train, cache.log.Test
        ]
        cache.CheckPointList = []
        
        ModelParam = utils_torch.JsonFile2PyObj(param.Model.ParamFile)
        DatasetParam = utils_torch.JsonFile2PyObj(param.Task.Dataset.ParamFile)
        AgentParam = utils_torch.JsonFile2PyObj(param.Agent.ParamFile)
        agent = self.agent = cache.agent = Agents.Agent().LoadParam(AgentParam)
        
        agent.SetTask(self.param.Task)
        agent.ParseParam()
        agent.AddModuleParam("model", ModelParam)
        agent.AddModuleParam("dataset", DatasetParam)
        config.OverwriteParam(agent)
        config.RegisterCheckPoint(self)
        agent.Build(IsLoad=False)

        
    def RegisterCheckPoint(self, CheckPoint):
        self.cache.CheckPointList.append(CheckPoint)
        self.cache.SetEpochBatchList.append(CheckPoint)
    def GetBatchParam(self):
        return self.param.Train
    def GetOptimizeParam(self):
        return self.param.Optimize
    def Train(self):
        cache = self.cache
        param = self.param
        agent = cache.agent
        analyzer = cache.analyzer
        TensorLocation = self.EnsureTensorLocation()
        #agent.SetTensorLocation(TensorLocation)
        agent.BeforeTrain(TensorLocation=TensorLocation)
        flow = agent.CreateTrainFlow("train", param.Train)
        self.SetBatchNum(flow.BatchNum)
        self.SetEpochNum(param.Train.Epoch.Num)
        self.SetEpochIndex(-1)
        self.SetBatchIndex(self.GetBatchNum() - 1)
        analyzer.BeforeTrain(self.GetTrainContext())
        self.SaveAndLoad(self.GetTrainContext())

        for EpochIndex in range(param.Train.Epoch.Num):
            utils_torch.AddLog("Epoch%d"%(EpochIndex))
            analyzer.BeforeEpoch(self.GetTrainContext())
            self.SetEpochIndex(EpochIndex)
            agent.ResetFlow("train")
            BatchIndex = 0
            while True:
                self.SetBatchIndex(BatchIndex)
                IsEnd = agent.RunTrainBatch("train", param.Optimize, cache.logTrain)

                for CheckPoint in self.cache.CheckPointList:
                    if CheckPoint.IsCheckPoint():
                        CheckPoint.GetMethod()(self.GetTrainContext())
                if IsEnd:
                    break
                BatchIndex += 1

            for CheckPoint in self.cache.CheckPointList:
                IsCheckPoint, Method = CheckPoint.NotifyEndOfEpoch()
                if IsCheckPoint:
                    Method(self.GetTrainContext())    
        agent.RemoveFlow("train")
    def GetTrainContext(self):
        cache = self.cache
        return utils_torch.PyObj({
            "Trainer": self,
            "EpochIndex": self.GetEpochIndex(),
            "BatchIndex": self.GetBatchIndex(),
            "EpochNum": self.GetEpochNum(),
            "BatchNum": self.GetBatchNum()
        })
    def SetBatchNum(self, BatchNum):
        cache = self.cache
        data  = self.data
        data.BatchNum = BatchNum
        for Obj in cache.SetEpochBatchList:
            Obj.SetBatchNum(BatchNum)  
    def SetEpochIndex(self, EpochIndex):
        self.data.EpochIndex = EpochIndex
        cache = self.cache
        for Obj in cache.SetEpochBatchList:
            Obj.SetEpochIndex(EpochIndex)
    def SetBatchIndex(self, BatchIndex):
        self.data.BatchIndex = BatchIndex
        cache = self.cache
        for Obj in cache.SetEpochBatchList:
            Obj.SetBatchIndex(BatchIndex)
    def Register2SetEpochBatchList(self, Obj):
        self.cache.SetEpochBatchList.append(Obj)
    def EnsureTensorLocation(self):
        # To be implemented: tensors on multiple GPUs
        param = self.param
        if not HasAttrs(param, "system.TensorLocation"):
            TensorLocation = utils_torch.GetTensorLocation(Method="auto")
            SetAttrs(param, "system.TensorLocation", value=TensorLocation)
        utils_torch.SetTensorLocation(param.system.TensorLocation)
        return param.system.TensorLocation
    def SaveAndLoad(self, ContextObj):
        SaveDir = utils_torch.GetMainSaveDir() + "SavedModel/" \
            + "Epoch%d-Batch%d/"%(ContextObj.EpochIndex, ContextObj.BatchIndex)
        cache = self.cache
        cache.agent.ToFile(SaveDir, "agent")
        #del cache.agent
        delattr(cache, "agent")
        cache.agent = Agents.Agent().FromFile(SaveDir, "agent").Build(IsLoad=True)
        return self


import transform
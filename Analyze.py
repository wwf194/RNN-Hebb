import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict

import utils_torch
from utils_torch.attrs import *
from utils_torch.json import EmptyPyObj

class LogForWeightAndResponseSimilarityCorrelation:
    def __init__(self):
        self.BatchCount = 0
        self.data = utils_torch.GetDefaultDict(lambda:utils_torch.EmptyPyObj())
    def LogResponse(self, Name, ResponseA, ResponseB):
        Data = self.data[Name]
        EnsureAttrs(Data, "ResponseA", default=[])
        EnsureAttrs(Data, "ResponseB", default=[])
        ResponseA = utils_torch.ToNpArray(ResponseA)
        ResponseB = utils_torch.ToNpArray(ResponseB)
        ResponseA = ResponseA.reshape(-1, ResponseA.shape[-1])
        ResponseB = ResponseB.reshape(-1, ResponseB.shape[-1])
        Data.ResponseA.append(ResponseA)
        Data.ResponseB.append(ResponseB)
    def LogWeight(self, Name, Weight):
        Data = self.data[Name]
        Weight = utils_torch.ToNpArray(Weight)
        Weight = utils_torch.FlattenNpArray(Weight)
        Data.Weight = Weight
    def CalculateResponseSimilarity(self):
        for Name, Data in self.data.items():
            Data.ResponseA = np.concatenate(Data.ResponseA, axis=0)
            Data.ResponseB = np.concatenate(Data.ResponseB, axis=0)
            Data.CorrelationMatrix = utils_torch.math.CalculatePearsonCoefficientMatrix(Data.ResponseA, Data.ResponseB)
        return

class LogForWeightAndResponseSimilarityCorrelationAlongTrain:
    def __init__(self, EpochNum, BatchNum):
        #ConnectivityPattern = utils_torch.EmptyPyObj()
        self.EpochNum = EpochNum
        self.BatchNum = BatchNum
        self.Data = utils_torch.GetDefaultDict(lambda:[])
    def Log(self, Name, EpochIndex, BatchIndex, ResponseSimilarity, ConnectionStrength, CorrelationCoefficient):
        self.Data[Name].append(utils_torch.PyObj({
            "EpochIndex": EpochIndex, 
            "BatchIndex": BatchIndex, 
            "ResponseSimilarity": ResponseSimilarity,
            "ConnectionStrength": ConnectionStrength,
            "CorrelationCoefficient": CorrelationCoefficient
        }))
        return self
    def Plot(self, PlotNum=100, SaveDir=None):
        for Name in self.Data.keys():
            self._Plot(Name, PlotNum, SaveDir, Name)
    def _Plot(self, Name, PlotNum, SaveDir, SaveName):
        BatchNum = self.BatchNum
        Data = self.Data[Name]
        Data.sort(key=lambda Item:Item.EpochIndex + Item.BatchIndex * 1.0 / BatchNum)
        Data = self.Data
        LogNum = len(Data)
        SampleNum = Data[0].ResponseSimilarity.size
        
        PlotIndices = utils_torch.RandomSelect(range(SampleNum), PlotNum)
        PlotNum = len(PlotIndices)

        EpochFloats = []
        CorrelationCoefficients = []
        for _Data in Data:
            EpochFloats.append(_Data.EpochIndex + _Data.BatchIndex * 1.0 / BatchNum)
            CorrelationCoefficients.append(_Data.CorrelationCoefficient)
        fig, ax = utils_torch.plot.CreateFigurePlt(1)
        utils_torch.plot.PlotLineChart(
            ax, EpochFloats, CorrelationCoefficients,
            XLabel="Epochs", YLabel="CorrelationCoefficient of Weight~ResponseSimilarity",
            Title="CorrelationCoefficient of Weight~ResponseSimilarity - Training Process"
        )
        utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "CorrelationCoefficient-Epochs.svg")
        
        YMins, YMaxs = [], []
        XMins, XMaxs = [], []
        for _Data in Data:
            ConnectionStrength = _Data.ConnectionStrength
            ResponseSimilarity = _Data.ResponseSimilarity
            XMin, XMax = np.nanmin(ResponseSimilarity), np.nanmax(ResponseSimilarity)
            YMin, YMax = np.nanmin(ConnectionStrength), np.nanmax(ConnectionStrength) 
            XMins.append(XMin)
            XMaxs.append(XMax)
            YMins.append(YMin)
            YMaxs.append(YMax)
        XMin, XMax, YMin, YMax = min(XMins), max(XMaxs), min(YMins), max(YMaxs)
        
        
        ImagePaths, ImagePathsNoArrow = [], []
        for Index, _Data in enumerate(Data):
            EpochIndex = _Data.EpochIndex
            BatchIndex = _Data.BatchIndex

            Title = "Weight - ResponseSimilarity : Epoch%d-Batch%d"%(EpochIndex, BatchIndex)
            ResponseSimilarity = utils_torch.EnsureFlatNp(_Data.ResponseSimilarity)
            ConnectionStrength = utils_torch.EnsureFlatNp(_Data.ConnectionStrength)
            XYs = np.stack(
                [
                    ResponseSimilarity[PlotIndices],
                    ConnectionStrength[PlotIndices],
                ],
                axis=1
            )

            if Index > 0:
                fig, ax = utils_torch.plot.CreateFigurePlt()
                utils_torch.plot.PlotArrows(ax, _XYs, XYs-_XYs, SizeScale=0.5, HeadWidth=0.005,
                    XLabel="Response Similarity", YLabel="Connection Strength", 
                    Title=Title, XRange=[XMin, XMax], YRange=[YMin, YMax]
                )
                utils_torch.plot.PlotPoints(
                    ax, _XYs, Color="Black", Type="Circle", Size=1.0,
                    XLabel="Response Similarity", YLabel="Connection Strength", 
                    Title=Title, XRange=[XMin, XMax], YRange=[YMin, YMax]
                )
            ImagePath = SaveDir + "cache/" + "Epoch%d-Batch%d-%s-Arrow.png"%(EpochIndex, BatchIndex, SaveName)
            plt.tight_layout()
            utils_torch.plot.SaveFigForPlt(SavePath=ImagePath)
            ImagePaths.append(ImagePath)
            
            fig, ax = utils_torch.plot.CreateFigurePlt()
            utils_torch.plot.PlotPoints(
                ax, XYs, Color="Black", Type="Circle", Size=1.0,
                XLabel="Response Similarity", YLabel="Connection Strength", 
                Title=Title, XRange=[XMin, XMax], YRange=[YMin, YMax]
            )
            ImagePath = SaveDir + "cache/" + "Epoch%d-Batch%d-%s.png"%(EpochIndex, BatchIndex, SaveName)
            plt.tight_layout()
            utils_torch.plot.SaveFigForPlt(SavePath=ImagePath)
            ImagePaths.append(ImagePath)
            ImagePathsNoArrow.append(ImagePath)

            _XYs = XYs
        utils_torch.plot.ImageFiles2GIFFile(
            ImagePaths,
            TimePerFrame=2.0, 
            SavePath=SaveDir + SaveName + "-WithArrow.gif"
        )
        utils_torch.plot.ImageFiles2GIFFile(
            ImagePathsNoArrow,
            TimePerFrame=0.5, 
            SavePath=SaveDir + SaveName + ".gif"
        )

class AnalysisForImageClassificationTask:
    def __init__(self):
        AddAnalysisMethods = utils_torch.PyObj({
            "AnalyzePCAAndConnectivityPattern": AnalyzePCAAndConnectivityPattern,
            "AnalyzeConnectivityPattern": AnalyzeConnectivityPattern,
            "AnalyzePCA": utils_torch.analysis.AnalyzePCA
        })
    def SaveAndLoad(self, ContextObj):
        GlobalParam = utils_torch.GetGlobalParam()
        ContextObj.setdefault("ObjRoot", GlobalParam)
        SaveDir = utils_torch.SetSubSaveDirEpochBatch("SavedModel", ContextObj.EpochIndex, ContextObj.BatchIndex)
        utils_torch.DoTasks(
                "&^param.task.Save", 
                In={"SaveDir": SaveDir},
                **ContextObj.ToDict()
            )
        #print(utils_torch.json.DataFile2PyObj(SaveDir + "agent.model.Recurrent.FiringRate2RecurrentInput.data").Weight[0][0:5])
        utils_torch.DoTasks(
            "&^param.task.Load",
            In={"SaveDir": SaveDir}, 
            **ContextObj.ToDict()
        )
        # GlobalParam.object.agent.ReportSelf()
        GlobalParam = utils_torch.GetGlobalParam()
        GlobalParam.object.trainer.ParseRouters()
        ContextObj.Trainer.agent = GlobalParam.object.agent

    def AnalyzeAfterBatch(self, ContextObj):
        logTest = self.AnalyzeAfterBatchTest(ContextObj.Copy())
        logTrain = self.AnalyzeAfterBatchTrain(ContextObj.Copy())

        utils_torch.AddLog("Plotting Accuracy Curve...")
        utils_torch.analysis.PlotAccuracyEpochBatch(
            LogTrain= logTrain.GetLogValueByName("Accuracy"),
            LogTest = logTest.Main.GetLogValueByName("Accuracy"),
            SaveDir = utils_torch.GetMainSaveDir() + "Performance/",
            ContextObj=ContextObj.Copy(),
        )
    def AnalyzeAfterBatchTrain(self, ContextObj):
        Trainer = ContextObj.Trainer
        EpochIndex = ContextObj["EpochIndex"]
        #model = Trainer.agent.Modules.model

        if EpochIndex < 0:
            return
        BatchIndex = ContextObj["BatchIndex"]
        log = Trainer.cache.LogTrain

        agent = Trainer.agent
        FullName = agent.Modules.model.GetFullName()

        utils_torch.AddLog("Plotting Loss Curve...")
        utils_torch.analysis.AnalyzeLossEpochBatch(
            log.GetLogOfType("Loss"),
            SaveDir = utils_torch.GetMainSaveDir() + "Performance/",
            ContextObj=ContextObj.Copy(),
        )

        
        utils_torch.AddLog("Plotting Neural Activity...")
        utils_torch.analysis.AnalyzeTimeVaryingActivitiesEpochBatch(
            Logs=log.GetLogOfType("TimeVaryingActivity"),
            SaveDir=utils_torch.GetMainSaveDir() + "Activity-Plot/",
            ContextObj=ContextObj.Copy(),
        )

        utils_torch.AddLog("Plotting Activity Statistics...")
        utils_torch.analysis.AnalyzeStatAlongTrainingEpochBatch(
            Logs=log.GetLogOfType("TimeVaryingActivity-Stat"),
            SaveDir=utils_torch.GetMainSaveDir() + "Activity-Stat/",
            ContextObj=ContextObj.Copy()
        )

        utils_torch.AddLog("Plotting Weight...")
        utils_torch.analysis.AnalyzeWeightsEpochBatch(
            Logs=log.GetLogByName(FullName + "." + "Weight"),
            SaveDir=utils_torch.GetMainSaveDir() + "Weight-Plot/",
            ContextObj=ContextObj.Copy()
        )

        utils_torch.AddLog("Plotting Weight Statistics...")
        utils_torch.analysis.AnalyzeStatAlongTrainingEpochBatch(
            Logs=log.GetLogOfType("Weight-Stat"),
            SaveDir=utils_torch.GetMainSaveDir() + "Weight-Stat/",
            ContextObj=ContextObj.Copy()
        )

        return log
    def AnalyzeAfterBatchTest(self, ContextObj):
        log = EmptyPyObj()
        log.FromPyObj(self.RunTestBatches(
            utils_torch.PyObj(ContextObj).Update({"TestBatchNum": 10})
        ))
        utils_torch.analysis.PlotResponseSimilarityAndWeightCorrelation(
            log.Correlation,
            # CorrelationMatrix=logCorrelation.ResponseSimilarity,
            # Weight=logCorrelation.Weight,
            SaveDir=utils_torch.GetMainSaveDir() + "Hebb-Analysis-Test/",
            ContextObj=ContextObj
            #SaveName="Epoch%d-Batch%d-Recurrent.FiringRate2RecurrentInput.Weight"%(EpochIndex, BatchIndex),
        )
        return
    def RunTestBatches(self, ContextObj, MethodsAfterBatch):
        # To Be Modified
        GlobalParam = utils_torch.GetGlobalParam()
        EpochIndex = ContextObj.EpochIndex
        BatchIndex = ContextObj.BatchIndex
        TestBatchNum = ContextObj.setdefault("TestBatchNum", 10)

        agent = GlobalParam.object.agent
        BatchParam = GlobalParam.param.task.Train.BatchParam
        Dataset = GlobalParam.object.image
        Dataset.PrepareBatches(BatchParam, "Test")    
        
        log = utils_torch.log.LogForEpochBatchTrain()
        log.SetEpochIndex(0)
        logCorrelation = LogForWeightAndResponseSimilarityCorrelation()
        logPCA = utils_torch.analysis.LogForPCA()

        logAccuracy = utils_torch.LogForAccuracyAlongTrain()

        FullName = agent.Modules.model.GetFullName()
        
        for TestBatchIndex in range(TestBatchNum):
            log.SetBatchIndex(TestBatchIndex)
            utils_torch.AddLog("Epoch%d-Index%d-TestBatchIndex-%d"%(EpochIndex, BatchIndex, TestBatchIndex))
            InList = utils_torch.parse.ParsePyObjDynamic(
                utils_torch.PyObj([
                    "&^param.task.Train.BatchParam",
                    "&^param.task.Train.OptimizeParam",
                    #"&^param.task.Train.NotifyEpochBatchList"
                    log,
                ]),
                ObjRoot=GlobalParam,
            )
            utils_torch.CallGraph(agent.Dynamics.TestBatchRandom, InList=InList)
            
            # Log Response and Weight Pairs
            for Name, Pair in agent.Modules.model.param.Analyze.ResponseAndWeightPairs.Items():
                logCorrelation.LogResponse(
                    Name,
                    log.GetLogValueByName(FullName + "." + Pair.ResponseA),
                    log.GetLogValueByName(FullName + "." + Pair.ResponseB),
                )

            # Log for PCA Analysis
            for Name, DataName in agent.Modules.model.param.Analyze.PCA.Items():
                logPCA.Log(
                    Name, log.GetLogValueByName(FullName + "." + DataName),
                )

        # Calculate Accuracy
        self.CalculateAccuracyForLog(
            log.GetLogByName("Accuracy"),
            ContextObj,
        )

        # Calculate ResponseSimilarity
        logCorrelation.CalculateResponseSimilarity()
        for Name, Pair in agent.Modules.model.param.Analyze.ResponseAndWeightPairs.Items():
            logCorrelation.LogWeight(Name, log.GetLogValueByName(FullName + "." + "Weight")[Pair.Weight])
        
        # Calculate PCA

        
        return utils_torch.PyObj({
                "Correlation": logCorrelation,
                "PCA": logPCA,
                "Main": log,
            })
            
    def AddAnalysis(self):
        GlobalParam = utils_torch.GetGlobalParam()
        TaskName = GlobalParam.CmdArgs.TaskName2
        _CmdArgs = utils_torch.EnsurePyObj(GlobalParam.CmdArgs)
        if TaskName is not None: # Specify AddAnalysis method from CommandLineArgs
            method = utils_torch.GetAttr(self.AddAnalysisMethods, TaskName)
            method(**_CmdArgs.ToDict())
        else: # To be implemented. Specify from file
            raise Exception()
    def AnalyzeBeforeTrain(self, ContextInfo):
        self.AnalyzeAfterBatchTest(ContextInfo)
        Trainer = ContextInfo.Trainer
        # Trainer.Modules.LogTrain.AccuracyAlongTraining = []
        # Trainer.cache.log.AccuracyAlongTraining = utils_torch.log.LogForAccuracyAlongTrain()
    def AnalyzeAfterTrain(self):
        AnalyzeConnectivityPattern()
        return
    def AnalyzeAfterEveryBatch(self, ContextObj):
        self.CalculateRecentAccuracyForLog(
            ContextObj.Trainer.cahce.LogTrain.GetLogByName("Accuracy"),
            ContextObj,
        )
        return
    def CalculateAccuracyForLog(self, Log, ContextInfo):
        SampleNumTotal = sum(Log["SampleNumTotal"])
        SampleNumCorrect = sum(Log["SampleNumCorrect"])
        Log["CorrectRate"].append(1.0 * SampleNumCorrect / SampleNumTotal)
    def CalculateRecentAccuracyForLog(self, Log, ContextInfo, BatchNumMerge = 5):
        Trainer = ContextInfo.Trainer
        #Accuracy = Trainer.cache.LogTrain.GetLogByName("Accuracy")
        # "Accuracy":{
        #     "SampleNumTotal": [...],
        #     "SampleNumCorrect": [...],
        #     "EpochIndex": [...],
        #     "BatchIndex": [...]
        # }
        Accuracy = Log

        BatchNumTotal = len(Log["SampleNumTotal"])
        
        if BatchNumTotal < BatchNumMerge:
            BatchStartIndex = 0
        else:
            BatchStartIndex = - BatchNumMerge
        SampleNumTotal = sum(Log["SampleNumTotal"][BatchStartIndex:])
        SampleNumCorrect = sum(Log["SampleNumCorrect"][BatchStartIndex:])
        
        Log["CorrectRate"].append(1.0 * SampleNumCorrect / SampleNumTotal)

def AnalyzeConnectivityPattern(*Args, **kw):
    TestBatchNum = kw.setdefault("TestBatchNum", 10)
    # Do supplementary analysis for all saved models under main save directory.
    GlobalParam = utils_torch.GetGlobalParam()
    kw.setdefault("ObjRoot", GlobalParam)
    
    utils_torch.DoTasks( # Dataset can be reused.
        "&^param.task.BuildDataset", **kw
    )

    SaveDirs = utils_torch.GetAllSubSaveDirsEpochBatch("SavedModel")
    
    EpochNum = GlobalParam.param.task.Train.Epoch.Num
    BatchSize = GlobalParam.param.task.Train.BatchParam.Batch.Size
    BatchNum = GlobalParam.object.image.EstimateBatchNum(BatchSize, Type="Train")
    
    AnalysisSaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-Along-Learning-Test/"

    logCorrelation = LogForWeightAndResponseSimilarityCorrelationAlongTrain(EpochNum, BatchNum)
    for SaveDir in SaveDirs:
        EpochIndex, BatchIndex = utils_torch.train.ParseEpochBatchFromStr(SaveDir)
        CacheSavePath = AnalysisSaveDir + "cache/" + "Epoch%d-Batch%d.data"%(EpochIndex, BatchIndex)
        if utils_torch.ExistsFile(CacheSavePath):
            Data = utils_torch.json.DataFile2PyObj(CacheSavePath)

            # Data.CorrelationCoefficient = utils_torch.math.CalculatePearsonCoefficient(Data.ResponseSimilarity, Data.Weight)
            # utils_torch.json.PyObj2DataFile(Data, CacheSavePath)
            
            logCorrelation.Log(
                EpochIndex, BatchIndex, Data.ResponseSimilarity, Data.Weight, Data.CorrelationCoefficient,
            )
            continue
        utils_torch.AddLog("Testing Model at Epoch%d-Batch%d"%(EpochIndex, BatchIndex))
        log = utils_torch.Getlog("DataTest")
        log.SetEpochIndex(EpochIndex)
        log.SetBatchIndex(BatchIndex)

        utils_torch.DoTasks(
            "&^param.task.Load",
            In={"SaveDir": SaveDir}, 
            **kw
        )
        utils_torch.DoTasks(
            "&^param.task.BuildTrainer", **kw
        )
        
        _logCorrelation = CalculateConnectivityPattern(
            EpochIndex=EpochIndex, BatchIndex=BatchIndex, log=log, TestBatchNum=TestBatchNum
        )

        _logCorrelation.CorrelationCoefficient = utils_torch.math.CalculatePearsonCoefficient(
            _logCorrelation.ResponseSimilarity, _logCorrelation.Weight
        )

        utils_torch.json.PyObj2DataFile(
            utils_torch.PyObj({
                "ResponseSimilarity": _logCorrelation.ResponseSimilarity,
                "Weight": _logCorrelation.Weight,
                "CorrelationCoefficient": _logCorrelation.CorrelationCoefficient,
            }),
            CacheSavePath
        )
        logCorrelation.Log(
            EpochIndex, BatchIndex, _logCorrelation.ResponseSimilarity, _logCorrelation.Weight
        )
    logCorrelation.Plot(
        PlotNum=100, SaveDir=AnalysisSaveDir, SaveName="Recurrent.FiringRate2Output.Weight"
    )

def CalculateConnectivityPattern(ContextInfo):
    GlobalParam = utils_torch.GetGlobalParam()
    EpochIndex = ContextInfo.EpochIndex
    BatchIndex = ContextInfo.BatchIndex
    TestBatchNum = ContextInfo.setdefault("TestBatchNum", 10)
    Trainer = ContextInfo.Trainer

    agent = GlobalParam.object.agent
    BatchParam = GlobalParam.param.task.Train.BatchParam
    Dataset = GlobalParam.object.image
    Dataset.PrepareBatches(BatchParam, "Test")    
    
    log = utils_torch.log.LogForEpochBatchTrain()
    log.SetEpochIndex(0)
    logCorrelation = LogForWeightAndResponseSimilarityCorrelation()
    FullName = agent.Modules.model.GetFullName()
    
    for TestBatchIndex in range(TestBatchNum):
        log.SetBatchIndex(TestBatchIndex)
        utils_torch.AddLog("Epoch%d-Index%d-TestBatchIndex-%d"%(EpochIndex, BatchIndex, TestBatchIndex))
        InList = utils_torch.parse.ParsePyObjDynamic(
            utils_torch.PyObj([
                "&^param.task.Train.BatchParam",
                "&^param.task.Train.OptimizeParam",
                #"&^param.task.Train.NotifyEpochBatchList"
                log,
            ]),
            ObjRoot=GlobalParam,
        )
        utils_torch.CallGraph(agent.Dynamics.TestBatchRandom, InList=InList)
        for Name, Pair in agent.Modules.model.param.Analyze.ResponseAndWeightPairs.Items():
            logCorrelation.LogResponse(
                Name,
                log.GetLogValueByName(FullName + "." + Pair.ResponseA),
                log.GetLogValueByName(FullName + "." + Pair.ResponseB),
            )
    logCorrelation.CalculateResponseSimilarity()
    for Name, Pair in agent.Modules.model.param.Analyze.ResponseAndWeightPairs.Items():
        logCorrelation.LogWeight(Name, log.GetLogValueByName(FullName + "." + "Weight")[Pair.Weight])
    return logCorrelation

def AnalyzePCAAndConnectivityPattern(*Args, **kw):
    TestBatchNum = kw.setdefault("TestBatchNum", 10)
    # Do supplementary analysis for all saved models under main save directory.
    GlobalParam = utils_torch.GetGlobalParam()
    kw.setdefault("ObjRoot", GlobalParam)
    
    EffectiveDimRatios = kw.setdefault("EffeciveDimRatios", ["100", "099", "095", "080", "050"])

    utils_torch.DoTasks( # Dataset can be reused.
        "&^param.task.BuildDataset", **kw
    )

    SaveDirs = utils_torch.GetAllSubSaveDirsEpochBatch("SavedModel")
    
    #EpochNum = GlobalParam.param.task.Train.Epoch.Num
    
    BatchSize = GlobalParam.param.task.Train.BatchParam.Batch.Size
    BatchNum = GlobalParam.object.image.EstimateBatchNum(BatchSize, Type="Train")
    
    Data = []
    for SaveDir in SaveDirs:
        EpochIndex, BatchIndex = utils_torch.train.ParseEpochBatchFromStr(SaveDir)
        CacheSavePath = utils_torch.GetMainSaveDir() + "Hebb-Analysis-Test/"\
            + "cache/" + "Epoch%d-Batch%d.data"%(EpochIndex, BatchIndex)
        assert utils_torch.ExistsFile(CacheSavePath)
        DataConnectivityPattern = utils_torch.json.DataFile2PyObj(CacheSavePath)
        #print("WightExamples-Epoch%d-Batch%d"%(EpochIndex, BatchIndex), Data.Weight[0:5])
        CacheSavePath = utils_torch.GetMainSaveDir() + "PCA-Analysis-Along-Training-Test/"\
            + "cache/" + "Epoch%d-Batch%d.data"%(EpochIndex, BatchIndex)
        DataPCA = utils_torch.json.DataFile2PyObj(CacheSavePath)
        Data.append(utils_torch.PyObj({
            "EpochIndex": EpochIndex, "BatchIndex": BatchIndex,
            "CorrelationCoefficient": DataConnectivityPattern.CorrelationCoefficient,
            "EffectiveDimNums": DataPCA.EffectiveDimNums,
        }))
    
    Data.sort(key=lambda Item:Item.EpochIndex + Item.BatchIndex * 1.0 / BatchNum)
    EpochIndices, BatchIndices, EpochFloats = [], [], []
    EffectiveDimNums = defaultdict(lambda:[])
    CorrelationCoefficients = []
    for _Data in Data:
        EpochIndices.append(_Data.EpochIndex)
        BatchIndices.append(_Data.BatchIndex)
        EpochFloats.append(_Data.EpochIndex + _Data.BatchIndex * 1.0 / BatchNum)
        CorrelationCoefficients.append(_Data.CorrelationCoefficient)
        EffectiveDimNums["100"].append(_Data.EffectiveDimNums.P100)
        EffectiveDimNums["099"].append(_Data.EffectiveDimNums.P099)
        EffectiveDimNums["095"].append(_Data.EffectiveDimNums.P095)
        EffectiveDimNums["080"].append(_Data.EffectiveDimNums.P080)
        EffectiveDimNums["050"].append(_Data.EffectiveDimNums.P050)

    fig, axes = utils_torch.plot.CreateFigurePlt(1)
    ax = utils_torch.plot.GetAx(axes, 0)
    GroupNum = len(EffectiveDimNums)
    utils_torch.plot.SetMatplotlibParamToDefault()
    utils_torch.plot.PlotMultiPoints(
        ax, 
        [CorrelationCoefficients for _ in range(GroupNum)],
        EffectiveDimNums.values(),
        XLabel="Correlation between Weight - ResponsSimilarity", YLabel="Effective DimNums",
        Labels = ["$100\%$", "$99\%$", "$95\%$", "$80\%$", "$50\%$"],
        Title = "Effective Dimension Num - Weight~ResponseSimilarity Correlation"
    )
    utils_torch.plot.SaveFigForPlt(SavePath=utils_torch.GetMainSaveDir() + "PCA-Hebb/" + "Hebb-PCA.svg")
    return

# def AnalyzeResponseSimilarityAndWeightUpdateCorrelation(*Args, **kw):
#     # Do supplementary analysis for all saved models under main save directory.
#     kw.setdefault("ObjRoot", utils_torch.GetGlobalParam())
    
#     utils_torch.DoTasks( # Dataset can be reused.
#         "&^param.task.BuildDataset", **kw
#     )

#     SaveDirs = utils_torch.GetAllSubSaveDirsEpochBatch("SavedModel")
#     for SaveDir in SaveDirs:
#         EpochIndex, BatchIndex = utils_torch.train.ParseEpochBatchFromStr(SaveDir)
#         utils_torch.AddLog("Testing Model at Epoch%d-Batch%d"%(EpochIndex, BatchIndex))
#         log = utils_torch.Getlog("DataTest")
#         log.SetEpochIndex(EpochIndex)
#         log.SetBatchIndex(BatchIndex)

#         utils_torch.DoTasks(
#             "&^param.task.Load",
#             In={"SaveDir": SaveDir}, 
#             **kw
#         )
#         utils_torch.DoTasks(
#             "&^param.task.BuildTrainer", **kw
#         )
#         GlobalParam = utils_torch.GetGlobalParam()
#         Trainer = GlobalParam.object.trainer

#         InList = utils_torch.parse.ParsePyObjDynamic(
#             utils_torch.PyObj([
#                 "&^param.task.Train.BatchParam",
#                 "&^param.task.Train.OptimizeParam",
#                 "&^param.task.Train.NotifyEpochBatchList"
#             ]),
#             ObjRoot=GlobalParam
#         )
#         utils_torch.CallGraph(Trainer.Dynamics.TestEpoch, InList=InList)

#         utils_torch.analysis.AnalyzeResponseSimilarityAndWeightUpdateCorrelation(
#             ResponseA=log.GetLogByName("agent.model.FiringRates")["Value"],
#             ResponseB=log.GetLogByName("agent.model.Outputs")["Value"],
#             WeightUpdate=log.GetLogByName("MinusGrad")["Value"]["Recurrent.FiringRate2Output.Weight"],
#             Weight = log.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2Output.Weight"],
#             SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-1/" + "Recurrent.FiringRate2Output/",
#             SaveName = "Epoch%d-Batch%d-Recurrent.FiringRate2Output.Weight"%(EpochIndex, BatchIndex),
#         )

#         utils_torch.analysis.AnalyzeResponseSimilarityAndWeightUpdateCorrelation(
#             ResponseA=log.GetLogByName("agent.model.FiringRates")["Value"],
#             ResponseB=log.GetLogByName("agent.model.FiringRates")["Value"],
#             WeightUpdate=log.GetLogByName("MinusGrad")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
#             Weight = log.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
#             SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-2/" + "Recurrent.FiringRate2RecurrentInput/",
#             SaveName = "Epoch%d-Batch%d-Recurrent.FiringRate2RecurrentInput.Weight"%(EpochIndex, BatchIndex),
#         )
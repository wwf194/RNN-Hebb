import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

import utils_torch
from utils_torch.log.AbstractLog import AbstractLogAlongEpochBatchTrain
from utils_torch.attr import *

class AbstractAnalysisAlongEpochBatchTrain(AbstractLogAlongEpochBatchTrain):
    def __init__(self, **kw):
        return
    def BeforeTrain(self, ContextObj):
        #utils_torch.AddLog("Analysis before train: Null")
        return
    def BeforeEpoch(self, ContextObj):
        #utils_torch.AddLog("Analysis before epoch: Null")
        return
    def AfterEpoch(self, ContextObj):
        #utils_torch.AddLog("Analysis after epoch: Null")
        return

class AnalysisForImageClassificationTask(AbstractAnalysisAlongEpochBatchTrain):
    def __init__(self):
        AddAnalysisMethods = utils_torch.PyObj({
            "AnalyzePCAAndResponseWeightCorrelation": AnalyzePCAAndResponseWeightCorrelation,
            "PlotResponseSimilarityAndWeightCorrelation": PlotResponseSimilarityAndWeightCorrelationAlongTrain,
            "AnalyzePCA": utils_torch.analysis.AnalyzePCAForEpochBatchTrain
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

    def AfterBatch(self, ContextObj):
        logTest  = self.AfterBatchTest(ContextObj.Copy())
        logTrain = self.AfterBatchTrain(ContextObj.Copy())

        utils_torch.AddLog("Plotting Accuracy Along Train...")
        utils_torch.analysis.PlotAccuracyEpochBatch(
            LogTrain = logTrain.GetLogValueByName("Accuracy"),
            LogTest  = ContextObj.Trainer.cache.logTest.GetLogValueByName("Accuracy"),
            SaveDir  = utils_torch.GetMainSaveDir() + "Performance-Accuracy/",
            #SaveName = "Epoch%d-Batch%d-CorrectRate~Epoch"%(ContextObj.EpochIndex, ContextObj.BatchIndex),
            SaveName = "CorrectRate~Epoch",
            ContextObj=ContextObj.Copy(),
        )

        utils_torch.AddLog("Plotting Loss Along Train...")
        utils_torch.analysis.PlotTotalLossEpochBatch(
            LogTrain = logTrain.GetLogByName("TotalLoss"),
            LogTest  = ContextObj.Trainer.cache.logTest.GetLogByName("TotalLoss"),
            SaveDir = utils_torch.GetMainSaveDir() + "Performance-Loss/",
            #SaveName = "Epoch%d-Batch%d-TotalLoss~Epoch"%(ContextObj.EpochIndex, ContextObj.BatchIndex),
            SaveName = "TotalLoss~Epoch",
            ContextObj=ContextObj.Copy(),
        )
    def AfterBatchTrain(self, ContextObj):
        Trainer = ContextObj.Trainer
        EpochIndex = ContextObj.EpochIndex
        BatchIndex = ContextObj.BatchIndex
        # if EpochIndex < 0:
        #     return

        log = Trainer.cache.logTrain
        agent = Trainer.agent
        FullName = agent.Modules.model.GetFullName()

        utils_torch.AddLog("Plotting Loss Along Train...")
        utils_torch.analysis.PlotAllLossEpochBatch(
            log.GetLogOfType("Loss"),
            SaveDir = utils_torch.GetMainSaveDir() + "Performance-Loss/",
            SaveName = "Epoch%d-Batch%d-Loss(Train)~Epoch"%(ContextObj.EpochIndex, ContextObj.BatchIndex),
            ContextObj=ContextObj.Copy(),
        )

        utils_torch.AddLog("Plotting Neural Activity...")
        utils_torch.analysis.AnalyzeTimeVaryingActivitiesEpochBatch(
            Logs=log.GetLogOfType("ActivityAlongTime"),
            SaveDir=utils_torch.GetMainSaveDir() + "Activity-Plot/",
            ContextObj=ContextObj.Copy(),
        )

        utils_torch.AddLog("Plotting Activity Statistics...")
        utils_torch.analysis.AnalyzeStatAlongTrainEpochBatch(
            Logs=log.GetLogOfType("ActivityAlongTime-Stat"),
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
        utils_torch.analysis.AnalyzeStatAlongTrainEpochBatch(
            Logs=log.GetLogOfType("Weight-Stat"),
            SaveDir=utils_torch.GetMainSaveDir() + "Weight-Stat/",
            ContextObj=ContextObj.Copy()
        )
        return log
    def AfterBatchTest(self, ContextObj):
        log = self.RunTestBatches(
            utils_torch.PyObj(ContextObj).Update({"TestBatchNum": 10})
        )

        # Save Weight ~ ResponseSimilarity Correlation data and plot.
        DirHebb = utils_torch.GetMainSaveDir() + "Hebb-Analysis-Test/"
        log.Correlation.ToFile(DirHebb + "cache/", "Epoch%d-Batch%d-ResponseWeightCorrelation"%(ContextObj.EpochIndex, ContextObj.BatchIndex))
        log.Correlation.Plot(
            SaveDir=DirHebb, ContextObj=ContextObj.Copy()
        )
        Logs = ScanLogForSimilarityAndWeightCorrelation(DirHebb + "cache/", ContextObj.Copy())
        logCorrelationAlongTrain = LogForResponseSimilarityAndWeightCorrelationAlongEpochBatchTrain(ContextObj.EpochNum, ContextObj.BatchNum)
        logCorrelationAlongTrain.LogBatches(Logs)
        logCorrelationAlongTrain.Plot(
            SaveDir = DirHebb,
            ContextObj = ContextObj.Copy()
        )

        # Save PCA data
        DirPCA = utils_torch.GetMainSaveDir() + "PCA-Analysis-Along-Train-Test/"
        log.PCA.ToFile(DirPCA + "cache/", "Epoch%d-Batch%d-PCA"%(ContextObj.EpochIndex, ContextObj.BatchIndex))

        # Load saved PCA data at different batches and plot PCA along train.
        LogsPCA = utils_torch.analysis.ScanLogPCA(
            ScanDir = DirPCA + "cache/"
        )
        utils_torch.analysis.PlotPCAAlongTrain(
            LogsPCA,
            SaveDir = DirPCA + "Along-Train/",
            ContextObj = ContextObj.Copy()
        )

        return log
    def RunTestBatches(self, ContextObj):
        Trainer = ContextObj.Trainer
        EpochIndex = ContextObj.EpochIndex
        BatchIndex = ContextObj.BatchIndex
        TestBatchNum = ContextObj.setdefault("TestBatchNum", 10)

        agent = Trainer.agent
        BatchParam = Trainer.GetBatchParam()
        dataset = agent.Modules.dataset
        dataset.CreateFlow("RunTestBatches", BatchParam, Type="Test")

        logTest = Trainer.cache.log.Test
        logTest.SetEpochIndex(ContextObj.EpochIndex)
        logTest.SetBatchIndex(ContextObj.BatchIndex)
        
        log = utils_torch.log.LogAlongEpochBatchTrain().Build(IsLoad=False)
        log.SetEpochIndex(0)

        logCorrelation = LogForResponseSimilarityAndWeightCorrelation().Build().SetEpochBatchIndex(EpochIndex, BatchIndex)
        logPCA = utils_torch.analysis.LogForPCA().Build().SetEpochBatchIndex(EpochIndex, BatchIndex)

        #logAccuracy = utils_torch.analysis.LogForAccuracyAlongTrain()
        FullName = agent.Modules.model.GetFullName()
        
        agent.CreateTestFlow(Name="test", BatchParam=Trainer.GetBatchParam())
        for TestBatchIndex in range(TestBatchNum):
            log.SetBatchIndex(TestBatchIndex)
            utils_torch.AddLog("Epoch%d-Batch%d-TestBatch-%d"%(EpochIndex, BatchIndex, TestBatchIndex))
            # InList = [
            #     Trainer.GetBatchParam(), 
            #     Trainer.GetOptimizeParam(),
            #     log
            # ]
            # utils_torch.Call(agent.Dynamics.RunTestBatch, *InList)
            agent.RunTestBatch(Name="test", OptimizeParam=Trainer.GetOptimizeParam(), log=log)

            # Log Response and Weight Pairs
            for Name, Pair in agent.Modules.model.param.Analyze.ResponseAndWeightPairs.Items():
                logCorrelation.LogBatch(
                    Name=Name,
                    ResponseA=log.GetLogValueByName(FullName + "." + Pair.ResponseA),
                    ResponseB=log.GetLogValueByName(FullName + "." + Pair.ResponseB),
                )
            # Log for PCA Analysis
            for Name in agent.Modules.model.param.Analyze.PCA:
                logPCA.LogBatch(
                    FullName + "." + Name, log.GetLogValueByName(FullName + "." + Name),
                )
        agent.RemoveFlow("test")

        # Calculate performance-accuracy and add to Trainer.cache.LogTest
        self.CalculateAverageAccuracyForLog(
            log.GetLogByName("Accuracy"), ContextObj=ContextObj.Copy()
        )
        _logAccuracy = log.GetLogByName("Accuracy")
        logTest.AddLogDict("Accuracy", 
            {
                "SampleNumCorrect": sum(_logAccuracy["SampleNumCorrect"]),
                "SampleNumTotal": sum(_logAccuracy["SampleNumTotal"]),
                "CorrectRate": _logAccuracy["CorrectRate"],
            }
        )

        logLoss = log.GetLogOfType("Loss")
        self.CalculateAverageLossForLog(
            logLoss, ContextObj=ContextObj.Copy()
        )
        for Name, Log in logLoss.items():
            logTest.AddLogList(Name, Log["AverageLoss"], Type="Loss")

        # Calculate ResponseSimilarity and set weight for each response pair
        for Name, Pair in agent.Modules.model.param.Analyze.ResponseAndWeightPairs.Items():
            logCorrelation.LogWeight(Name, log.GetLogValueByName(FullName + "." + "Weight")[Pair.Weight])
        logCorrelation.Analyze()
        
        # Calculate PCA
        logPCA.CalculatePCA()
        
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
    def BeforeTrain(self, ContextObj):
        self.AfterBatchTest(ContextObj)
    def AfterTrain(self, ContextObj):
        self.AfterBatchTest(ContextObj)
        return
    def AfterEveryBatch(self, ContextObj):
        self.CalculateRecentAccuracyForLog(
            ContextObj.Trainer.cache.logTrain.GetLogByName("Accuracy"),
            ContextObj,
        )
        return
    def CalculateAverageAccuracyForLog(self, Log, ContextObj):
        SampleNumTotal = sum(Log["SampleNumTotal"])
        SampleNumCorrect = sum(Log["SampleNumCorrect"])
        Log["CorrectRate"].append(1.0 * SampleNumCorrect / SampleNumTotal)
    def CalculateAverageLossForLog(self, LogDict, ContextObj):
        for Name, Log in LogDict.items():
            Loss = Log["Value"]
            Log["AverageLoss"] = sum(Loss)/len(Loss)
    def CalculateRecentAccuracyForLog(self, Log, ContextObj, BatchNumMerge = 5):
        #Trainer = ContextObj.Trainer
        #Accuracy = Trainer.cache.LogTrain.GetLogByName("Accuracy")
        # "Accuracy":{
        #     "SampleNumTotal": [...],
        #     "SampleNumCorrect": [...],
        #     "EpochIndex": [...],
        #     "BatchIndex": [...]
        # }
        #Accuracy = Log

        BatchNumTotal = len(Log["SampleNumTotal"])
        
        if BatchNumTotal < BatchNumMerge:
            BatchStartIndex = 0
        else:
            BatchStartIndex = - BatchNumMerge
        SampleNumTotal = sum(Log["SampleNumTotal"][BatchStartIndex:])
        SampleNumCorrect = sum(Log["SampleNumCorrect"][BatchStartIndex:])
        Log["CorrectRate"].append(1.0 * SampleNumCorrect / SampleNumTotal)

def ScanLogForSimilarityAndWeightCorrelation(ScanDir, ContextObj):
    FileList = utils_torch.file.ListAllFiles(ScanDir)
    Logs = []
    LoadedFileNames = []
    for FileName in FileList:
        #utils_torch.AddLog(FileName)
        FileName = utils_torch.RStrip(FileName, ".data")
        FileName = utils_torch.RStrip(FileName, ".jsonc")
        # if FileName.endswith(".data"):
        #     FileName = FileName.rstrip(".data")
        # elif FileName.endswith(".jsonc"): # rstrip does not work correctly here.
        #     FileName = FileName.rstrip(".jsonc")
        # else:
        #     raise Exception()
        if FileName in LoadedFileNames:
            continue
        LoadedFileNames.append(FileName)
        Logs.append(LogForResponseSimilarityAndWeightCorrelation().FromFile(ScanDir, FileName).Build(IsLoad=True))
    return Logs

def PlotResponseSimilarityAndWeightCorrelationAlongTrain(*Args, **kw):
    TestBatchNum = kw.setdefault("TestBatchNum", 10)
    # Do supplementary analysis for all saved models under main save directory.
    GlobalParam = utils_torch.GetGlobalParam()
    kw.setdefault("ObjRoot", GlobalParam)
    
    utils_torch.DoTasks( # Dataset can be reused.
        "&^param.task.BuildDataset", **kw
    )
    
    EpochNum = GlobalParam.param.task.Train.Epoch.Num
    BatchSize = GlobalParam.param.task.Train.BatchParam.Batch.Size
    BatchNum = GlobalParam.object.image.EstimateBatchNum(BatchSize, Type="Train")
    
    AnalysisSaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-Along-Learning-Test/"

    logCorrelation = LogForResponseSimilarityAndWeightCorrelationAlongEpochBatchTrain(EpochNum, BatchNum)
    SaveDirs = utils_torch.GetAllSubSaveDirsEpochBatch("SavedModel")
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
        
        _logCorrelation = CalculateResponseSimilarityAndWeightCorrelation(
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

def CalculateResponseSimilarityAndWeightCorrelation(ContextInfo):
    GlobalParam = utils_torch.GetGlobalParam()
    EpochIndex = ContextInfo.EpochIndex
    BatchIndex = ContextInfo.BatchIndex
    TestBatchNum = ContextInfo.setdefault("TestBatchNum", 10)
    Trainer = ContextInfo.Trainer

    agent = Trainer.GetAgent()

    BatchParam = Trainer.GetBatchParam()
    Dataset = Trainer.GetWorld()
    Dataset.CreateFlow(BatchParam, "Test")    
    
    log = utils_torch.log.LogForEpochBatchTrain()
    log.SetEpochIndex(0)
    logCorrelation = LogForResponseSimilarityAndWeightCorrelation()
    FullName = agent.Modules.model.GetFullName()
    
    for TestBatchIndex in range(TestBatchNum):
        log.SetBatchIndex(TestBatchIndex)
        utils_torch.AddLog("Epoch%d-Index%d-TestBatchIndex-%d"%(EpochIndex, BatchIndex, TestBatchIndex))
        InList = utils_torch.parse.ParsePyObjDynamic(
            utils_torch.PyObj([
                "&^param.task.Train.BatchParam",
                "&^param.task.Train.OptimizeParam",
                log,
            ]),
            ObjRoot=GlobalParam,
        )
        utils_torch.CallGraph(agent.Dynamics.RunTestBatch, *InList)
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

def AnalyzePCAAndResponseWeightCorrelation(*Args, **kw):
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

class LogForResponseSimilarityAndWeightCorrelation(utils_torch.log.AbstractLogAlongBatch):
    def __init__(self, EpochIndex=None, BatchIndex=None):
        super().__init__(DataOnly=True)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad=IsLoad)
        data = self.data
        IsInit = not IsLoad
        if IsInit:
            # self.BatchCount = 0
            # data = self.data = utils_torch.EmptyPyObj()
            data.log = utils_torch.GetDefaultDict(lambda:utils_torch.EmptyPyObj())
            # if EpochIndex is not None:
            #     data.EpochIndex = EpochIndex
            # if BatchIndex is not None:
            #     data.BatchIndex = BatchIndex
            data.status = "Initialized"        
        self.log = self.data.log
        return self
    # def FromFile(self, FilePath):
    #     self.data = utils_torch.file.DataFile2PyObj(FilePath)
    #     return self
    def LogBatch(self, Name, ResponseA, ResponseB):
        Data = self.log[Name]
        EnsureAttrs(Data, "ResponseA", default=[])
        EnsureAttrs(Data, "ResponseB", default=[])
        ResponseA = utils_torch.ToNpArray(ResponseA)
        ResponseB = utils_torch.ToNpArray(ResponseB)
        ResponseA = ResponseA.reshape(-1, ResponseA.shape[-1])
        ResponseB = ResponseB.reshape(-1, ResponseB.shape[-1])
        Data.ResponseA.append(ResponseA)
        Data.ResponseB.append(ResponseB)
    def LogWeight(self, Name, Weight):
        Data = self.log[Name]
        Weight = utils_torch.ToNpArray(Weight)
        Weight = utils_torch.FlattenNpArray(Weight)
        Data.Weight = Weight
    def Analyze(self):
        for Name, Data in self.log.items():
            Data.ResponseA = np.concatenate(Data.ResponseA, axis=0)
            Data.ResponseB = np.concatenate(Data.ResponseB, axis=0)
            Data.CorrelationMatrix = utils_torch.math.CalculatePearsonCoefficientMatrix(Data.ResponseA, Data.ResponseB)
            # Correlation of CorrelationMatrix and WeightMatrix. "Correlation of Correlation"
            Data.CorrelationCoefficient = utils_torch.math.CalculatePearsonCoefficient(Data.CorrelationMatrix, Data.Weight)
        self.data.status = "Analyzed"
        return
    def Plot(self, SaveDir, ContextObj):
        assert self.data.status in ["Analyzed"]
        for Name, Data in self.log.items():
            try:
                self.PlotItem(
                    Data.CorrelationMatrix, Data.Weight, 
                    SaveDir=SaveDir + Name + "/" + "Epoch%d-Batch%d/"%(ContextObj.EpochIndex, ContextObj.BatchIndex),
                    SaveName="Weight~ResponseSimilarity"
                )
            except Exception:
                utils_torch.system.Stack2File(
                    SaveDir + Name + "/" + "Epoch%d-Batch%d/"%(ContextObj.EpochIndex, ContextObj.BatchIndex) + "ErrorReport.txt",
                )

                utils_torch.AddWarning("LogForResponseSimilarityAndWeightCorrelation: Error when plotting.")
    def PlotItem(self, CorrelationMatrix, Weight, SaveDir, SaveName):
        # Scatter Plot and Binned Meand and Std.
        fig, axes = utils_torch.plot.CreateFigurePlt(2, Size="Medium")
        
        CorrelationMatrixFlat = utils_torch.EnsureFlat(CorrelationMatrix)
        WeightFlat = utils_torch.EnsureFlat(Weight)
        
        ax = utils_torch.plot.GetAx(axes, 0)
        
        XYs = np.stack(
            [
                CorrelationMatrixFlat,
                WeightFlat,
            ],
            axis=1
        ) # [NeuronNumA * NeuronNumB, (Correlation, Weight)]

        Title = "Weight - ResponseSimilarity"
        utils_torch.plot.PlotPoints(
            ax, XYs, Color="Blue", Type="EmptyCircle", Size=0.5,
            XLabel="Response Similarity", YLabel="Connection Strength", 
            Title=Title,
        )

        ax = utils_torch.plot.GetAx(axes, 1)
        BinStats = utils_torch.math.CalculateBinnedMeanAndStd(CorrelationMatrixFlat, WeightFlat)
        
        utils_torch.plot.PlotMeanAndStdCurve(
            ax, BinStats.BinCenters, BinStats.Mean, BinStats.Std,
            XLabel = "Response Similarity", YLabel="Connection Strength", Title="Weight - Response Similarity Binned Mean And Std",
        )
        
        plt.suptitle(SaveName)
        plt.tight_layout()
        # Scatter plot points num might be very large, so saving in .svg might cause unsmoothness when viewing.
        utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + SaveName + "-Weight-Response-Similarity.png")
        return
#utils_torch.transform.SetMethodForLogClass(LogForResponseSimilarityAndWeightCorrelation, SaveDataOnly=True)        

class LogForResponseSimilarityAndWeightCorrelationAlongEpochBatchTrain(AbstractLogAlongEpochBatchTrain):
    def __init__(self, EpochNum, BatchNum):
        #ConnectivityPattern = utils_torch.EmptyPyObj()
        self.data = utils_torch.EmptyPyObj()
        self.data.EpochNum = EpochNum
        self.data.BatchNum = BatchNum
        self.data.log = utils_torch.GetDefaultDict(lambda:[])
        self.log = self.data.log
    def LogBatches(self, Logs):
        for Log in Logs:
            self.LogBatch(Log)
    def LogBatch(self, Log, EpochIndex=None, BatchIndex=None):
        if EpochIndex is None:
            EpochIndex = Log.GetEpochIndex()
        if BatchIndex is None:
            BatchIndex = Log.GetBatchIndex()
        for Name, Item in Log.log.items():
            self.log[Name].append(
                Item.Copy().RemoveAttrsIfExist("Epoch", "Batch").FromDict({
                    "Epoch": EpochIndex,
                    "Batch": BatchIndex,
                    "EpochFloat": EpochIndex + BatchIndex * 1.0 / self.data.BatchNum
                })
            )
    def Log(self, Name, EpochIndex, BatchIndex, ResponseSimilarity, ConnectionStrength, CorrelationCoefficient):
        self.log[Name].append(utils_torch.PyObj({
            "EpochIndex": EpochIndex, 
            "BatchIndex": BatchIndex, 
            "ResponseSimilarity": ResponseSimilarity,
            "ConnectionStrength": ConnectionStrength,
            "CorrelationCoefficient": CorrelationCoefficient
        }))
        return self
    def Plot(self, PlotNum=100, SaveDir=None, ContextObj=None):
        for Name, Item in self.log.items():
            self.PlotItem(
                Item, PlotNum, 
                SaveDir=SaveDir + Name + "/" + "Epoch%d-Batch%d/"%(ContextObj.EpochIndex, ContextObj.BatchIndex),
                SaveName=Name,
            )
    def PlotItem(self, Data, PlotNum, SaveDir, SaveName):
        self.PlotGIF(Data, PlotNum, SaveDir, SaveName)
        self.PlotCorrelationCoefficientAndEpoch(Data, SaveDir, SaveName)
    def PlotCorrelationCoefficientAndEpoch(self, Data, SaveDir, SaveName):
        EpochFloats = []
        CorrelationCoefficients = []
        for _Data in Data:
            CorrelationCoefficients.append(_Data.CorrelationCoefficient)
            EpochFloats.append(_Data.EpochFloat)
        fig, ax = utils_torch.plot.CreateFigurePlt(1)
        utils_torch.plot.PlotLineChart(
            ax, EpochFloats, CorrelationCoefficients,
            XLabel="Epochs", YLabel="CorrelationCoefficient of Weight~ResponseSimilarity",
            Title="CorrelationCoefficient between Weight~ResponseSimilarity - Training Process"
        )
        plt.tight_layout()
        utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + SaveName + "-CorrelationCoefficient~Epoch.svg")
        return
    def PlotGIF(self, Data, PlotNum, SaveDir, SaveName):
        utils_torch.SortListByCmpMethod(Data, utils_torch.train.CmpEpochBatchDict)
        #Data.sort(key=lambda Item:Item.EpochIndex + Item.BatchIndex * 1.0 / BatchNum)
        SampleNum = Data[0].CorrelationMatrix.size
        
        PlotIndices = utils_torch.RandomSelect(range(SampleNum), PlotNum)
        PlotNum = len(PlotIndices)

        EpochFloats = []
        for _Data in Data:
            EpochFloats.append(_Data.EpochFloat)
        YMins, YMaxs = [], []
        XMins, XMaxs = [], []
        for _Data in Data:
            ConnectionStrength = _Data.Weight
            CorrelationMatrix = _Data.CorrelationMatrix
            XMin, XMax = np.nanmin(CorrelationMatrix), np.nanmax(CorrelationMatrix)
            YMin, YMax = np.nanmin(ConnectionStrength), np.nanmax(ConnectionStrength) 
            XMins.append(XMin)
            XMaxs.append(XMax)
            YMins.append(YMin)
            YMaxs.append(YMax)
        XMin, XMax, YMin, YMax = min(XMins), max(XMaxs), min(YMins), max(YMaxs)

        ImagePaths, ImagePathsNoArrow = [], []
        for Index, _Data in enumerate(Data):
            EpochIndex, BatchIndex = utils_torch.train.GetEpochBatchIndexFromPyObj(_Data)

            Title = "Weight - ResponseSimilarity : Epoch%d-Batch%d"%(EpochIndex, BatchIndex)
            CorrelationMatrix = utils_torch.EnsureFlatNp(_Data.CorrelationMatrix)
            ConnectionStrength = utils_torch.EnsureFlatNp(_Data.Weight)
            XYs = np.stack(
                [
                    CorrelationMatrix[PlotIndices],
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

        utils_torch.file.RemoveFiles(ImagePaths)

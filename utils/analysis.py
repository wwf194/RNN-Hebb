# import matplotlib as mpl
# from matplotlib import pyplot as plt
# import numpy as np
# import utils_torch
# from utils_torch.plot import CreateFigurePlt

# class LogForWeightAndResponseSimilarityCorrelation:
#     def __init__(self):
#         self.BatchCount = 0
#         self.ResponseA = []
#         self.ResponseB = []
#     def LogResponse(self, ResponseA, ResponseB):
#         ResponseA = utils_torch.ToNpArray(ResponseA)
#         ResponseB = utils_torch.ToNpArray(ResponseB)
#         ResponseA = ResponseA.reshape(-1, ResponseA.shape[-1])
#         ResponseB = ResponseB.reshape(-1, ResponseB.shape[-1])
#         self.ResponseA.append(ResponseA)
#         self.ResponseB.append(ResponseB)
#     def LogWeight(self, Weight):
#         Weight = utils_torch.ToNpArray(Weight)
#         Weight = utils_torch.FlattenNpArray(Weight)
#         self.Weight = Weight
#     def CalculateResponseSimilarity(self):
#         self.ResponseA = np.concatenate(self.ResponseA, axis=0)
#         self.ResponseB = np.concatenate(self.ResponseB, axis=0)
#         self.ResponseSimilarity = utils_torch.math.CalculatePearsonCoefficientMatrix(self.ResponseA, self.ResponseB)
#         return
# class LogForWeightAndResponseSimilarityCorrelationAlongTrain:
#     def __init__(self, EpochNum, BatchNum):
#         #ConnectivityPattern = utils_torch.EmptyPyObj()
#         self.EpochNum = EpochNum
#         self.BatchNum = BatchNum
#         self.Data = []
#     def Log(self, EpochIndex, BatchIndex, ResponseSimilarity, ConnectionStrength):
#         self.Data.append(utils_torch.PyObj({
#             "EpochIndex": EpochIndex, 
#             "BatchIndex": BatchIndex, 
#             "ResponseSimilarity": ResponseSimilarity,
#             "ConnectionStrength": ConnectionStrength,
#         }))
#         return self
#     def Plot(self, PlotNum=100, SaveDir=None, SaveName=None):
#         BatchNum = self.BatchNum
#         self.Data.sort(key=lambda Item:Item.EpochIndex + Item.BatchIndex * 1.0 / BatchNum)
#         Data = self.Data
#         LogNum = len(Data)
#         SampleNum = Data[0].ResponseSimilarity.size
        
#         PlotIndices = utils_torch.RandomSelect(range(SampleNum), PlotNum)
#         PlotNum = len(PlotIndices)

#         PlotData = []
#         YMins, YMaxs = [], []
#         XMins, XMaxs = [], []
#         for _Data in Data:
#             ConnectionStrength = _Data.ConnectionStrength
#             ResponseSimilarity = _Data.ResponseSimilarity
#             XMin, XMax = np.nanmin(ResponseSimilarity), np.nanmax(ResponseSimilarity)
#             YMin, YMax = np.nanmin(ConnectionStrength), np.nanmax(ConnectionStrength) 
#             XMins.append(XMin)
#             XMaxs.append(XMax)
#             YMins.append(YMin)
#             YMaxs.append(YMax)
#         XMin, XMax, YMin, YMax = min(XMins), max(XMaxs), min(YMins), max(YMaxs)
#         ImagePaths = []
#         for Index, _Data in enumerate(Data):
#             EpochIndex = _Data.EpochIndex
#             BatchIndex = _Data.BatchIndex


            
#             Title = "Weight - ResponseSimilarity : Epoch%d-Batch%d"%(EpochIndex, BatchIndex)
#             ResponseSimilarity = utils_torch.EnsureFlatNp(_Data.ResponseSimilarity)
#             ConnectionStrength = utils_torch.EnsureFlatNp(_Data.ConnectionStrength)
#             XYs = np.stack(
#                 [
#                     ResponseSimilarity[PlotIndices],
#                     ConnectionStrength[PlotIndices],
#                 ],
#                 axis=1
#             )

#             if Index > 0:
#                 fig, ax = CreateFigurePlt()
#                 utils_torch.plot.PlotArrows(ax, _XYs, XYs-_XYs, SizeScale=0.5, HeadWidth=0.005,
#                     XLabel="Response Similarity", YLabel="Connection Strength", 
#                     Title=Title, XRange=[XMin, XMax], YRange=[YMin, YMax]
#                 )
#                 utils_torch.plot.PlotPoints(
#                     ax, _XYs, Color="Black", Type="Circle", Size=1.0,
#                     XLabel="Response Similarity", YLabel="Connection Strength", 
#                     Title=Title, XRange=[XMin, XMax], YRange=[YMin, YMax]
#                 )
#             ImagePath = SaveDir + "Epoch%d-Batch%d-%s-Arrow.png"%(EpochIndex, BatchIndex, SaveName)
#             plt.tight_layout()
#             utils_torch.plot.SaveFigForPlt(SavePath=ImagePath)
#             ImagePaths.append(ImagePath)
            
#             fig, ax = CreateFigurePlt()
#             utils_torch.plot.PlotPoints(
#                 ax, XYs, Color="Black", Type="Circle", Size=1.0,
#                 XLabel="Response Similarity", YLabel="Connection Strength", 
#                 Title=Title, XRange=[XMin, XMax], YRange=[YMin, YMax]
#             )
#             ImagePath = SaveDir + "Epoch%d-Batch%d-%s.png"%(EpochIndex, BatchIndex, SaveName)
#             plt.tight_layout()
#             utils_torch.plot.SaveFigForPlt(SavePath=ImagePath)
#             ImagePaths.append(ImagePath)

#             _XYs = XYs
#         utils_torch.plot.ImageFiles2GIFFile(
#             ImagePaths,
#             TimePerFrame=2.0, 
#             SavePath=SaveDir + SaveName + ".gif"
#         )


# def AnalyzeBeforeTrain(ContextInfo):
#     AnalyzeTest(ContextInfo)

# def AnalyzeAfterBatch(ContextInfo):
#     AnalyzeTrain(utils_torch.CopyDict(ContextInfo))
#     AnalyzeTest(utils_torch.CopyDict(ContextInfo))

# def AnalyzeTrain(ContextInfo):
#     EpochIndex = ContextInfo["EpochIndex"]
#     BatchIndex = ContextInfo["BatchIndex"]
#     logger = utils_torch.GetLogger("DataTrain")

#     utils_torch.AddLog("Plotting Loss Curve...")
#     utils_torch.analysis.AnalyzeLossEpochBatch(
#         Logs=logger.GetLogOfType("Loss"), **ContextInfo
#     )

#     utils_torch.AddLog("Plotting Neural Activity...")
#     utils_torch.analysis.AnalyzeTimeVaryingActivitiesEpochBatch(
#         Logs=logger.GetLogOfType("TimeVaryingActivity"),
#     )

#     utils_torch.AddLog("Plotting Weight...")
#     utils_torch.analysis.AnalyzeWeightsEpochBatch(
#         Logs=logger.GetLogOfType("Weight"),
#     )

#     utils_torch.AddLog("Plotting Weight Statistics...")
#     utils_torch.analysis.AnalyzeWeightStatAlongTrainingEpochBatch(
#         Logs=logger.GetLogOfType("Weight-Stat"), **ContextInfo
#     )

#     utils_torch.AddLog("Analyzing ConnectionStrength - ResponseSimilarity Relationship...")
#     if logger.GetLogByName("MinusGrad") is not None:
#         utils_torch.analysis.AnalyzeResponseSimilarityAndWeightUpdateCorrelation(
#             ResponseA=logger.GetLogByName("agent.model.FiringRates")["Value"],
#             ResponseB=logger.GetLogByName("agent.model.FiringRates")["Value"],
#             WeightUpdate=logger.GetLogByName("MinusGrad")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
#             Weight = logger.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
#             SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-Recurrent/",
#             SaveName = "Epoch%d-Batch%d-Recurrent.FiringRate2RecurrentInput.Weight"%(EpochIndex, BatchIndex),
#         )
#     return

# def AnalyzeTest(ContextInfo):
#     EpochIndex = ContextInfo["EpochIndex"]
#     BatchIndex = ContextInfo["BatchIndex"]
#     Trainer = ContextInfo["Trainer"]
#     logger = utils_torch.GetLogger("DataTest") # In test router, data are logged onto GlobalParam.log.DataTrain
#     logger.SetEpochIndex(EpochIndex)
#     logger.SetBatchIndex(BatchIndex)

#     _LoggerCorrelation = _AnalyzeConnectivityPattern(
#         EpochIndex=EpochIndex, BatchIndex=BatchIndex, logger=logger, TestBatchNum=10
#     )
#     utils_torch.analysis.PlotResponseSimilarityAndWeightUpdateCorrelation(
#         CorrelationMatrix=_LoggerCorrelation.ResponseSimilarity,
#         Weight=_LoggerCorrelation.Weight,
#         SaveDir=utils_torch.GetMainSaveDir() + "Hebb-Analysis-Recurrent-Test/",
#         SaveName="Epoch%d-Batch%d-Recurrent.FiringRate2RecurrentInput.Weight"%(EpochIndex, BatchIndex),
#     )

#     return

# def AddAnalysis():
#     GlobalParam = utils_torch.GetGlobalParam()
#     TaskName = GlobalParam.CmdArgs.TaskName2
#     _CmdArgs = utils_torch.EnsurePyObj(GlobalParam.CmdArgs)
#     if TaskName is not None: # Specify AddAnalysis method from CommandLineArgs
#         method = GetAttr(AddAnalysisMethods, TaskName)
#         method(**_CmdArgs.ToDict())
#     else: # To be implemented. Specify from file
#         raise Exception()

# AddAnalysisMethods = utils_torch.EmptyPyObj()

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
#         logger = utils_torch.GetLogger("DataTest")
#         logger.SetEpochIndex(EpochIndex)
#         logger.SetBatchIndex(BatchIndex)

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
#             ResponseA=logger.GetLogByName("agent.model.FiringRates")["Value"],
#             ResponseB=logger.GetLogByName("agent.model.Outputs")["Value"],
#             WeightUpdate=logger.GetLogByName("MinusGrad")["Value"]["Recurrent.FiringRate2Output.Weight"],
#             Weight = logger.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2Output.Weight"],
#             SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-1/" + "Recurrent.FiringRate2Output/",
#             SaveName = "Epoch%d-Batch%d-Recurrent.FiringRate2Output.Weight"%(EpochIndex, BatchIndex),
#         )

#         utils_torch.analysis.AnalyzeResponseSimilarityAndWeightUpdateCorrelation(
#             ResponseA=logger.GetLogByName("agent.model.FiringRates")["Value"],
#             ResponseB=logger.GetLogByName("agent.model.FiringRates")["Value"],
#             WeightUpdate=logger.GetLogByName("MinusGrad")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
#             Weight = logger.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
#             SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-2/" + "Recurrent.FiringRate2RecurrentInput/",
#             SaveName = "Epoch%d-Batch%d-Recurrent.FiringRate2RecurrentInput.Weight"%(EpochIndex, BatchIndex),
#         )
# AddAnalysisMethods.AnalyzeResponseSimilarityAndWeightUpdateCorrelation = AnalyzeResponseSimilarityAndWeightUpdateCorrelation

# def AnalyzeConnectivityPattern(*Args, **kw):
#     TestBatchNum = kw.setdefault("TestBatchNum", 10)
#     # Do supplementary analysis for all saved models under main save directory.
#     GlobalParam = utils_torch.GetGlobalParam()
#     kw.setdefault("ObjRoot", GlobalParam)
    
#     utils_torch.DoTasks( # Dataset can be reused.
#         "&^param.task.BuildDataset", **kw
#     )

#     SaveDirs = utils_torch.GetAllSubSaveDirsEpochBatch("SavedModel")
    
#     EpochNum = GlobalParam.param.task.Train.Epoch.Num
    
#     BatchSize = GlobalParam.param.task.Train.BatchParam.Batch.Size
#     BatchNum = GlobalParam.object.image.EstimateBatchNum(BatchSize, Type="Train")
    
#     AnalysisSaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-Along-Learning-Test-2/"

#     LoggerCorrelation = Analyze.LoggerForWeightAndResponseSimilarityCorrelationAlongTraining(EpochNum, BatchNum)
#     for SaveDir in SaveDirs:
#         EpochIndex, BatchIndex = utils_torch.train.ParseEpochBatchFromStr(SaveDir)
#         CacheSavePath = AnalysisSaveDir + "Epoch%d-Batch%d.data"%(EpochIndex, BatchIndex)
#         if utils_torch.ExistsFile(CacheSavePath):
#             Data = utils_torch.json.DataFile2PyObj(CacheSavePath)
#             print("WightExamples-Epoch%d-Batch%d"%(EpochIndex, BatchIndex), Data.Weight[0:5])
#             LoggerCorrelation.Log(
#                 EpochIndex, BatchIndex, Data.ResponseSimilarity, Data.Weight
#             )
#             continue
#         utils_torch.AddLog("Testing Model at Epoch%d-Batch%d"%(EpochIndex, BatchIndex))
#         logger = utils_torch.GetLogger("DataTest")
#         logger.SetEpochIndex(EpochIndex)
#         logger.SetBatchIndex(BatchIndex)

#         utils_torch.DoTasks(
#             "&^param.task.Load",
#             In={"SaveDir": SaveDir}, 
#             **kw
#         )
#         utils_torch.DoTasks(
#             "&^param.task.BuildTrainer", **kw
#         )
        
#         _LoggerCorrelation = _AnalyzeConnectivityPattern(
#             EpochIndex=EpochIndex, BatchIndex=BatchIndex, logger=logger, TestBatchNum=TestBatchNum
#         )

#         utils_torch.json.PyObj2DataFile(
#             utils_torch.PyObj({
#                 "ResponseSimilarity": _LoggerCorrelation.ResponseSimilarity,
#                 "Weight": _LoggerCorrelation.Weight,
#             }),
#             CacheSavePath
#         )
#         LoggerCorrelation.Log(
#             EpochIndex, BatchIndex, _LoggerCorrelation.ResponseSimilarity, _LoggerCorrelation.Weight
#         )
#     LoggerCorrelation.Plot(
#         PlotNum=100, SaveDir=AnalysisSaveDir, SaveName="Recurrent.FiringRate2Output.Weight"
#     )
# AddAnalysisMethods.AnalyzeConnectivityPattern = AnalyzeConnectivityPattern


# def _AnalyzeConnectivityPattern(**kw):
#     GlobalParam = utils_torch.GetGlobalParam()
#     agent = GlobalParam.object.agent
#     Dataset = GlobalParam.object.image
#     BatchParam = GlobalParam.param.task.Train.BatchParam
#     Dataset.PrepareBatches(BatchParam, "Test")
#     logger = kw.get("logger")
#     TestBatchNum = kw.setdefault("TestBatchNum", 10)
#     EpochIndex = kw["EpochIndex"]
#     BatchIndex = kw["BatchIndex"]
#     _LoggerCorrelation = Analyze.LoggerForWeightAndResponseSimilarityCorrelation()
#     for TestBatchIndex in range(TestBatchNum):
#         utils_torch.AddLog("Epoch%d-Index%d-TestBatchIndex-%d"%(EpochIndex, BatchIndex, TestBatchIndex))
#         InList = utils_torch.parse.ParsePyObjDynamic(
#             utils_torch.PyObj([
#                 "&^param.task.Train.BatchParam",
#                 "&^param.task.Train.OptimizeParam",
#                 #"&^param.task.Train.NotifyEpochBatchList"
#             ]),
#             ObjRoot=GlobalParam
#         )
#         utils_torch.CallGraph(agent.Dynamics.TestBatchRandom, InList=IInListn)

#         _LoggerCorrelation.LogResponse(
#             logger.GetLogByName("agent.model.FiringRates")["Value"],
#             logger.GetLogByName("agent.model.FiringRates")["Value"],
#         )
#     _LoggerCorrelation.CalculateResponseSimilarity()  
#     _LoggerCorrelation.LogWeight(logger.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"])

#     return _LoggerCorrelation
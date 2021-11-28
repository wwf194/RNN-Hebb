import utils_torch
import Tasks
class Trainer(utils_torch.train.TrainerForEpochBatchTrain):
    def __init__(self, param, **kw):
        super().__init__(param, **kw)
    def InitFromParam(self, IsLoad=False):
        self.SetAnalyzer()
        super().InitFromParam(IsLoad=IsLoad)
    def SetAnalyzer(self):
        param = self.param
        cache = self.cache
        utils_torch.parse.ParsePyObjStatic(param)
        task = param.Task
        if "CIFAR10" in task:
            cache.analyzer = Tasks.AnalysisForImageClassificationTask()
        elif "MNIST" in task:
            cache.analyzer = Tasks.AnalysisForImageClassificationTask()
        else:
            raise Exception(task)
    def BeforeTrain(self):
        cache = self.cache
        analyzer = cache.analyzer
        self.SetEpochIndex(-1)
        self.SetBatchIndex(cache.BatchNum - 1)
        if hasattr(cache.analyzer, "SaveAndLoad"):
            analyzer.SaveAndLoad(self.GenerateContextInfo())
        if hasattr(cache.analyzer, "BeforeTrain"):
            analyzer.BeforeTrain(self.GenerateContextInfo())
        self.ClearEpoch()
    def AfterTrain(self):
        cache = self.cache
        analyzer = cache.analyzer
        if hasattr(analyzer, "AfterTrain"):
            analyzer.AfterTrain(
                self.GenerateContextInfo().FromDict({
                    "EpochIndex": -2, "BatchIndex": 0
                })
            )
    def GetBatchParam(self):
        return self.BatchParam
    def GetOptimizeParam(self):
        return self.OptimizeParam
    def GetWorld(self):
        return self.world
    def GetAgent(self):
        return self.agent
    def Train(self, agent, world, EpochNum, BatchParam, OptimizeParam, NotifyEpochBatchList):
        cache = self.cache
        data = self.data
        self.SetEpochNum(EpochNum)
        self.agent = agent
        self.world = world
        self.BatchParam = BatchParam
        self.OptimizeParam = OptimizeParam
        BatchNum = utils_torch.functions.Call(
            self.agent.Dynamics.TrainEpochInit,
            BatchParam, OptimizeParam, cache.LogTrain,
        )[0]
        self.SetBatchNum(BatchNum)
        self.Register2NotifyEpochBatchList(NotifyEpochBatchList)
        self.BeforeTrain()
        self.NotifyEpochNum()
        self.NotifyBatchNum()
        for EpochIndex in range(cache.EpochNum):
            self.SetEpochIndex(EpochIndex)
            self.NotifyEpochIndex()
            self.TrainEpoch(BatchParam, OptimizeParam)
        self.AfterTrain()
    def TrainEpoch(self, BatchParam, OptimizeParam):
        cache = self.cache
        self.ClearBatch()
        BatchNum = utils_torch.functions.Call(self.agent.Dynamics.TrainEpochInit, 
            BatchParam, OptimizeParam, self.Modules.LogTrain,
        )[0]
        self.SetBatchNum(BatchNum)
        for BatchIndex in range(BatchNum):
            self.SetBatchIndex(BatchIndex)
            self.NotifyBatchIndex()
            self.TrainBatch(BatchParam, OptimizeParam)
            #cache.analyzer.AnalyzeAfterEveryBatch(self.GenerateContextInfo())
    def TrainBatch(self, BatchParam, OptimizeParam):
        self.ReportEpochBatch()
        self.agent.Dynamics.TrainBatch(BatchParam, OptimizeParam, self.cache.LogTrain)
        for CheckPoint in self.cache.CheckPointList:
            IsCheckPoint = CheckPoint.AddBatchAndReturnIsCheckPoint()
            if IsCheckPoint:
                CheckPoint.GetMethod()(self.GenerateContextInfo())
    # def SaveAndLoad(self, ContextInfo):
    #     self.cache.analyzer.SaveAndLoad(ContextInfo)
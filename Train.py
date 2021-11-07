import utils_torch
class Trainer(utils_torch.train.TrainerForEpochBatchTraining):
    def Train(self, agent, EpochNum, BatchParam, OptimizeParam, NotifyEpochBatchList):
        cache = self.cache
        self.SetEpochNum(EpochNum)
        self.agent = agent
        BatchNum = utils_torch.functions.Call(self.agent.Dynamics.InitBeforeEpochTrain, BatchParam, OptimizeParam)[0]
        self.SetBatchNum(BatchNum)
        self.Register2NotifyEpochBatchList(NotifyEpochBatchList)
        self.SetEpochIndex(-1)
        self.SetBatchIndex(BatchNum - 1)
        self.SaveAndLoad(self.GenerateContextInfo())
        self.ClearEpoch()
        for EpochIndex in range(cache.EpochNum):
            self.SetEpochIndex(EpochIndex)
            self.NotifyEpochIndex()
            self.TrainEpoch(BatchParam, OptimizeParam)
    def TrainEpoch(self, BatchParam, OptimizeParam):
        self.ClearBatch()
        BatchNum = utils_torch.functions.Call(self.agent.Dynamics.InitBeforeEpochTrain, BatchParam, OptimizeParam)[0]
        self.SetBatchNum(BatchNum)
        for BatchIndex in range(BatchNum):
            self.SetBatchIndex(BatchIndex)
            self.NotifyBatchIndex()
            self.TrainBatch(BatchParam, OptimizeParam)
            #self.AddBatchIndex()
    def TrainBatch(self, BatchParam, OptimizeParam):
        self.ReportEpochBatch()     
        self.agent.Dynamics.Train(BatchParam, OptimizeParam)
        for CheckPoint in self.cache.CheckPointList:
            IsCheckPoint = CheckPoint.AddBatchAndReturnIsCheckPoint()
            if IsCheckPoint:
                CheckPoint.GetMethod()(self.GenerateContextInfo())
    def SaveAndLoad(self, ContextInfo):
        SaveAndLoad(ContextInfo)

def SaveAndLoad(ContextInfo):
    EpochIndex = ContextInfo["EpochIndex"]
    BatchIndex = ContextInfo["BatchIndex"]
    ContextInfo.setdefault("ObjRoot", utils_torch.GetGlobalParam())
    SaveDir = utils_torch.SetSubSaveDirEpochBatch("SavedModel", EpochIndex, BatchIndex)
    GlobalParam = utils_torch.GetGlobalParam()
    # GlobalParam.object.agent.ReportSelf()
    # print(id(GlobalParam.object.agent.Modules.model.GetTrainWeight()["Recurrent.FiringRate2RecurrentInput.Weight"]))
    # print(id(GlobalParam.object.agent.Modules.model.Modules.Recurrent.Modules.FiringRate2RecurrentInput.data.Weight))

    #print(GlobalParam.object.agent.Modules.model.Modules.Recurrent.Modules.FiringRate2RecurrentInput.data.Weight[0][0:5])
    utils_torch.DoTasks(
            "&^param.task.Save", 
            In={"SaveDir": SaveDir},
            **ContextInfo
        )
    #print(utils_torch.json.DataFile2PyObj(SaveDir + "agent.model.Recurrent.FiringRate2RecurrentInput.data").Weight[0][0:5])
    utils_torch.DoTasks(
        "&^param.task.Load",
        In={"SaveDir": SaveDir}, 
        **ContextInfo
    )
    # GlobalParam.object.agent.ReportSelf()
    GlobalParam = utils_torch.GetGlobalParam()
    GlobalParam.object.trainer.ParseRouters()

    ContextInfo["Trainer"].agent = GlobalParam.object.agent
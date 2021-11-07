from re import L
import sys
import argparse
import traceback

def ParseCmdArgs():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--task", dest="task", nargs="?", default="QuickScript")
    parser.add_argument("-t", "--task", dest="task", nargs="?", default="DoTasksFromFile")
    # parser.add_argument("-t", "--task", dest="task", nargs="?", default="CopyProject2DirAndRun")

    # if CmdArgs.task in ["CopyProject2DirAndRun"], task2 will be implemented after copy.
    parser.add_argument("-t2", "--task2", dest="task2", default="DoTasksFromFile")
    parser.add_argument("-id", "--IsDebug", dest="IsDebug", default=True)

    # If Args.task in ['CopyProject2DirAndRun'], this argument will be used to designate file to be run.
    parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default=None)
    # parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default="./log/DoTasksFromFile-0/")
    
    parser.add_argument("-tf", "--TaskFile", dest="TaskFile", default="./task.jsonc")

    parser.add_argument("-tn", "--TaskName", dest="TaskName", default="Main")
    # parser.add_argument("-tn", "--TaskName", dest="TaskName", default="AddAnalysis")

    # If Args.task in ['CopyProject2DirAndRun'], this argument will be used to designate file to be run.
    parser.add_argument("-ms", "--MainScript", dest="MainScript", default="main.py")

    parser.add_argument("-tn2", "--TaskName2", dest="TaskName2", default="AnalyzeConnectivityPattern")
    # parser.add_argument("-tn2", "--TaskName2", dest="TaskName2", default="AnalyzeResponseSimilarityAndWeightUpdateCorrelation")
    
    CmdArgs = parser.parse_args()
    return CmdArgs
CmdArgs = ParseCmdArgs()

def main():
    utils_torch.Main(
        CmdArgs=CmdArgs,
        QuickScript=QuickScript
    )

def ScanConfigFile(FilePath="./config.jsonc"):
    import json5
    with open(FilePath, "r") as f:
        Config = json5.load(f) # json5 allows comments. config is either dict or list.
    sys.path.append(Config["Library"]["utils_torch"]["IncludePath"])
    import utils_torch
    import utils
    utils_torch.attrs.SetAttrs(utils.GlobalParam, "config", utils_torch.PyObj(Config)) # mount config on utils_torch.GlobalParam.config
ScanConfigFile()

def ParseMainTask(task):
    if task in ["CleanLog", "CleanLog", "cleanlog"]:
        task = "CleanLog"
    elif task in ["DoTasksFromFile"]:
        task = "DoTasksFromFile"
    elif task in ["QuickScript", "quick"]:
        task = "QuickScript"
    else:
        pass
    return task

def InitUtils():
    import utils_torch
    import utils
    CmdArgs.task = ParseMainTask(CmdArgs.task)
    utils_torch.SetGlobalParam(GlobalParam=utils.GlobalParam)
    if CmdArgs.SaveDir is not None:
        if not CmdArgs.SaveDir.endswith("/"):
            CmdArgs.SaveDir += "/"
        CmdArgs.SaveDir = utils_torch.SetMainSaveDir(GlobalParam=utils.GlobalParam, SaveDir=CmdArgs.SaveDir)
    else:  # Create
        CmdArgs.SaveDir = utils_torch.SetMainSaveDir(GlobalParam=utils.GlobalParam, Name=CmdArgs.task)
    utils_torch.SetLoggerGlobal(GlobalParam=utils.GlobalParam)
InitUtils()

import utils
import utils_torch
from utils_torch.attrs import *
utils_torch.SetGlobalParam(utils.GlobalParam)
utils.GlobalParam.CmdArgs = CmdArgs

def QuickScript(Args):
    # Write temporary code here, and run "python main.py quick"
    utils_torch.files.CopyFile2AllSubDirsUnderDestDir(
        "agent.param.jsonc",
        #"log/RSLP-ReLU-Iter10-Tc0.1-1/SavedModel/Epoch-1-Batch0/",
        "./",
        "log/RSLP-ReLU-Iter10-Tc0.1-2/SavedModel/",
    )

    # utils_torch.Datasets.CIFAR10.OriginalFiles2DataFile(
    #     LoadDir = "/data3/wangweifan/Datasets/CIFAR10/",
    #     SaveDir = "/data3/wangweifan/Datasets/CIFAR10/CIFAR10-Data",
    # )
    # Data = utils_torch.json.DataFile2JsonObj("/data3/wangweifan/Datasets/CIFAR10/CIFAR10-Data")
    # return
# CmdArgs.QuickScript = QuickScript

def AddObjRefForParseRouters():
    ObjRefLocal = utils_torch.PyObj()
    #ObjRefLocal.LogSpatialActivity = utils.model.LogSpatialActivity
    utils_torch.model.Add2ObjRefListForParseRouters(ObjRefLocal)
def RegisterExternalClassesAndMethods():
    import Train
    utils_torch.RegisterExternalMethods("SaveAndLoad", Train.SaveAndLoad)
    utils_torch.RegisterExternalMethods("AnalyzeAfterBatch", AnalyzeAfterBatch)
    utils_torch.RegisterExternalMethods("AnalyzeBeforeTrain", AnalyzeBeforeTrain)
    utils_torch.RegisterExternalMethods("AddObjRefForParseRouters", AddObjRefForParseRouters)
    utils_torch.RegisterExternalMethods("AddAnalysis", AddAnalysis)
    utils_torch.RegisterExternalClasses("Trainer", Train.Trainer)
#utils_torch.RegisterExternalMethods("RegisterExternalMethods", RegisterExternalMethods)


def AnalyzeBeforeTrain(ContextInfo):
    AnalyzeTest(ContextInfo)

def AnalyzeAfterBatch(ContextInfo):
    AnalyzeTrain(utils_torch.CopyDict(ContextInfo))
    AnalyzeTest(utils_torch.CopyDict(ContextInfo))

def AnalyzeTrain(ContextInfo):
    EpochIndex = ContextInfo["EpochIndex"]
    BatchIndex = ContextInfo["BatchIndex"]
    logger = utils_torch.GetLogger("DataTrain")

    utils_torch.AddLog("Plotting Loss Curve...")
    utils_torch.analysis.AnalyzeLossEpochBatch(
        Logs=logger.GetLogOfType("Loss"), **ContextInfo
    )

    utils_torch.AddLog("Plotting Neural Activity...")
    utils_torch.analysis.AnalyzeTimeVaryingActivitiesEpochBatch(
        Logs=logger.GetLogOfType("TimeVaryingActivity"),
    )

    utils_torch.AddLog("Plotting Weight...")
    utils_torch.analysis.AnalyzeWeightsEpochBatch(
        Logs=logger.GetLogOfType("Weight"),
    )

    utils_torch.AddLog("Plotting Weight Statistics...")
    utils_torch.analysis.AnalyzeWeightStatAlongTrainingEpochBatch(
        Logs=logger.GetLogOfType("Weight-Stat"), **ContextInfo
    )

    utils_torch.AddLog("Analyzing ConnectionStrength - ResponseSimilarity Relationship...")
    if logger.GetLogByName("MinusGrad") is not None:
        utils_torch.analysis.AnalyzeResponseSimilarityAndWeightUpdateCorrelation(
            ResponseA=logger.GetLogByName("agent.model.FiringRates")["Value"],
            ResponseB=logger.GetLogByName("agent.model.FiringRates")["Value"],
            WeightUpdate=logger.GetLogByName("MinusGrad")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
            Weight = logger.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
            SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-Recurrent/",
            SaveName = "Epoch%d-Batch%d-Recurrent.FiringRate2RecurrentInput.Weight"%(EpochIndex, BatchIndex),
        )
    return

def AnalyzeTest(ContextInfo):
    EpochIndex = ContextInfo["EpochIndex"]
    BatchIndex = ContextInfo["BatchIndex"]
    Trainer = ContextInfo["Trainer"]
    logger = utils_torch.GetLogger("DataTest") # In test router, data are logged onto GlobalParam.log.DataTrain
    logger.SetEpochIndex(EpochIndex)
    logger.SetBatchIndex(BatchIndex)

    _LoggerCorrelation = _AnalyzeConnectivityPattern(
        EpochIndex=EpochIndex, BatchIndex=BatchIndex, logger=logger, TestBatchNum=10
    )
    utils_torch.analysis.PlotResponseSimilarityAndWeightUpdateCorrelation(
        CorrelationMatrix=_LoggerCorrelation.ResponseSimilarity,
        Weight=_LoggerCorrelation.Weight,
        SaveDir=utils_torch.GetMainSaveDir() + "Hebb-Analysis-Recurrent-Test/",
        SaveName="Epoch%d-Batch%d-Recurrent.FiringRate2RecurrentInput.Weight"%(EpochIndex, BatchIndex),
    )

    return

def AddAnalysis():
    GlobalParam = utils_torch.GetGlobalParam()
    TaskName = GlobalParam.CmdArgs.TaskName2
    _CmdArgs = utils_torch.EnsurePyObj(GlobalParam.CmdArgs)
    if TaskName is not None: # Specify AddAnalysis method from CommandLineArgs
        method = GetAttr(AddAnalysisMethods, TaskName)
        method(**_CmdArgs.ToDict())
    else: # To be implemented. Specify from file
        raise Exception()

AddAnalysisMethods = utils_torch.EmptyPyObj()

def AnalyzeResponseSimilarityAndWeightUpdateCorrelation(*Args, **kw):
    # Do supplementary analysis for all saved models under main save directory.
    kw.setdefault("ObjRoot", utils_torch.GetGlobalParam())
    
    utils_torch.DoTasks( # Dataset can be reused.
        "&^param.task.BuildDataset", **kw
    )

    SaveDirs = utils_torch.GetAllSubSaveDirsEpochBatch("SavedModel")
    for SaveDir in SaveDirs:
        EpochIndex, BatchIndex = utils_torch.train.ParseEpochBatchFromStr(SaveDir)
        utils_torch.AddLog("Testing Model at Epoch%d-Batch%d"%(EpochIndex, BatchIndex))
        logger = utils_torch.GetLogger("DataTest")
        logger.SetEpochIndex(EpochIndex)
        logger.SetBatchIndex(BatchIndex)

        utils_torch.DoTasks(
            "&^param.task.Load",
            In={"SaveDir": SaveDir}, 
            **kw
        )
        utils_torch.DoTasks(
            "&^param.task.BuildTrainer", **kw
        )
        GlobalParam = utils_torch.GetGlobalParam()
        Trainer = GlobalParam.object.trainer

        In = utils_torch.parse.ParsePyObjDynamic(
            utils_torch.PyObj([
                "&^param.task.Train.BatchParam",
                "&^param.task.Train.OptimizeParam",
                "&^param.task.Train.NotifyEpochBatchList"
            ]),
            ObjRoot=GlobalParam
        )
        utils_torch.CallGraph(Trainer.Dynamics.TestEpoch, In=In)

        utils_torch.analysis.AnalyzeResponseSimilarityAndWeightUpdateCorrelation(
            ResponseA=logger.GetLogByName("agent.model.FiringRates")["Value"],
            ResponseB=logger.GetLogByName("agent.model.Outputs")["Value"],
            WeightUpdate=logger.GetLogByName("MinusGrad")["Value"]["Recurrent.FiringRate2Output.Weight"],
            Weight = logger.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2Output.Weight"],
            SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-1/" + "Recurrent.FiringRate2Output/",
            SaveName = "Epoch%d-Batch%d-Recurrent.FiringRate2Output.Weight"%(EpochIndex, BatchIndex),
        )

        utils_torch.analysis.AnalyzeResponseSimilarityAndWeightUpdateCorrelation(
            ResponseA=logger.GetLogByName("agent.model.FiringRates")["Value"],
            ResponseB=logger.GetLogByName("agent.model.FiringRates")["Value"],
            WeightUpdate=logger.GetLogByName("MinusGrad")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
            Weight = logger.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
            SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-2/" + "Recurrent.FiringRate2RecurrentInput/",
            SaveName = "Epoch%d-Batch%d-Recurrent.FiringRate2RecurrentInput.Weight"%(EpochIndex, BatchIndex),
        )
AddAnalysisMethods.AnalyzeResponseSimilarityAndWeightUpdateCorrelation = AnalyzeResponseSimilarityAndWeightUpdateCorrelation

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
    
    AnalysisSaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis-Along-Learning-Test-2/"

    LoggerCorrelation = utils.analysis.LoggerForWeightAndResponseSimilarityCorrelationAlongTraining(EpochNum, BatchNum)
    for SaveDir in SaveDirs:
        EpochIndex, BatchIndex = utils_torch.train.ParseEpochBatchFromStr(SaveDir)
        CacheSavePath = AnalysisSaveDir + "Epoch%d-Batch%d.data"%(EpochIndex, BatchIndex)
        if utils_torch.ExistsFile(CacheSavePath):
            Data = utils_torch.json.DataFile2PyObj(CacheSavePath)
            print("WightExamples-Epoch%d-Batch%d"%(EpochIndex, BatchIndex), Data.Weight[0:5])
            LoggerCorrelation.Log(
                EpochIndex, BatchIndex, Data.ResponseSimilarity, Data.Weight
            )
            continue
        utils_torch.AddLog("Testing Model at Epoch%d-Batch%d"%(EpochIndex, BatchIndex))
        logger = utils_torch.GetLogger("DataTest")
        logger.SetEpochIndex(EpochIndex)
        logger.SetBatchIndex(BatchIndex)

        utils_torch.DoTasks(
            "&^param.task.Load",
            In={"SaveDir": SaveDir}, 
            **kw
        )
        utils_torch.DoTasks(
            "&^param.task.BuildTrainer", **kw
        )
        
        _LoggerCorrelation = _AnalyzeConnectivityPattern(
            EpochIndex=EpochIndex, BatchIndex=BatchIndex, logger=logger, TestBatchNum=TestBatchNum
        )

        utils_torch.json.PyObj2DataFile(
            utils_torch.PyObj({
                "ResponseSimilarity": _LoggerCorrelation.ResponseSimilarity,
                "Weight": _LoggerCorrelation.Weight,
            }),
            CacheSavePath
        )
        LoggerCorrelation.Log(
            EpochIndex, BatchIndex, _LoggerCorrelation.ResponseSimilarity, _LoggerCorrelation.Weight
        )
    LoggerCorrelation.Plot(
        PlotNum=100, SaveDir=AnalysisSaveDir, SaveName="Recurrent.FiringRate2Output.Weight"
    )
AddAnalysisMethods.AnalyzeConnectivityPattern = AnalyzeConnectivityPattern


def _AnalyzeConnectivityPattern(**kw):
    GlobalParam = utils_torch.GetGlobalParam()
    agent = GlobalParam.object.agent
    Dataset = GlobalParam.object.image
    BatchParam = GlobalParam.param.task.Train.BatchParam
    Dataset.PrepareBatches(BatchParam, "Test")
    logger = kw.get("logger")
    TestBatchNum = kw.setdefault("TestBatchNum", 10)
    EpochIndex = kw["EpochIndex"]
    BatchIndex = kw["BatchIndex"]
    _LoggerCorrelation = utils.analysis.LoggerForWeightAndResponseSimilarityCorrelation()
    for TestBatchIndex in range(TestBatchNum):
        utils_torch.AddLog("Epoch%d-Index%d-TestBatchIndex-%d"%(EpochIndex, BatchIndex, TestBatchIndex))
        In = utils_torch.parse.ParsePyObjDynamic(
            utils_torch.PyObj([
                "&^param.task.Train.BatchParam",
                "&^param.task.Train.OptimizeParam",
                #"&^param.task.Train.NotifyEpochBatchList"
            ]),
            ObjRoot=GlobalParam
        )
        utils_torch.CallGraph(agent.Dynamics.TestRandom, In=In)

        _LoggerCorrelation.LogResponse(
            logger.GetLogByName("agent.model.FiringRates")["Value"],
            logger.GetLogByName("agent.model.FiringRates")["Value"],
        )
    _LoggerCorrelation.CalculateResponseSimilarity()  
    _LoggerCorrelation.LogWeight(logger.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"])

    return _LoggerCorrelation

RegisterExternalClassesAndMethods()

if __name__=="__main__":
    main()
    #QuickScript(Args)
from re import L
import sys
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("task", nargs="?", default="DoTasksFromFile")
parser.add_argument("-IsDebug", default=True)
# parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default=None)
parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default="./log/DoTasksFromFile-2021-10-24-03:40:38/")
parser.add_argument("-tf", "--TaskFile", dest="TaskFile", default="./task.jsonc")
# parser.add_argument("-tn", "--TaskName", dest="TaskName", default="Main")
parser.add_argument("-tn", "--TaskName", dest="TaskName", default="AddAnalysis")
Args = parser.parse_args()

TaskFilePath = Args.TaskFile
def main():
    if Args.task in ["CleanLog", "CleanLog", "cleanlog"]:
        CleanLog()
    elif Args.task in ["CleanFigure"]:
        CleanFigures()
    elif Args.task in ["DoTasksFromFile"]:
        TaskObj = utils_torch.LoadTaskFile(TaskFilePath)
        Tasks = getattr(TaskObj, Args.TaskName)
        if not Args.IsDebug:
            try: # catch all unhandled exceptions
                utils_torch.DoTasks(Tasks, ObjRoot=utils_torch.GetGlobalParam())
            except Exception:
                utils_torch.AddError(traceback.format_exc())
                raise Exception()
        else:
            utils_torch.DoTasks(Tasks, ObjRoot=utils_torch.GetGlobalParam())
    elif Args.task in ["TotalLines"]:
        utils_torch.CalculateGitProjectTotalLines()
    elif Args.task in ["QuickScript"]:
        QuickScript(Args)
    else:
        raise Exception("Inavlid Task: %s"%Args.task)

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
    Args.task = ParseMainTask(Args.task)
    utils_torch.SetGlobalParam(GlobalParam=utils.GlobalParam)
    if Args.SaveDir is not None:
        utils_torch.SetMainSaveDir(GlobalParam=utils.GlobalParam, SaveDir=Args.SaveDir)
    else:  # Create
        utils_torch.SetMainSaveDir(GlobalParam=utils.GlobalParam, Name=Args.task)
    utils_torch.SetLoggerGlobal(GlobalParam=utils.GlobalParam)
InitUtils()

import utils
import utils_torch
from utils_torch.attrs import *
utils_torch.SetGlobalParam(utils.GlobalParam)

def CleanLog():
    utils_torch.files.RemoveAllFilesAndDirs("./log/")

def CleanFigures():
    utils_torch.files.RemoveMatchedFiles("./", r".*\.png")

def QuickScript(Args):
    # Write temporary code here, and run by "python main.py quick"
    utils_torch.Datasets.CIFAR10.OriginalFiles2DataFile(
        LoadDir = "/data3/wangweifan/Datasets/CIFAR10/",
        SaveDir = "/data3/wangweifan/Datasets/CIFAR10/CIFAR10-Data",
    )
    Data = utils_torch.json.DataFile2JsonObj("/data3/wangweifan/Datasets/CIFAR10/CIFAR10-Data")
    return


def AddObjRefForParseRouters():
    ObjRefLocal = utils_torch.PyObj()
    ObjRefLocal.LogSpatialActivity = utils.model.LogSpatialActivity
    utils_torch.model.Add2ObjRefListForParseRouters(ObjRefLocal)


def RegisterExternalMethods():
    utils_torch.RegisterExternalMethods("SaveAndLoad", SaveAndLoad)
    utils_torch.RegisterExternalMethods("AnalyzeAfterBatch", AnalyzeAfterBatch)
    utils_torch.RegisterExternalMethods("AnalyzeBeforeTrain", AnalyzeBeforeTrain)
    utils_torch.RegisterExternalMethods("AddObjRefForParseRouters", AddObjRefForParseRouters)
    utils_torch.RegisterExternalMethods("AddAnalysis", AddAnalysis)
#utils_torch.RegisterExternalMethods("RegisterExternalMethods", RegisterExternalMethods)

# External Methods that will be registered into Core Objects.
def SaveAndLoad(ContextInfo):
    EpochIndex = ContextInfo["EpochIndex"]
    BatchIndex = ContextInfo["BatchIndex"]
    ContextInfo.setdefault("ObjRoot", utils_torch.GetGlobalParam())
    SaveDir = utils_torch.SetSubSaveDirEpochBatch("SavedModel", EpochIndex, BatchIndex)
    utils_torch.DoTasks(
            "&^param.task.Save", 
            In={"SaveDir": SaveDir},
            **ContextInfo
        )
    utils_torch.DoTasks(
        "&^param.task.Load",
        In={"SaveDir": SaveDir}, 
        **ContextInfo
    )

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
            SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis/",
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
    RouterTest = Trainer.Dynamics.TestEpoch
    return

def AddAnalysis(*Args, **kw):
    # Do supplementary analysis for all saved models under main save directory.
    kw.setdefault("ObjRoot", utils_torch.GetGlobalParam())
    SaveDirs = utils_torch.GetAllSubSaveDirsEpochBatch("SavedModel")
    for SaveDir in SaveDirs:
        EpochIndex, BatchIndex = utils_torch.train.ParseEpochBatchFromStr(SaveDir)
        utils_torch.AddLog("Testing Model at Epoch%d-Batch%d"%(EpochIndex, BatchIndex))
        logger = utils_torch.GetLogger("DataTest")
        logger.SetEpochIndex(EpochIndex)
        logger.SetBatchIndex(BatchIndex)

        utils_torch.DoTasks(
            "&^param.task.BuildDataset", **kw
        )
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
            ResponseB=logger.GetLogByName("agent.model.FiringRates")["Value"],
            WeightUpdate=logger.GetLogByName("MinusGrad")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
            Weight = logger.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
            SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis/",
            SaveName = "Epoch%d-Batch%d-Recurrent.FiringRate2RecurrentInput.Weight"%(EpochIndex, BatchIndex),
        )

RegisterExternalMethods()

if __name__=="__main__":
    main()
    #QuickScript(Args)
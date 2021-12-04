from re import L
import sys
import argparse
import traceback

def ParseCmdArgs():
    parser = argparse.ArgumentParser()
    # # parser.add_argument("-t", "--task", dest="task", nargs="?", default="QuickScript")
    # parser.add_argument("-t", "--task", dest="task", nargs="?", default="DoTasksFromFile")
    # # parser.add_argument("-t", "--task", dest="task", nargs="?", default="TotalLines")
    # # parser.add_argument("-t", "--task", dest="task", nargs="?", default="CopyProject2DirAndRun")

    # # if CmdArgs.task in ["CopyProject2DirAndRun"], task2 will be implemented after copy.
    # parser.add_argument("-t2", "--task2", dest="task2", default="DoTasksFromFile")
    # parser.add_argument("-id", "--IsDebug", dest="IsDebug", default=True)

    # # If Args.task in ['CopyProject2DirAndRun'], this argument will be used to designate file to be run.
    #parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default=None)
    # # parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default="./log/DoTasksFromFile-0/")
    
    # parser.add_argument("-tf", "--TaskFile", dest="TaskFile", default="./task.jsonc")

    # parser.add_argument("-tn", "--TaskName", dest="TaskName", default="Main")
    # # parser.add_argument("-tn", "--TaskName", dest="TaskName", default="AddAnalysis")

    # # If Args.task in ['CopyProject2DirAndRun'], this argument will be used to designate file to be run.
    # parser.add_argument("-ms", "--MainScript", dest="MainScript", default="main.py")

    # # parser.add_argument("-tn2", "--TaskName2", dest="TaskName2", default="AnalyzeConnectivityPattern")
    # # parser.add_argument("-tn2", "--TaskName2", dest="TaskName2", default="AnalyzePCA")
    # parser.add_argument("-tn2", "--TaskName2", dest="TaskName2", default="AnalyzePCAAndResponseWeightCorrelation")
    
    parser.add_argument("-tf", "--TaskFile", dest="TaskFile", nargs="?", default="./task.jsonc")

    CmdArgs = parser.parse_args()
    if not (CmdArgs.TaskFile.endswith(".json") or CmdArgs.TaskFile.endswith(".jsonc")):
        CmdArgs.TaskFile += ".jsonc"
    return CmdArgs
CmdArgs = ParseCmdArgs()
def ScanConfigFile(FilePath="./config.jsonc"):
    import json5
    with open(FilePath, "r") as f:
        Config = json5.load(f) # json5 allows comments. config is either dict or list.
    sys.path.append(Config["Library"]["utils_torch"]["IncludePath"])
    import utils_torch
    import utils
    utils_torch.attrs.SetAttrs(utils.GlobalParam, "config", utils_torch.PyObj(Config)) # mount config on utils_torch.GlobalParam.config
ScanConfigFile()

import utils_torch
TaskParam = utils_torch.JsonFile2PyObj(CmdArgs.TaskFile)

def main():
    global TaskParam # if there is TaskParam=... is this code block, Python will by default treat it as local variable.
    MainTask = TaskParam.MainTask # Type of main task.
    if MainTask in ["Train"]:
        # Train a model on given task, using given algorithm
        Task = GetAttrs(TaskParam.Task)
        if Task in ["ImageClassification"]:
            EnsureAttrs(TaskParam, "Task.dataset", default="cifar10")
            MainTasks = Tasks.MainTasksForImageClassification(TaskParam)
            MainTasks.DoTask()
        else:
            raise Exception(Task)
    elif MainTask in ["AddAnalysis"]:
        # Load a trained model, and do analysis
        TaskFile = utils_torch.GetMainSaveDir() + "MainTasks.jsonc"
        TaskParam = utils_torch.JsonFile2PyObj(TaskFile)
    else:
        raise Exception(MainTask)

    # utils_torch.Main(
    #     CmdArgs=CmdArgs,
    #     QuickScript=QuickScript
    # )
    utils_torch.file.PyObj2JsonFile(TaskParam, TaskParam.SaveDir + "tasks-parsed-%s.jsonc"%(utils_torch.system.GetTime()))


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
    import utils
    utils_torch.SetGlobalParam(GlobalParam=utils.GlobalParam)
    if hasattr(TaskParam, "SaveDir"):
        if not TaskParam.SaveDir.endswith("/"):
            TaskParam.SaveDir += "/"
        TaskParam.SaveDir = utils_torch.SetMainSaveDir(GlobalParam=utils.GlobalParam, SaveDir=TaskParam.SaveDir)
    else:  # Create
        TaskParam.SaveDir = utils_torch.SetMainSaveDir(GlobalParam=utils.GlobalParam, Name=TaskParam.Task)
    utils_torch.SetLogGlobal(GlobalParam=utils.GlobalParam)
    utils_torch.AddLog("Using Main Save Directory: %s"%utils_torch.GetMainSaveDir())
InitUtils()

import utils
from utils_torch.attrs import *
utils_torch.SetGlobalParam(utils.GlobalParam)
utils.GlobalParam.CmdArgs = CmdArgs

def QuickScript(Args):
    # Write temporary code here, and run "python main.py quick"
    utils_torch.file.CopyFile2AllSubDirsUnderDestDir(
        "agent.param.jsonc",
        #"log/RSLP-ReLU-Iter10-Tc0.1-1/SavedModel/Epoch-1-Batch0/",
        "./",
        "log/RSLP-ReLU-Iter10-Tc0.1-2/SavedModel/",
    )

    # utils_torch.dataset.cifar10.OriginalFiles2DataFile(
    #     LoadDir = "/data3/wangweifan/Datasets/CIFAR10/",
    #     SaveDir = "/data3/wangweifan/Datasets/CIFAR10/CIFAR10-Data",
    # )
    # Data = utils_torch.json.DataFile2JsonObj("/data3/wangweifan/Datasets/CIFAR10/CIFAR10-Data")
    # return
# CmdArgs.QuickScript = QuickScript

import Tasks
def AddObjRefForParseRouters():
    ObjRefLocal = utils_torch.PyObj()
    #ObjRefLocal.LogSpatialActivity = utils.model.LogSpatialActivity
    utils_torch.transform.Add2ObjRefListForParseRouters(ObjRefLocal)
def RegisterExternalClassesAndMethods():
    import Train
    # utils_torch.RegisterExternalMethods("SaveAndLoad", Train.SaveAndLoad)
    # utils_torch.RegisterExternalMethods("AnalyzeAfterBatch", Analyze.AnalyzeAfterBatch)
    # utils_torch.RegisterExternalMethods("AnalyzeBeforeTrain", Analyze.AnalyzeBeforeTrain)
    # utils_torch.RegisterExternalMethods("AddObjRefForParseRouters", AddObjRefForParseRouters)
    # utils_torch.RegisterExternalMethods("AddAnalysis", Analyze.AddAnalysis)
    utils_torch.RegisterExternalClasses("Trainer", Train.Trainer)
#utils_torch.RegisterExternalMethods("RegisterExternalMethods", RegisterExternalMethods)

RegisterExternalClassesAndMethods()

if __name__=="__main__":
    main()
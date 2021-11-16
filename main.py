from re import L
import sys
import argparse
import traceback

def ParseCmdArgs():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--task", dest="task", nargs="?", default="QuickScript")
    parser.add_argument("-t", "--task", dest="task", nargs="?", default="DoTasksFromFile")
    # parser.add_argument("-t", "--task", dest="task", nargs="?", default="TotalLines")
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

    # parser.add_argument("-tn2", "--TaskName2", dest="TaskName2", default="AnalyzeConnectivityPattern")
    # parser.add_argument("-tn2", "--TaskName2", dest="TaskName2", default="AnalyzePCA")
    parser.add_argument("-tn2", "--TaskName2", dest="TaskName2", default="AnalyzePCAAndConnectivityPattern")
    
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
    utils_torch.SetLogGlobal(GlobalParam=utils.GlobalParam)
    utils_torch.AddLog("Using Main Save Directory: %s"%utils_torch.GetMainSaveDir())
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

import Analyze
def AddObjRefForParseRouters():
    ObjRefLocal = utils_torch.PyObj()
    #ObjRefLocal.LogSpatialActivity = utils.model.LogSpatialActivity
    utils_torch.model.Add2ObjRefListForParseRouters(ObjRefLocal)
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
    #QuickScript(Args)
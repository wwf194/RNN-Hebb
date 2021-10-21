from re import L
import sys
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("task", nargs="?", default="DoTasksFromFile")
parser.add_argument("-IsDebug", default=True)
parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default=None)
# parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default="./log/DoTasksFromFile-2021-10-16-16:04:16/")
parser.add_argument("-tf", "--TaskFile", dest="TaskFile", default="./task.jsonc")
parser.add_argument("-tn", "--TaskName", dest="TaskName", default="Main")
# parser.add_argument("-tn", "--TaskName", dest="TaskName", default="AddAnalysis")
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

def AddObjRefForParseRouters():
    ObjRefLocal = utils_torch.PyObj()
    ObjRefLocal.LogSpatialActivity = utils.model.LogSpatialActivity
    utils_torch.model.Add2ObjRefListForParseRouters(ObjRefLocal)
utils_torch.RegisterExternalMethods("AddObjRefForParseRouters", AddObjRefForParseRouters)

def RegisterExternalMethods():
    return
utils_torch.RegisterExternalMethods("RegisterExternalMethods", RegisterExternalMethods)

def QuickScript(Args):
    # Write temporary code here, and run by "python main.py quick"
    utils_torch.Datasets.CIFAR10.OriginalFiles2DataFile(
        LoadDir = "/data3/wangweifan/Datasets/CIFAR10/",
        SaveDir = "/data3/wangweifan/Datasets/CIFAR10/CIFAR10-Data",
    )
    return

if __name__=="__main__":
    #main()
    QuickScript(Args)
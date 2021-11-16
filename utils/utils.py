import os
import time
import logging
import json5
import utils
from inspect import getframeinfo, stack

import utils_torch

GlobalParam = utils_torch.PyObj({
    "log":{}
})

def Init(SaveDirName="UnknownTask"):
    utils_torch.SetGlobalParam(GlobalParam=GlobalParam)
    utils_torch.SetSaveDir(GlobalParam=GlobalParam, Name=SaveDirName)
    utils_torch.SetLogGlobal(GlobalParam=GlobalParam)

# basic Json manipulation methods
def JsonFile2JsonObj(file_path):
    with open(file_path, "r") as f:
        json_dict = json5.load(f)
    return json_dict
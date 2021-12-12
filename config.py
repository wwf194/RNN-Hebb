


import utils_torch
# Overwrite Param Here
def OverwriteParam(agent):
    agent.OverwriteParam(
        "Modules.model.Neurons.Recurrent.Num", Value=500 # Neurons Num
    )
    agent.OverwriteParam(
        "Modules.model.Neurons.Recurrent.IsExciInhi", Value=False # Whether or not neurons are excaitatory-inhibitory
    )


def RegisterCheckPoint(Trainer):
    Trainer.RegisterCheckPoint(
        utils_torch.train.CheckPointForEpochBatchTrain().LoadParam({
            "Interval.Start":100,
        }).Build().SetMethod(
            Trainer.SaveAndLoad
        )
    )
    # "CheckPointSaveAndLoad":{
    #     "Type": "CheckPointForEpochBatchTrain",
    #     // "Epoch.Num": "$Epoch.Num",
    #     // "Batch.Num": "$Batch.Num",
    #     "Interval.Start": 100,
    #     "Method": "&~*cache.analyzer.SaveAndLoad"
    # },
    Trainer.RegisterCheckPoint(
        utils_torch.train.CheckPointForEpochBatchTrain().LoadParam({
            "CalculateCheckPointMode": "Always",
        }).Build().SetMethod(
            Trainer.cache.analyzer.AfterEveryBatch
        )
    )
    # "CheckPointAnalyze1":{
    #     "Type": "CheckPointForEpochBatchTrain",
    #     // "Epoch.Num": "$Epoch.Num",
    #     // "Batch.Num": "$Batch.Num",
    #     "CalculateCheckPointMode": "Always",
    #     "Method": "&~*cache.analyzer.AfterEveryBatch"
    # },

    Trainer.RegisterCheckPoint(
        utils_torch.train.CheckPointForEpochBatchTrain().LoadParam({
            "IntervalStart": 100,
        }).Build().SetMethod(
            Trainer.cache.analyzer.AfterBatch
        )
    )
    # "CheckPointAnalyze2":{
    #     "Type": "CheckPointForEpochBatchTrain",
    #     // "Epoch.Num": "$Epoch.Num",
    #     // "Batch.Num": "$Batch.Num",
    #     "Interval.Start": 100,
    #     "Method": "&~*cache.analyzer.AfterBatch"
    # },

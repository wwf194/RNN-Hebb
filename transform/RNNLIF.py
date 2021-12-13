
import utils_torch
class RNNLIF(utils_torch.transform.RNNLIF):
    def __init__(self, **kw):
        super().__init__(**kw)
    def LogPerformance(self, states , log:utils_torch.log.LogAlongEpochBatchTrain):
        self.LogLoss("TotalLoss", states.totalLoss, Type="Loss")
        self.LogLoss("MainLoss",  states.mainLoss,  Type="Loss")
        self.LogLoss("ActivityConstrainLoss", states.activityConstrainLoss, Type="Loss")
        self.LogLoss("WeightConstrainLoss",   states.weightConstrainLoss,   Type="Loss")
        states.outputPredicted = utils_torch.loss.Probability2MostProbableIndex
    # "LogPerformance":{
    #     "Routings":[
    #         "totalLoss,             Name=TotalLoss,             log=%log |--> &LogLoss",
    #         "mainLoss,              Name=MainLoss,              log=%log |--> &LogLoss",
    #         "activityConstrainLoss, Name=ActivityConstrainLoss, log=%log |--> &LogLoss",
    #         "weightConstrainLoss,   Name=WeightConstrainLoss,   log=%log |--> &LogLoss",
    #         "outputLast             |--> &Probability2MostProbableIndex  |--> outputPredicted",
    #         "outputPredicted,       outputTarget,               log=%log |--> &LogAccuracyForSingleClassPrediction"
    #     ]
    # },
    def BuildModules(self, IsLoad=False):
        param = self.param
        assert hasattr(param.Modules, "TransformInput") # Module that transform input:[ BatchSize, InputNum] to inputTransformed: [BatchSize, RecurrentNum]
        # Must be specified
        super().BuildModules(IsLoad=IsLoad)

        if not self.HasModule("InputList"):
            self.AddModule(
                "InputList",
                utils_torch.transform.SerialReceiver().LoadParam(Type="ActivityAlongTime")
            )
        if not self.HasModule("OutputList"):
            self.AddModule(
                "OutputList",
                utils_torch.transform.SerialReceiver().LoadParam(Type="ActivityAlongTime")
            )

        if not self.HasModule("MembranePotentialList"):
            self.AddModule(
                "MembranePotentialList",
                utils_torch.transform.SerialReceiver().LoadParam(Type="ActivityAlongTime")
            )

        if not self.HasModule("FiringRateList"):
            self.AddModule(
                "FiringRateList",
                utils_torch.transform.SerialReceiver().LoadParam(Type="ActivityAlongTime")
            )     

        if not self.HasModule("RecurrentInputList"):
            self.AddModule(
                "RecurrentInputList",
                utils_torch.transform.SerialReceiver().LoadParam(Type="ActivityAlongTime")
            )

        if not self.HasModule("InputTransformedList"):
            self.AddModule(
                "InputTransformedList",
                utils_torch.transform.SerialReceiver().LoadParam(Type="ActivityAlongTime")
            )

        if not self.HasModule("InputHolder"):
            self.AddModule(
                "InputHolder", utils_torch.module.BuildModule("SignalHolder"), Type="SignalHolder"
            )
    def LogWeight(self, log):
        plotWeight = self.GetPlotWeight()
        trainWeight = self.GetTrainWeight()
        super().LogWeight("Weight", plotWeight, Type="Weight", log=log)
        self.LogWeightStat("Weight-Stat", trainWeight, Type="Weight-Stat", log=log)
        # "LogWeight":{
        #     "Routings":[
        #         "&*GetPlotWeight |--> weightPlot",
        #         "&*GetTrainWeight |--> weightTrain",
        #         "weightPlot,  Name=Weight,      log=%log |--> &LogCache", // Log weight before updating                
        #         "weightTrain, Name=Weight-Stat, log=%log |--> &LogWeightStat",
        #     ]
        # },
    def LogActivity(self, log: utils_torch.log.LogAlongEpochBatchTrain):
        for Name, Activity in log.GetLogValueOfType("ActivityAlongTime").items():
            self.LogActivityStat(Name + "-Stat", Activity, "Activity-Stat", log)
        # "Routings":[
        #         // "outputs     |--> &GetLast |--> outputLast",
        #         // "firingRates |--> &GetLast |--> firingRatesLast",
        #         // "recurrentInputs,    Name=RecurrentInputs,    Type=ActivityAlongTime, log=%log |--> &LogCache",
        #         // "membranePotentials, Name=MembranePotentials, Type=ActivityAlongTime, log=%log |--> &LogCache",
        #         // "firingRates,        Name=FiringRates,        Type=ActivityAlongTime, log=%log |--> &LogCache",
        #         "firingRatesLast,    Name=FiringRatesLast,    Type=Activity           , log=%log |--> &LogCache",
        #         "outputs,            Name=Outputs,            Type=ActivityAlongTime, log=%log |--> &LogCache",
        #         "outputLast,         Name=OutputLast,         Type=Activity,            log=%log |--> &LogCache",
        #         //"outputTarget,     Name=OutputTarget,       Type=ActivityAlongTime, log=%log |--> &LogCache",
        #         "inputs,             Name=Inputs,             Type=ActivityAlongTime, log=%log |--> &LogCache",
        #         "inputTransformed,   Name=InputTransformed,   Type=ActivityAlongTime, log=%log |--> &LogCache",
        #         "recurrentInputs,    Name=RecurrentInputs,    Type=ActivityAlongTime-Stat, log=%log |--> &LogStat",
        #         "membranePotentials, Name=MembranePotentials, Type=ActivityAlongTime-Stat, log=%log |--> &LogStat",
        #         "firingRates,        Name=FiringRates,        Type=ActivityAlongTime-Stat, log=%log |--> &LogStat",
        #         "outputs,            Name=Outputs,            Type=ActivityAlongTime-Stat, log=%log |--> &LogStat",
        #         //"outputTarget,       Name=OutputTarget-Stat,       Type=ActivityAlongTime-Stat, log=%log |--> &LogStat",
        #         "inputs,             Name=Inputs,             Type=ActivityAlongTime-Stat, log=%log |--> &LogStat",
        #         "inputTransformed,   Name=InputTransformed,   Type=ActivityAlongTime-Stat, log=%log |--> &LogStat",
        #     ]
        # },
    def RunTrainBatch(self, Data, OptimizeParam, log):
        self.LogWeight(log)
        self.RunBatch(Data, OptimizeParam, log, IsTrain=True)
        self.LogActivity(log)
    def RunTestBatch(self, Data, OptimizeParam, log):
        self.LogWeight(log)
        self.RunBatch(Data, OptimizeParam, log, IsTrain=False)
        self.LogActivity(log)
    def RunBatch(self, Data, OptimizeParam, log:utils_torch.log.LogAlongEpochBatchTrain, IsTrain=True):
        param = self.param
        Input = Data['Input']
        OutputTarget = Data['Output']
        IterationTime = self.GetIterationTime()
        States = self.Run(
            Input, IterationTime, log
        )

        self.Optimize(States['OutputList'], OutputTarget, 
            [
                States["RecurrentInputList"],
                States["MembranePotentialList"], 
                States["FiringRateList"],
            ],
            OptimizeParam, log=log, IsTrain=IsTrain
        )

        # {
        #     "In":["input", "outputTarget", "OptimizeParam", "log"],
        #     "Out":[],
        #     "Routings":[
        #         "&LogWeight || InheritStates=True",

        #         "&GetIterationTime |--> iterationTime",
        #         "input, iterationTime, log |--> &Run |--> outputs, recurrentInputs, membranePotentials, firingRates",
        #         "recurrentInputs, membranePotentials, firingRates |--> &Merge |--> activity",
        #         "outputs, outputTarget, activity, OptimizeParam, IsTest=True, log=%log |--> &Optimize",

        #         "&LogInput.Send    |--> inputs",
        #         "&LogInputTransformed.Send  |--> inputTransformed",

        #         "&LogActivity || InheritStates=True",
                
        #     ]
        # },
    def Run(self, Input, IterationTime, log:utils_torch.log.LogAlongEpochBatchTrain):
        cache = self.cache
        Modules = self.Modules
        Dynamics = self.Dynamics
        if IterationTime is None:
            IterationTime = cache.IterationTime
        Modules.InputList.Receive(Input)
        initState = self.GenerateZeroInitState(RefInput=Input)
        #recurrentInput, membranePotential = Modules.SplitHiddenAndCellState(initState)
        
        recurrentInput = initState[:, :cache.NeuronNum] # Initial recurrentInput
        membranePotential = initState[:, cache.NeuronNum:] # Initial membranePotential

        Modules.InputHolder.Receive(Input)
        for TimeIndex in range(IterationTime):
            input = Modules.InputHolder.Send()        
            #input = Input
            Modules.InputList.Receive(input)
            inputTransformed = Modules.TransformInput(input)
            Modules.InputTransformedList.Receive(inputTransformed)
            recurrentInput, membranePotential = Dynamics.Iterate(inputTransformed, recurrentInput, membranePotential, log=log)

        outputList = Modules.OutputList.Send()
        recurrentInputList = Modules.RecurrentInputList.Send()
        membranePotentialList = Modules.MembranePotentialList.Send()
        firingRateList = Modules.FiringRateList.Send()
        inputList = Modules.InputList.Send()
        inputTransformed = Modules.InputTransformedList.Send()

        self.LogCache("OutputList", outputList, "ActivityAlongTime", log)
        self.LogCache("RecurrentInputList", recurrentInputList, "ActivityAlongTime", log)
        self.LogCache("MembranePotentialList", membranePotentialList, "ActivityAlongTime", log)
        self.LogCache("FiringRateList", firingRateList, "ActivityAlongTime", log)
        self.LogCache("FiringRateLast", firingRateList[:, -1, :], "Activity", log)
        self.LogCache("InputList", inputList, "ActivityAlongTime", log)
        self.LogCache("InputTransformed", inputTransformed, "ActivityAlongTime", log)
        
        return {
            "OutputList": outputList,
            "RecurrentInputList": recurrentInputList,
            "MembranePotentialList": membranePotentialList,   
            "FiringRateList": firingRateList
        }
        # return {
        #     "outputList": outputList
        #     # "outputSeries": outputSeries,
        #     # "recurrentInputSeries": recurrentInputSeries,
        #     # "membranePotentialSeries": membranePotentialSeries,
        #     # "firingRateSeries": firingRateSeries,
        # }
        # {
        #     "In":["input", "time", "log"],
        #     "Out":["logOutput", "logRecurrentInput", "logMembranePotential", "logFiringRate"],
        #     "Routings":[
        #         "input |--> &InputCache.Receive",
        #         "RefInput=%input |--> &*GenerateZeroInitState |--> state",  States start from zero
        #         "state |--> &SplitHiddenAndMembranePotential |--> recurrentInput, membranePotential",
        #         "recurrentInput, membranePotential |--> &Iterate |--> recurrentInput, membranePotential || repeat=%time",
        #         "&LogOutput.Send            |--> logOutput",
        #         "&LogRecurrentInput.Send    |--> logRecurrentInput",
        #         "&LogMembranePotential.Send |--> logMembranePotential",
        #         "&LogFiringRate.Send        |--> logFiringRate",
        #     ]
        # },
    def Optimize(self, outputs, outputTarget, activity, OptimizeParam, log:utils_torch.log.LogAlongEpochBatchTrain, IsTrain=False):
        Modules = self.Modules
        output = outputs[:, -1, :]
        mainLoss = Modules.PredictionLoss(output, outputTarget)
        weightConstrainLoss   = Modules.WeightConstrainLoss(activity, mainLoss)
        activityConstrainLoss = Modules.ActivityConstrainLoss(activity, mainLoss)
        totalLoss = mainLoss + weightConstrainLoss + activityConstrainLoss
        totalLoss.backward()

        trainWeight = self.GetTrainWeight()
        MinusGrad = Modules.GradientDescend(trainWeight, OptimizeParam, Update=IsTrain)
        super().LogActivity("OutputLast", output, "Activity", log)
        self.LogLossDict(
            "Loss",
            {
                "MainLoss": mainLoss,
                "ActivityConstrinLoss": activityConstrainLoss,
                "WeightConstrainLoss": weightConstrainLoss,
                "TotalLoss": totalLoss 
            },
            Type="Loss",
            log=log
        )
        super().LogGrad("MinusGrad", MinusGrad, "Grad", log)

        outputIndex = utils_torch.loss.Probability2MostProbableIndex(output)
        utils_torch.loss.LogAccuracyForSingleClassPrediction(outputIndex, outputTarget, log)
        return
        # {
        #     "In": ["outputs", "outputTarget", "activity", "OptimizeParam", "IsTest", "log"],
        #     "Routings":[
        #         "outputs |--> &GetLast |--> outputLast",
        #         "outputLast, outputTarget |--> &PredictionLoss                 |--> mainLoss",
        #         "activity, mainLoss   |--> &CalculateActivityConstrainLoss |--> activityConstrainLoss",
        #         "mainLoss |--> &CalculateWeightConstrainLoss |--> weightConstrainLoss",
        #         "mainLoss, activityConstrainLoss, weightConstrainLoss |--> &Add |--> totalLoss",
        #         "totalLoss |--> &CalculateGradient",

        #         "&*GetTrainWeight |--> trainWeight",
        #         "trainWeight,    OptimizeParam,      Update=%IsTest      |--> &GradientDescend |--> MinusGrad",
        #         "Name=MinusGrad, data=%MinusGrad, Type=Grad, log=%log |--> &LogCache",

        #         "&LogPerformance || InheritStates"
        #     ],
        # },
        # {
        #     "Routings":[
        #         "totalLoss,             Name=TotalLoss,             log=%log |--> &LogLoss",
        #         "mainLoss,              Name=MainLoss,              log=%log |--> &LogLoss",
        #         "activityConstrainLoss, Name=ActivityConstrainLoss, log=%log |--> &LogLoss",
        #         "weightConstrainLoss,   Name=WeightConstrainLoss,   log=%log |--> &LogLoss",
        #         "outputLast             |--> &Probability2MostProbableIndex  |--> outputPredicted",
        #         "outputPredicted,       outputTarget,               log=%log |--> &LogAccuracyForSingleClassPrediction"
        #     ]
        # },
    def Iterate(self, input, recurrentInput, membranePotential, log=None):
        # recurrentInput: recurrent input from last time step
        Modules = self.Modules
        recurrentInput, membranePotential, output, firingRate = Modules.RecurrentLIFNeurons(recurrentInput, membranePotential, input)
        Modules.RecurrentInputList.append(recurrentInput)
        Modules.MembranePotentialList.append(membranePotential)
        Modules.OutputList.append(output)
        Modules.FiringRateList.append(firingRate)
        return recurrentInput, membranePotential


utils_torch.module.RegisterExternalModule("transform.RNNLIF", RNNLIF)
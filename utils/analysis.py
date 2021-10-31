import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import utils_torch
from utils_torch.plot import CreateFigurePlt

class LoggerForWeightAndResponseSimilarityCorrelation:
    def __init__(self):
        self.BatchCount = 0
        self.ResponseA = []
        self.ResponseB = []
    def LogResponse(self, ResponseA, ResponseB):
        ResponseA = utils_torch.ToNpArray(ResponseA)
        ResponseB = utils_torch.ToNpArray(ResponseB)
        ResponseA = ResponseA.reshape(-1, ResponseA.shape[-1])
        ResponseB = ResponseB.reshape(-1, ResponseB.shape[-1])
        self.ResponseA.append(ResponseA)
        self.ResponseB.append(ResponseB)
    def LogWeight(self, Weight):
        Weight = utils_torch.ToNpArray(Weight)
        Weight = utils_torch.FlattenNpArray(Weight)
        self.Weight = Weight
    def CalculateResponseSimilarity(self):
        self.ResponseA = np.concatenate(self.ResponseA, axis=0)
        self.ResponseB = np.concatenate(self.ResponseB, axis=0)
        self.ResponseSimilarity = utils_torch.math.CalculatePearsonCoefficientMatrix(self.ResponseA, self.ResponseB)

class LoggerForWeightAndResponseSimilarityCorrelationAlongTraining:
    def __init__(self, EpochNum, BatchNum):
        #ConnectivityPattern = utils_torch.EmptyPyObj()
        self.EpochNum = EpochNum
        self.BatchNum = BatchNum
        self.Data = []
    def Log(self, EpochIndex, BatchIndex, ResponseSimilarity, ConnectionStrength):
        self.Data.append(utils_torch.PyObj({
            "EpochIndex": EpochIndex, 
            "BatchIndex": BatchIndex, 
            "ResponseSimilarity": ResponseSimilarity,
            "ConnectionStrength": ConnectionStrength,
        }))
        return self
    def Plot(self, PlotNum=100, SaveDir=None, SaveName=None):
        BatchNum = self.BatchNum
        self.Data.sort(key=lambda Item:Item.EpochIndex + Item.BatchIndex * 1.0 / BatchNum)

        Data = self.Data
        LogNum = len(Data)
        SampleNum = Data[0].ResponseSimilarity.shape[0]
        
        PlotIndices = utils_torch.RandomSelect(range(SampleNum), PlotNum)
        PlotNum = len(PlotIndices)

        PlotData = []
        YMins, YMaxs = [], []
        XMins, XMaxs = [], []
        for _Data in Data:
            ConnectionStrength = _Data.ConnectionStrength
            ResponseSimilarity = _Data.ResponseSimilarity
            XMin, XMax = np.nanmin(ResponseSimilarity), np.nanmax(ResponseSimilarity)
            YMin, YMax = np.nanmin(ConnectionStrength), np.nanmax(ConnectionStrength) 
            XMins.append(XMin)
            XMaxs.append(XMax)
            YMins.append(YMin)
            YMaxs.append(YMax)
        XMin, XMax, YMin, YMax = min(XMins), max(XMaxs), min(YMins), max(YMaxs)
        ImagePaths = []
        for _Data in Data:
            EpochIndex = _Data.EpochIndex
            BatchIndex = _Data.BatchIndex
            fig, ax = CreateFigurePlt()
            Title = "Weight - ResponseSimilarity : Epoch%d-Batch%d"%(EpochIndex, BatchIndex)
            XYs = np.stack(
                [
                    _Data.ResponseSimilarity[PlotIndices, :],
                    _Data.ConnectionStrength[PlotIndices, :],
                ],
                axis=1
            )
            utils_torch.plot.PlotPoints(
                ax, XYs, Color="Black", Type="Circle", Size=1.0,
                XLabel="Response Similarity", YLabel="Connection Strength", 
                Title=Title, XRange=[XMin, XMax], YRange=[YMin, YMax]
            )
            ImagePath = SaveDir + "Epoch%d-Batch%d-%s.png"%(EpochIndex, BatchIndex, SaveName)
            utils_torch.plot.SaveFigForPlt(SavePath=ImagePath)
            ImagePaths.append(ImagePath)
        utils_torch.plot.ImageFiles2GIFFile(
            ImagePaths,
            TimePerFrame=0.5, 
            SavePath=SaveDir + SaveName + ".gif"
        )
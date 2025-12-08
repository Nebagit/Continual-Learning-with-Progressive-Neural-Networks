# I implemented the blocks for our columns in the Progressive Neural Network (PNN)
# These blocks are written by me and customized for my continual learning experiments

from src.ProgNet import *
import torch.nn as nn

"""
A ProgBlock containing a single fully connected layer (nn.Linear).
Activation function can be customized; defaults to nn.ReLU.
"""
class ProgDenseBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, drop_out = 0,activation = nn.ReLU()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module = nn.Linear(inSize, outSize)
        self.dropOut = nn.Dropout(drop_out)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        self.dropOut_laterals = nn.Dropout(drop_out)
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        return self.dropOut(self.module(x))

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return self.dropOut_laterals(lat(x))

    def runActivation(self, x):
        return self.activation(x)


# Implemented an LSTM block for PNN
# Each block contains a single LSTM layer; stacking LSTMs requires multiple blocks
class ProgLSTMBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, lateralsType = 'linear',drop_out = 0):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module = nn.LSTM(input_size=inSize,hidden_size = outSize,num_layers=1,batch_first=True)
        self.dropOut = nn.Dropout(drop_out)
        self.lateralsType = lateralsType
        # Implemented two options for lateral connections between blocks:
        # 1) Linear layer
        # 2) LSTM layer
        if lateralsType == 'linear':
            self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        else:
            self.laterals = nn.ModuleList([nn.LSTM(input_size= inSize, hidden_size= outSize,num_layers=1,batch_first=True) for _ in range(numLaterals)])
            self.dropOut_laterals = nn.Dropout(0.2)  # Added dropout to reduce overfitting
        self.activation = (lambda x: x)  # Passing outputs as-is to next block

    def runBlock(self, x):
        out, (h, c) = self.module(x)
        return self.dropOut(out)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        if self.lateralsType == 'linear':
            return lat(x)
        else:
            out,(h,c) = lat(x)
            return self.dropOut_laterals(out)

    def runActivation(self, x):
        return self.activation(x)
        
        
I_FUNCTION = (lambda x : x)

"""
A ProgBlock with a single Conv2D layer (nn.Conv2d) and Batch Normalization.
Supports optional skip connections and customizable activation.
"""
class ProgConv2DBNBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals,flatten = False, activation = nn.ReLU(), layerArgs = dict(), bnArgs = dict(), skipConn = False, lambdaSkip = I_FUNCTION):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.kernSize = kernelSize
        self.flatten = flatten
        self.module = nn.Conv2d(inSize, outSize, kernelSize, **layerArgs)
        self.moduleBN = nn.BatchNorm2d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm2d(outSize, **bnArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        if self.skipConn:
            self.skipVar = x
        if self.flatten:
            return self.moduleBN(self.module(x)).view(x.shape[0], -1)
        return self.moduleBN(self.module(x))

    def runLateral(self, i, x):
        lat = self.laterals[i]
        bn = self.lateralBNs[i]
        return bn(lat(x))

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Conv2DBN"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["kernel_size"] = self.kernSize
        data["skip"] = self.skipConn
        return data

"""Three compact PyTorch sequence regression models."""
import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self,features=3,hidden=24):
        super().__init__()
        self.rnn=nn.LSTM(features,hidden,batch_first=True)
        self.head=nn.Linear(hidden,1)
    def forward(self,x):
        outputs,_=self.rnn(x)
        return self.head(outputs[:,-1]).squeeze(-1)

class TCN(nn.Module):
    def __init__(self,features=3,hidden=24):
        super().__init__()
        self.conv1=nn.Conv1d(features,hidden,kernel_size=3,dilation=1)
        self.conv2=nn.Conv1d(hidden,hidden,kernel_size=3,dilation=2)
        self.relu=nn.ReLU()
        self.head=nn.Linear(hidden,1)
    def forward(self,x):
        # Left-only padding gives causal temporal convolutions.
        z=self.relu(self.conv1(torch.nn.functional.pad(x.transpose(1,2),(2,0))))
        z=self.relu(self.conv2(torch.nn.functional.pad(z,(4,0))))
        return self.head(z[:,:,-1]).squeeze(-1)

class Transformer(nn.Module):
    def __init__(self,features=3,hidden=24,heads=4):
        super().__init__()
        self.embed=nn.Linear(features,hidden)
        self.position=nn.Parameter(torch.zeros(1,512,hidden))
        block=nn.TransformerEncoderLayer(hidden,heads,dim_feedforward=hidden*2,
                                          dropout=0.0,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,num_layers=1,enable_nested_tensor=False)
        self.head=nn.Linear(hidden,1)
    def forward(self,x):
        if x.shape[1]>512:raise ValueError("window must be <= 512")
        z=self.embed(x)+self.position[:,:x.shape[1]]
        # Entire input window is known as of forecast time; no future tokens.
        return self.head(self.encoder(z)[:,-1]).squeeze(-1)

MODELS={"LSTM":LSTM,"TCN":TCN,"Transformer":Transformer}

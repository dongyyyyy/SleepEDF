from include.header import *
from models.cnn.DeepSleepNet_cnn import *
from models.rnn.DeepSleepNet_lstm import *

class DeepSleepNet_pretrained(nn.Module):
    def __init__(self,in_channel=1,sequence_length=1,hidden_dim=512,class_num=5):
        super(DeepSleepNet_pretrained,self).__init__()
        self.featureExtract = DeepSleepNet_FE(in_channel=in_channel)
        self.dropout = nn.Dropout(p=0.5)
        self.flat = 3456

    def forward(self,input):
        out = self.featureExtract(input)
        out = out.view(-1,1,self.flat)
        out = self.dropout(out)

        return out



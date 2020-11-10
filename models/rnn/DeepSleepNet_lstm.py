from include.header import *

class BiLSTMClassification_batch(nn.Module):
    def __init__(self, input_size=3456,class_num = 5, hidden_dim = 256,num_layers=1,use_gpu=True):
        super(BiLSTMClassification_batch,self).__init__()
        self.input_size = input_size
        self.num_directions = 2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu

        self.bi_lstm1 = nn.LSTM(input_size=self.input_size, hidden_size = self.hidden_dim,batch_first=True,bidirectional=True)
        print(self.bi_lstm1)
        self.bi_lstm2 = nn.LSTM(input_size=self.hidden_dim*2, hidden_size=self.hidden_dim,batch_first=True, bidirectional=True)

        self.hidden2label = nn.Linear(hidden_dim *2, class_num)

    def init_hidden1(self,batch_size):
        #first = h(hidden) / second = c(cell)
        if self.use_gpu:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim).cuda()),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim).cuda()))# cell
        else:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim)),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim)))# cell
    def init_hidden2(self,batch_size):
        #first = h(hidden) / second = c(cell)
        if self.use_gpu:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim).cuda()),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim).cuda()))# cell
        else:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim)),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim)))# cell


    def forward(self, input):
        #print('input shape : ',input.shape)

        self.hidden1 = self.init_hidden1(input.size(0))
        #print('hidden1[0]')
        #print(self.hidden1[0].shape)
        self.hidden2 = self.init_hidden2(input.size(0))


        output, self.hidden1 = self.bi_lstm1(input,self.hidden1)
        #print('lstm1 output shape : ',output.shape)
        output, self.hidden2 = self.bi_lstm2(output,self.hidden2)
        #print('lstm2 output shape : ',output.shape)
        output = output.view(-1,self.hidden_dim*2)
        y = self.hidden2label(output)
        #print(y.shape)
        return y

class BiLSTMClassification_batch_manyToOne(nn.Module):
    def __init__(self, input_size=128,class_num = 5, hidden_dim = 256,num_layers=1,use_gpu=True):
        super(BiLSTMClassification_batch_manyToOne,self).__init__()
        self.input_size = input_size
        self.num_directions = 2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu

        self.bi_lstm1 = nn.LSTM(input_size=self.input_size, hidden_size = self.hidden_dim,batch_first=True, bidirectional=True)
        print(self.bi_lstm1)
        self.bi_lstm2 = nn.LSTM(input_size=self.hidden_dim*2, hidden_size=self.hidden_dim,batch_first=True, bidirectional=True)

        self.hidden2label = nn.Linear(hidden_dim *self.num_directions, class_num)

    def init_hidden1(self,batch_size):
        #first = h(hidden) / second = c(cell)
        if self.use_gpu:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim).cuda()),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim).cuda()))# cell
        else:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim)),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim)))# cell
    def init_hidden2(self,batch_size):
        #first = h(hidden) / second = c(cell)
        if self.use_gpu:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim).cuda()),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim).cuda()))# cell
        else:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim)),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,batch_size,self.hidden_dim)))# cell


    def forward(self, input):
        self.hidden1 = self.init_hidden1(input.size(0))
        self.hidden2 = self.init_hidden2(input.size(0))

        output, self.hidden1 = self.bi_lstm1(input,self.hidden1)
        #print('lstm1 output shape : ',output.shape)
        output, self.hidden2 = self.bi_lstm2(output,self.hidden2)
        #print('lstm2 output shape : ',output.shape)

        (hidden,cell) = self.hidden2
        print(hidden.shape)
        hidden = hidden[-self.num_directions:] # (num_directions,B,H)
        print(hidden.shape)
        hidden = torch.cat([h for h in hidden],1)

        y = self.hidden2label(hidden)

        #print(y.shape)
        return y




class BiLSTMClassification(nn.Module):
    def __init__(self, input_size=128,class_num = 5, hidden_dim = 256,num_layers=1,use_gpu=True):
        super(BiLSTMClassification,self).__init__()
        self.input_size = input_size
        self.num_directions = 2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu

        self.bi_lstm1 = nn.LSTM(input_size=self.input_size, hidden_size = self.hidden_dim, bidirectional=True)
        print(self.bi_lstm1)
        self.bi_lstm2 = nn.LSTM(input_size=self.hidden_dim*2, hidden_size=self.hidden_dim, bidirectional=True)

        self.hidden2label = nn.Linear(hidden_dim *2, class_num)
        self.hidden1 = self.init_hidden1()
        #print('hidden1[0]')
        #print(self.hidden1[0].shape)
        self.hidden2 = self.init_hidden2()
        #print('hidden2[0]')
        #print(self.hidden2[0].shape)

    def init_hidden1(self):
        #first = h(hidden) / second = c(cell)
        if self.use_gpu:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,1,self.hidden_dim).cuda()),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,1,self.hidden_dim).cuda()))# cell
        else:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,1,self.hidden_dim)),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,1,self.hidden_dim)))# cell
    def init_hidden2(self):
        #first = h(hidden) / second = c(cell)
        if self.use_gpu:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,1,self.hidden_dim).cuda()),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,1,self.hidden_dim).cuda()))# cell
        else:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,1,self.hidden_dim)),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,1,self.hidden_dim)))# cell


    def forward(self, input):
        #print('input shape : ',input.shape)

        self.hidden1 = self.init_hidden1()
        #print('hidden1[0]')
        #print(self.hidden1[0].shape)
        self.hidden2 = self.init_hidden2()

        output, self.hidden1 = self.bi_lstm1(input,self.hidden1)
        #print('lstm1 output shape : ',output.shape)
        output, self.hidden2 = self.bi_lstm2(output,self.hidden2)
        #print('lstm2 output shape : ',output.shape)
        output = output.view(-1,self.hidden_dim*2)
        y = self.hidden2label(output)
        #print(y.shape)
        return y

class BiLSTMClassification_sequence(nn.Module):
    def __init__(self, input_size=128,class_num = 5, hidden_dim = 256,num_layers=1,sequence_length=100,use_gpu=True):
        super(BiLSTMClassification_sequence,self).__init__()
        self.input_size = input_size
        self.num_directions = 2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.sequence_length = sequence_length

        print('model sequence length : %d'%self.sequence_length)

        self.bi_lstm1 = nn.LSTM(input_size=self.input_size, hidden_size = self.hidden_dim, bidirectional=True)
        print(self.bi_lstm1)
        self.bi_lstm2 = nn.LSTM(input_size=self.hidden_dim*2, hidden_size=self.hidden_dim, bidirectional=True)

        self.hidden2label = nn.Linear(hidden_dim *2, class_num)
        self.hidden1 = self.init_hidden1()
        #print('hidden1[0]')
        #print(self.hidden1[0].shape)
        self.hidden2 = self.init_hidden2()
        #print('hidden2[0]')
        #print(self.hidden2[0].shape)

    def init_hidden1(self):
        #first = h(hidden) / second = c(cell)
        if self.use_gpu:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,self.sequence_length,self.hidden_dim).cuda()),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,self.sequence_length,self.hidden_dim).cuda()))# cell
        else:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,self.sequence_length,self.hidden_dim)),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,self.sequence_length,self.hidden_dim)))# cell
    def init_hidden2(self):
        #first = h(hidden) / second = c(cell)
        if self.use_gpu:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,self.sequence_length,self.hidden_dim).cuda()),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,self.sequence_length,self.hidden_dim).cuda()))# cell
        else:
            return (Variable(torch.zeros(self.num_directions*self.num_layers,self.sequence_length,self.hidden_dim)),# hidden
                    Variable(torch.zeros(self.num_directions*self.num_layers,self.sequence_length,self.hidden_dim)))# cell


    def forward(self, input):
        #print('input shape : ',input.shape)

        self.hidden1 = self.init_hidden1()
        self.hidden2 = self.init_hidden2()

        output, self.hidden1 = self.bi_lstm1(input,self.hidden1)
        #print('lstm1 output shape : ',output.shape)
        output, self.hidden2 = self.bi_lstm2(output,self.hidden2)
        #print('lstm2 output shape : ',output.shape)
        output = output.view(-1,self.hidden_dim*2)
        y = self.hidden2label(output)
        #print(y.shape)
        return y


class Sequence_lstm(nn.Module):
    def __init__(self,in_channel=1,sequence_length=20,layer_filters=[64,128,256,512],first_conv=[200,12],hidden_dim=512,num_classes=5):
        super(Sequence_lstm, self).__init__()
        self.sequence_length = sequence_length
        self.classification = BiLSTMClassification_sequence(class_num=num_classes,input_size=layer_filters[3],hidden_dim=hidden_dim,sequence_length=sequence_length)
        self.dropout = nn.Dropout(p=0.5)
        self.flat = layer_filters[3]

    def forward(self,input):
        # (100,150)
        input = input.view(-1,1,self.flat) # batch , sequence length , input_size
        input = input.reshape(-1,self.sequence_length,self.flat)
        out = self.dropout(input)
        out = self.classification(out)

        return out
'''
# CNN을 통해서 나오는 최종 아웃풋 1Xdata_size(256)
print(BiLSTMClassification().cuda())
inputs = torch.Tensor(5,1,256).cuda()
model = BiLSTMClassification().cuda()
model(inputs)
'''
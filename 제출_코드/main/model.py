import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import math, copy, time
from torch.autograd import Variable
import joblib
from torch.nn.modules.activation import ReLU
from loss import LogManager, l2loss, calc_err

# 데이터의 shape는 (batch_size, time_step, latent_dim(dirX,dirY,linX,linY))) = (n, 100, 4) 
# 일종의 시계열 데이터 이므로 사용해볼 모델은 LSTM with Attention, Transformer 



########################
# LSTM Attention Model #
########################

class EncoderLSTM(nn.Module) :
    def __init__(self, *args, **kwargs) :
        super(EncoderLSTM, self).__init__()
        self.latent_dim = kwargs.get("latent_dim", 0)
        self.hidden_dim = kwargs.get("hidden_dim", 0)
        self.LSTM = nn.LSTM(input_size=self.latent_dim, hidden_size=self.hidden_dim)
    def forward(self, input, h, c) :
        cell_output, (hidden_state, cell_state) = self.LSTM(input, (h, c))
        return hidden_state, cell_state

class DecoderLSTM(nn.Module) :
    def __init__(self, *args, **kwargs) : 
        super(DecoderLSTM, self).__init__()
        self.latent_dim = kwargs.get("latent_dim", 0)
        self.hidden_dim = kwargs.get("hidden_dim", 0)
        self.LSTM = nn.LSTM(input_size=1, hidden_size=self.hidden_dim)
    def forward(self, input, enc_hidden_state, enc_cell_state) : 
        cell_output, (hidden_state, cell_state) = self.LSTM(input, (enc_hidden_state, enc_cell_state))
        return hidden_state, cell_state

class Attention(nn.Module) :
    def __init__(self, *args, **kwargs) :
        super(Attention, self).__init__()
        self.latent_dim = kwargs.get("latent_dim", 0)
        self.hidden_dim = kwargs.get("hidden_dim", 0)
        self.w1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.w2 = nn.Linear(self.hidden_dim, 1)
        self.tanh = nn.Tanh()
    def forward(self, x) :
        #print("x : ", x.shape)
        x = self.w1(x)
        x = self.tanh(x)
        x = self.w2(x)
        o = F.log_softmax(x, dim=2)
        return o

# posX의 변화량을 예측하는 모델
class lstm_dX(nn.Module) : 
    def __init__(self, *args, **kwargs) :
        super(lstm_dX, self).__init__()
        self.device = kwargs.get("device", "cuda")
        self.batch_size = kwargs.get("batch_size", 0)
        self.hidden_dim = kwargs.get("hidden_dim", 0)
        self.latent_dim = kwargs.get("latent_dim", 0)
        if self.device == "cuda" :
            print("*******************")
            self.enc = EncoderLSTM(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim).cuda()
            self.dec = DecoderLSTM(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim).cuda()
            self.att = Attention(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim).cuda()
        else :
            print("--------------------")
            self.enc = EncoderLSTM(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
            self.dec = DecoderLSTM(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
            self.att = Attention(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, t) : 
        '''
        (n, 100, 4) -> (n, 1)
        '''
        # 기본 input은 (batch_size, 4(embeded len), 100(seq len)), 기본 target은 (batch_size)
        # 이를 (100(seq len), batch_size, 4(embeded len)) / (1, batch_size, 1)으로 바꿔줘야 한다
        x = x.permute(2, 0, 1)
        t = t.unsqueeze(0).unsqueeze(0).permute(0, 2, 1).float()    # t는 Long() 이므로, 신경망을 통과하기 위해서 float()으로 바꾼다
        
        if self.device == "cuda" : 
            h, c = torch.zeros((1, self.batch_size, self.hidden_dim)).cuda(), torch.zeros((1, self.batch_size, self.hidden_dim)).cuda()
            enc_hidden = torch.Tensor([]).cuda()
            dec_hidden = torch.Tensor([]).cuda()
        else :
            h, c = torch.zeros((1, self.batch_size, self.hidden_dim)), torch.zeros((1, self.batch_size, self.hidden_dim))
            enc_hidden = torch.Tensor([])
            dec_hidden = torch.Tensor([])
        
        for i in range(len(x)) :
            h, c = self.enc(x[i].unsqueeze(0), h, c)                # (1, batch_size, hidden_dim)
            enc_hidden = torch.cat((enc_hidden, h))                 # (100, batch_size, hidden_dim)
            
        enc_hidden = enc_hidden.permute(1, 0, 2)                    # (batch_size, 100, hidden_dim)

        h, c = self.dec(t, h, c)                                    # (1, batch_size, hidden_dim)
        dec_hidden = torch.cat((dec_hidden, h))                     # (1, batch_size, hidden_dim)
        
        dec_hidden = dec_hidden.permute(1, 0, 2)                    # (batch_size, 1, hidden_dim)
        #print("dec_hidden : ", dec_hidden.shape)
        
        attention_score = torch.matmul(enc_hidden, dec_hidden.permute(0, 2, 1)) # (batch_size, 100, 1)
        attention_score = self.softmax(attention_score)
        #print("att score : ", attention_score.shape)
        
        attention_value = enc_hidden * attention_score              # (batch_size, 100, hidden_dim)
        #print("att value1 : ", attention_value.shape)
        attention_value = torch.sum(attention_value, dim=1)         # (batch_size, hidden_dim)
        #print("att value2 : ", attention_value.shape)
        
        attention_input = torch.cat((attention_value, dec_hidden.squeeze()), dim=-1)    # (batch_size, hidden_dim*2)
        #print("att input : ", attention_input.shape)
        
        output = self.att(attention_input.unsqueeze(1))
        
        return output
    
# posY의 변화량을 예측하는 모델     
class lstm_dY(nn.Module) : 
    def __init__(self, *args, **kwargs) :
        super(lstm_dY, self).__init__()
        self.device = kwargs.get("device", "cuda")
        self.batch_size = kwargs.get("batch_size", 0)
        self.hidden_dim = kwargs.get("hidden_dim", 0)
        self.latent_dim = kwargs.get("latent_dim", 0)
        if self.device == "cuda" :
            self.enc = EncoderLSTM(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim).cuda()
            self.dec = DecoderLSTM(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim).cuda()
            self.att = Attention(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim).cuda()
        else :
            self.enc = EncoderLSTM(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
            self.dec = DecoderLSTM(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
            self.att = Attention(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, t) : 
        '''
        (n, 100, 4) -> (n, 1)
        '''
        # 기본 input은 (batch_size, 4(embeded len), 100(seq len)), 기본 target은 (batch_size)
        # 이를 (100(seq len), batch_size, 4(embeded len)) / (1, batch_size, 1)으로 바꿔줘야 한다
        x = x.permute(2, 0, 1)
        t = t.unsqueeze(0).unsqueeze(0).permute(0, 2, 1).float()    # t는 Long() 이므로, 신경망을 통과하기 위해서 float()으로 바꾼다
        
        if self.device == "cuda" : 
            h, c = torch.zeros((1, self.batch_size, self.hidden_dim)).cuda(), torch.zeros((1, self.batch_size, self.hidden_dim)).cuda()
            enc_hidden = torch.Tensor([]).cuda()
            dec_hidden = torch.Tensor([]).cuda()
        else :
            h, c = torch.zeros((1, self.batch_size, self.hidden_dim)), torch.zeros((1, self.batch_size, self.hidden_dim))
            enc_hidden = torch.Tensor([])
            dec_hidden = torch.Tensor([])
        
        for i in range(len(x)) :
            h, c = self.enc(x[i].unsqueeze(0), h, c)                # (1, batch_size, hidden_dim)
            enc_hidden = torch.cat((enc_hidden, h))                 # (100, batch_size, hidden_dim)
            
        enc_hidden = enc_hidden.permute(1, 0, 2)                    # (batch_size, 100, hidden_dim)
        h, c = self.dec(t, h, c)                                    # (1, batch_size, hidden_dim)
        dec_hidden = torch.cat((dec_hidden, h))                     # (1, batch_size, hidden_dim)
        
        dec_hidden = dec_hidden.permute(1, 0, 2)                    # (batch_size, 1, hidden_dim)
        
        attention_score = torch.matmul(enc_hidden, dec_hidden.permute(0, 2, 1)) # (batch_size, 100, 1)
        attention_score = self.softmax(attention_score)
        
        attention_value = enc_hidden * attention_score              # (batch_size, 100, hidden_dim)
        attention_value = torch.sum(attention_value, dim=1)         # (batch_size, hidden_dim)
        
        attention_input = torch.cat((attention_value, dec_hidden.squeeze()), dim=-1)    # (batch_size, hidden_dim*2)
        output = self.att(attention_input.unsqueeze(1))
        
        return output



##################################
# Dual Stage Attention Based RNN #
##################################
# 위의 LSTM Attention의 output이 무조건 0만 나오는 현상이 있어 다른 모델을 사용해본다
# 참고 : https://arxiv.org/abs/1704.02971, https://simpling.tistory.com/12

class InputAttention(nn.Module) :
    def __init__(self, *args, **kwargs) : 
        super(InputAttention, self).__init__()
        self.time_step = kwargs.get("time_step", 100)
        self.hidden_dim = kwargs.get("hidden_dim", 5)
        self.latent_dim = kwargs.get("latent_dim", 4)
        self.device = kwargs.get("device", "cuda")
        self.tanh = nn.Tanh()
        self.w1 = nn.Linear(self.time_step, self.time_step)
        self.w2 = nn.Linear(self.hidden_dim*2, self.time_step)
        self.v = nn.Linear(self.time_step, 1)
    
    def forward(self, x, h, c) : 
        '''
        x : (batch_size, latent_dim(4), time_step(100))
        '''
        if self.device == "cuda" : 
            query = torch.Tensor([]).cuda()
        else :
            query = torch.Tensor([]).cuda()
        tmp = torch.cat((h, c), dim=-1)                     # (1, batch, hidden_dim*2)                         # (batch, 1, hidden_dim*2)
        for i in range(self.latent_dim) : 
            query = torch.cat((query, tmp))                 # (latent_dim, batch, hidden_dim*2) 
            
        query = query.permute(1, 0, 2)                      # (batch, latent_dim, hidden_dim*2)
        #print("before : ", tmp.shape, x.shape, query.shape)
        #print("after : ", self.w1(x).shape, self.w2(query).shape)
        score = self.w1(x) + self.w2(query)
        score = self.tanh(score)                            # (batch, latent_dim, time_step)
        score = self.v(score)                               # (batch, latent_dim, 1)
        score = score.permute(0, 2, 1)                      # (batch, 1, latent_dim)
        attention_weight = F.log_softmax(score, dim=-1)         
        
        return attention_weight
    
class Dsa_EncoderLSTM(nn.Module) : 
    def __init__(self, *args, **kwargs) : 
        super(Dsa_EncoderLSTM, self).__init__()
        self.time_step = kwargs.get("time_step", 100)
        self.hidden_dim = kwargs.get("hidden_dim", 5)
        self.latent_dim = kwargs.get("latent_dim", 4)
        self.batch_size = kwargs.get("batch_size", 10)
        self.device = kwargs.get("device", "cuda")  
        self.lstm = nn.LSTM(input_size=self.latent_dim, hidden_size=self.hidden_dim)
    
    def forward(self, x, h, c) : 
        '''
        x : time_step 중 t 번째 step의 input data  (batch, 1, latent_dim)
        '''      
        x = x.permute(1,0,2)
        cell_output, (h, c) = self.lstm(x, (h, c))
        
        return h, c
    
class Dsa_Encoder(nn.Module) : 
    def __init__(self, *args, **kwargs) : 
        super(Dsa_Encoder, self).__init__()
        self.time_step = kwargs.get("time_step", 100)
        self.hidden_dim = kwargs.get("hidden_dim", 5)
        self.latent_dim = kwargs.get("latent_dim", 4)
        self.batch_size = kwargs.get("batch_size", 10)
        self.device = kwargs.get("device", "cuda")  
        self.input_att = InputAttention(time_step=self.time_step, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, device=self.device)
        self.lstm = Dsa_EncoderLSTM(batch_size=self.batch_size, time_step=self.time_step, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, device=self.device)
    
    def forward(self, x, h0, c0) : 
        '''
        x : input data (batch, latent_dim(4), time_step(100))
        '''
        if self.device == "cuda" : 
            alpha_seq = torch.Tensor([]).cuda()
            enc_h = torch.Tensor([]).cuda()
        else :
            alpha_seq = torch.Tensor([])
            enc_h = torch.Tensor([])
            
        for step in range(self.time_step) : 
            d = x[:,:,step]         # (batch, latent_dim)
            d = d.unsqueeze(1)      # (batch, 1 ,latent_dim)
            
            h, c = self.lstm(d, h0, c0)
            cur_alpha = self.input_att(x, h, c)
            alpha_seq = torch.cat((alpha_seq, cur_alpha), dim=1)    # (batch, time_step, latent_dim)
            enc_h = torch.cat((enc_h, h))                           # (time_step, batch, hidden_dim)
        
        enc_h = enc_h.permute(1, 0, 2)                              # (batch, time_step, latent_dim)   
        output = torch.multiply(x.permute(0,2,1), alpha_seq)        # (batch, time_step, latent_dim)
        
        return output, enc_h
    
class TemporalAttention(nn.Module) : 
    def __init__(self, *args, **kwargs) : 
        super(TemporalAttention, self).__init__()
        self.time_step = kwargs.get("time_step", 100)
        self.hidden_dim = kwargs.get("hidden_dim", 5)
        self.latent_dim = kwargs.get("latent_dim", 4)
        self.device = kwargs.get("device", "cuda")  
        self.tanh = nn.Tanh()
        self.w1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w2 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, enc_h, h, c) : 
        '''
        enc_h : encoder의 hidden state (batch_size, time_step(100), hidden_dim(5))
        '''
        if self.device == "cuda" : 
            query = torch.Tensor([]).cuda()
        else :
            query = torch.Tensor([]).cuda()
        tmp = torch.cat((h, c), dim=-1)                     # (1, batch, hidden_dim*2)                              # (batch, 1, hidden_dim*2)
        for i in range(self.time_step) : 
            query = torch.cat((query, tmp))                 # (time_step, batch, hidden_dim*2)
        
        query = query.permute(1, 0, 2)                      # (batch, time_step, hidden_dim*2)
        score = self.w1(enc_h) + self.w2(query)
        score = self.tanh(score)                            # (batch, time_step, hidden_dim)
        score = self.v(score)                               # (batch, time_step, 1)
        attention_weights = F.log_softmax(score, dim=-1)    # (batch, time_step, 1) => encoder의 hidden state에 대한 가중치
        
        return attention_weights

class Dsa_DecoderLSTM(nn.Module) : 
    def __init__(self, *args, **kwargs) : 
        super(Dsa_DecoderLSTM, self).__init__()
        self.time_step = kwargs.get("time_step", 100)
        self.hidden_dim = kwargs.get("hidden_dim", 5)
        self.target_dim = kwargs.get("target_dim", 1)
        self.device = kwargs.get("device", "cuda")  
        self.batch_size = kwargs.get("batch_size", 10)
        self.lstm = nn.LSTM(input_size=self.target_dim, hidden_size=self.hidden_dim)
    
    def forward(self, t, h, c) : 
        '''
        t : time_step 중 i 번째 step의 target data  (batch, 1, 1)
        '''
        t = t.permute(1,0,2)
        cell_output, (h, c) = self.lstm(t, (h, c))

        return h, c

class Dsa_Decoder(nn.Module) : 
    def __init__(self, *args, **kwargs) : 
        super(Dsa_Decoder, self).__init__()
        self.time_step = kwargs.get("time_step", 100)
        self.hidden_dim = kwargs.get("hidden_dim", 5)
        self.latent_dim = kwargs.get("latent_dim", 4)
        self.device = kwargs.get("device", "cuda")  
        self.batch_size = kwargs.get("batch_size", 10)
        self.tmp_att = TemporalAttention(time_step=self.time_step, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, device=self.device)
        self.lstm = Dsa_DecoderLSTM(batch_size=self.batch_size, time_step=self.time_step, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, device=self.device)
        self.dense = nn.Linear(self.hidden_dim+1, 1)

    def forward(self, t, enc_h, h0, c0) : 
        '''
        t : 이전 step까지의 target data (batch, time_step-1)
        enc_h = encoder의 hidden state (batch, time_step, hidden_dim)
        '''
        h_s = None
        if self.device == "cuda" :
            context_vector = torch.zeros((len(t), 1, self.hidden_dim)).cuda()  # (batch, 1, hidden_dim)
            dec_h_s = torch.zeros((len(t), self.hidden_dim)).cuda()            # (batch, hidden_dim)
        else :
            context_vector = torch.zeros((len(t), 1, self.hidden_dim))  # (batch, 1, hidden_dim)
            dec_h_s = torch.zeros((len(t), self.hidden_dim))            # (batch, hidden_dim)
        
        for step in range(self.time_step - 1) : 
            d = t[:, step]                                          # (batch)
            d = d.unsqueeze(1).unsqueeze(1)                         # (batch, 1, 1)
            d = torch.cat((d, context_vector), axis=-1)             # (batch, 1, hidden_dim+1)
            d = self.dense(d)                                       # (batch, 1, 1)
            
            h_s, c_s = self.lstm(d, h0, c0)                         # (1, batch, hidden_dim)

            beta_t = self.tmp_att(enc_h, h_s, c_s)                  # (batch, time_step, 1)
            beta_t = beta_t.permute(0, 2, 1)                        # (batch, 1, time_step)
            context_vector = torch.bmm(beta_t, enc_h)               # (batch, 1, hidden_dim)
            
        return torch.cat((h_s.permute(1, 0, 2), context_vector), axis=-1)   # (batch, 1, hidden_dim*2)
        
class DSA(nn.Module) :
    def __init__(self, *args, **kwargs) :
        super(DSA, self).__init__()
        self.device = kwargs.get("device", "cuda")
        self.batch_size = kwargs.get("batch_size", 10)
        self.hidden_dim = kwargs.get("hidden_dim", 5)
        self.latent_dim = kwargs.get("latent_dim", 4)   
        self.time_step = kwargs.get("time_step", 100)  
        self.enc = Dsa_Encoder(batch_size=self.batch_size, time_step=self.time_step, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, device=self.device)
        self.dec = Dsa_Decoder(batch_size=self.batch_size, time_step=self.time_step, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, device=self.device)
        self.lstm = nn.LSTM(input_size=self.latent_dim, hidden_size=self.hidden_dim)
        self.dense1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.dense2 = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, x, d) :
        '''
        x : (batch, latent_dim, time_step)
        d : 지난 time_step-1 동안의 posX 변화값들 (batch, time_step-1, 1)
        '''
        if self.device == "cuda" :
            h0 = torch.zeros((1, self.batch_size, self.hidden_dim)).cuda()
            c0 = torch.zeros((1, self.batch_size, self.hidden_dim)).cuda()
        else :
            h0 = torch.zeros((1, self.batch_size, self.hidden_dim))
            c0 = torch.zeros((1, self.batch_size, self.hidden_dim))
        
        enc_output, enc_h = self.enc(x, h0, c0)         # (batch, time_step, latent_dim)
        #_, (enc_h, _) = self.lstm(enc_output)           # (batch, time_step, hidden_dim)
        dec_output = self.dec(d, enc_h, h0, c0)         # (batch, 1, hidden_dim*2)
        output = self.dense2(self.dense1(dec_output))   # (batch, 1, 1)
        output = output.squeeze()
        
        return output


#########################
# Simple CNN Base Model #
#########################
# Attention Base의 훈련 속도가 너무 느리므로 기다리는 동안 간단한 CNN 버전으로도 제작해보자

class Conv2d_GLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv2d_GLU, self).__init__()
        inC = kwargs.get("inC", 0)
        outC = kwargs.get("outC", 0)
        k = kwargs.get("k", 0)
        s = kwargs.get("s", 0)
        p = kwargs.get("p", 0)
        T = kwargs.get("transpose", False)
        # style_dim = kwargs.get("style_dim", 0)
    
        self.T = T

        if T:   
            self.cnn = nn.ConvTranspose2d(inC, outC, kernel_size=k, stride=s, padding=p)
            self.gate = nn.ConvTranspose2d(inC, outC, kernel_size=k, stride=s, padding=p)
            self.cnn_norm = nn.InstanceNorm2d(outC)
            self.gate_norm = nn.InstanceNorm2d(outC)
            self.sig = nn.Sigmoid()

        else:
            self.cnn = nn.Conv2d(inC, outC, kernel_size=k, stride=s, padding=p)
            self.gate = nn.Conv2d(inC, outC, kernel_size=k, stride=s, padding=p)
            self.cnn_norm = nn.InstanceNorm2d(outC)
            self.gate_norm = nn.InstanceNorm2d(outC)
            self.sig = nn.Sigmoid()
            
    def forward(self, x):
        
        h1 = self.cnn_norm(self.cnn(x))
        h2 = self.gate_norm(self.gate(x))
        out = torch.mul(h1, self.sig(h2))        
        return out

class simpleCNN(nn.Module) : 
    def __init__(self, *args, **kwargs) : 
        super(simpleCNN, self).__init__()
        self.device = kwargs.get("device", "cuda")
        
        self.conv1 = Conv2d_GLU(inC=3, outC=12, k=(3,3), s=(1,1), p=(0,0))
        self.conv2 = Conv2d_GLU(inC=12, outC=48, k=(3,3), s=(1,1), p=(0,0))
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=(2,2), stride=(1,1), padding=(0,0))
        self.dense1 = nn.Linear(192, 32)
        self.dense2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x) :
        '''
        (6, 6, 3) -> (4, 4, 12) -> (2, 2, 48) -> (1, 1, 192)
            -> (192) -> (32) -> (1)
        '''    
        x = x.permute(0, 3, 1, 2)   
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.conv3(x))
        x = x.squeeze(dim=-1)    # (n, 192, 1)
        x = x.squeeze(dim=-1)    # (n, 192)
        x = self.relu(self.dense1(x))
        o = self.dense2(x)
        
        return o



#############################################
# Transformer Encoder base Regression Model #
#############################################
# transformer 전체 구조를 이용하지 말고, 데이터 분석에 사용되는 encoder layer만 이용해보자
# 아주 가벼운 encoder layer 5개를 만들고, 각 layer에는 prev_dX(or prev_dY), dirX, dirY, linX, linY 의 시계열 데이터가 들어간다
# 그리고 encoder layer들의 output을 하나의 텐서로 묶은 후에 간단한 fc layer를 통과시켜 next step의 dX(or dY)을 예측하도록 하자
# 이 아이디어는 https://doheon.github.io/%EC%BD%94%EB%93%9C%EA%B5%AC%ED%98%84/time-series/ci-4.transformer-post/#conclusion 를 참고했다

class transformer_layer(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dropout=0.1):
        super(transformer_layer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.extract = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)   # (batch, seq_len, d_model) 
        output = self.extract(output)                                                   # (batch, seq_len, 1)
        output = output.squeeze(-1)                                                     # (batch, seq_len)   
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TF(nn.Module) : 
    def __init__(self, *args, **kwargs) : 
        super(TF, self).__init__()
        self.seq_length = kwargs.get("seq_length", 30)
        self.device = kwargs.get("device", "cuda")
        self.layer_differ = transformer_layer(4, 4, 2, 0.1)
        self.layer_dirX = transformer_layer(4, 4, 2, 0.1)   
        self.layer_dirY = transformer_layer(4, 4, 2, 0.1)   
        self.layer_linX = transformer_layer(4, 4, 2, 0.1)   
        self.layer_linY = transformer_layer(4, 4, 2, 0.1)  
        self.dense1 = nn.Linear(self.seq_length*5, (self.seq_length*5)//2)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear((self.seq_length*5)//2, 1)
                    
    def forward(self, x) :
        (prev_differ, dirX, dirY, linX, linY) = x   # 각각 (batch, seq_len, 1)
        prev_differ_mask = self.layer_differ.generate_square_subsequent_mask(prev_differ.shape[1]).to(self.device)
        dirX_mask = self.layer_dirX.generate_square_subsequent_mask(dirX.shape[1]).to(self.device)
        dirY_mask = self.layer_dirY.generate_square_subsequent_mask(dirY.shape[1]).to(self.device)
        linX_mask = self.layer_linX.generate_square_subsequent_mask(linX.shape[1]).to(self.device)
        linY_mask = self.layer_linY.generate_square_subsequent_mask(linY.shape[1]).to(self.device)
        o1 = self.layer_differ(prev_differ, prev_differ_mask)
        o2 = self.layer_dirX(dirX, dirX_mask)
        o3 = self.layer_dirY(dirY, dirY_mask)
        o4 = self.layer_linX(linX, linX_mask)
        o5 = self.layer_linY(linY, linY_mask)                   # o# : (batch, seq_len)
        
        latent_vec = torch.cat((o1,o2,o3,o4,o5), dim=-1)        # (batch, seq_len*5)
        latent_vec = self.relu(self.dense1(latent_vec))         # (batch, (seq_len*5)//2)
        output = self.dense2(latent_vec)                        # (batch, 1)
        
        return output


##############################
# Transformer with CNN Model #
##############################
# 위의 transformer base 모델이 val data에 대해서 너무 빠르게 loss가 재상승하는 구간이 발생하기에 구조를 보완하였다
# 주어진 총 5개 dimension의 data를 하나로 모아, CNN을 통하여 (batch, 30, 5) => (batch, 30, 1) 의 latent vector를 생성하였고
# 이 latent vector를 transformer layer에 통과시켜 다시 새로운 latent vector를 얻은 후,
# 마지막 latent vector를 간단한 fc layer에 통과시켜 최종 regression값을 얻도록 하였다
# 즉 CNN으로 Transformer Encoder에게 줄 데이터를 만들고, 다시 Trnasformer Encoder의 출력으로
# 최종 변화값을 예측하는 것
class TF_with_CNN(nn.Module) : 
    def __init__(self, *args, **kwargs) : 
        super(TF_with_CNN, self).__init__()
        self.seq_length = kwargs.get("seq_length", 30)
        self.device = kwargs.get("device", "cuda")
        self.conv1 = Conv2d_GLU(inC=1, outC=3, k=(3,2), s=(2,1), p=(2,0))
        self.conv2 = Conv2d_GLU(inC=3, outC=9, k=(3,2), s=(2,1), p=(2,0))
        self.conv3 = Conv2d_GLU(inC=9, outC=18, k=(5,2), s=(2,1), p=(0,0))
        self.conv4 = nn.Conv2d(in_channels=18, out_channels=30, kernel_size=(3,2), stride=(1,1), padding=(0,0))
        self.tf = transformer_layer(512, 8, 4, 0.1)
        self.dense1 = nn.Linear(self.seq_length, self.seq_length//2)
        self.dense2 = nn.Linear(self.seq_length//2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x) :
        (prev_differ, dirX, dirY, linX, linY) = x   # 각각 (batch, seq_len, 1)
        x = torch.cat((prev_differ, dirX, dirY, linX, linY), dim=-1)    # (batch, 30, 5)
        x = x.unsqueeze(dim=1)                                          # (batch, 1, 30, 5)
        x = self.conv1(x)                   # (batch, 3, 16, 4)
        x = self.conv2(x)                   # (batch, 9, 9, 3)
        x = self.conv3(x)                   # (batch, 18, 3, 2)
        x = self.relu(self.conv4(x))        # (batch, 30, 1, 1)
        x = x.view(-1, 30, 1)               # (batch, 30, 1)
        
        mask = self.tf.generate_square_subsequent_mask(x.shape[1]).to(self.device)
        l = self.tf(x, mask)                        # (batch, 30)
        l = self.relu(self.dense1(l))               # (batch, 15)
        o = self.dense2(l)                          # (batch, 1)
        
        return o
        

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask


    
########################################################################################

if __name__ == '__main__' : 
    test_input = torch.load("./test_data/x.pt")
    test_prev = torch.load("./test_data/p.pt")
    test_output = [[  2.2128, -38.2749],
        [  4.8841,  -2.7704],
        [ 21.9354,  30.7265],
        [ -3.5283,  47.5623],
        [ -3.5433,  36.6826],
        [ -3.6395, -37.9780],
        [ 12.7021,  34.5348],
        [  5.9973, -29.9521],
        [  3.7158,  47.4573],
        [-15.7814, -44.6141]]
    test_output = torch.Tensor(test_output).float().cuda()
    time_step = 100
    latent_dim = 4
    hidden_dim = 6
    batch_size = 10
    
    # Test for LSTM Attention Model
    #lstm_dx = lstm_dX(time_step=time_step, latent_dim=latent_dim, hidden_dim=hidden_dim, batch_size=batch_size)
    #lstm_dy = lstm_dY(time_step=time_step, latent_dim=latent_dim, hidden_dim=hidden_dim, batch_size=batch_size)
    #dX = lstm_dx(test_input, test_output[:,0])
    #print("lstm att dX : ", dX.shape)
    #print(dX.squeeze())
    #dY = lstm_dy(test_input, test_output[:,1])
    #print("lstm att dY : ", dY.shape)
    #print(dX.squeeze())
    
    # Test for DSA Model
    #dsa_dx = DSA(batch_size=batch_size, hidden_dim=hidden_dim, latent_dim=latent_dim, time_step=time_step, device="cuda")
    #dsa_dx.cuda()
    #dX = dsa_dx(test_input, test_prev[:,:,0])
    #print(dX.shape)
    #print(dX.squeeze())
    #print(test_output[:,0])
    #print(dX.shape)
    
    # Test for SimpleCNN
    # test_img = torch.load("./test_data/img/x.pt")
    # test_img_t = torch.load("./test_data/img/t.pt")
    # cnn = simpleCNN(device="cuda").cuda()
    # posX_scaler = joblib.load("../data/1224_img/scaler/posX_scaler.pt")
    # posY_scaler = joblib.load("../data/1224_img/scaler/posY_scaler.pt")
    # o = cnn(test_img)
    # o_ = posX_scaler.inverse_transform(o.cpu().detach().numpy().reshape(-1,1))
    # t_ = posY_scaler.inverse_transform(test_img_t[:,0].cpu().detach().numpy().reshape(-1,1))
    # print("output : {}, \ntarget : {}".format(o.squeeze(), test_img_t[:,0]))
    # print("output : {}, \ntarget : {}".format(o_.squeeze(), t_.squeeze()))
    # print(t_.shape)
    # print("{}".format(l2loss(o.squeeze(), test_img_t[:,0])))
    # print("{}".format(l2loss(torch.Tensor(o_.squeeze()), torch.Tensor(t_.squeeze()))))
    
    # Test for Transformer
    tf = TF(device="cuda", seq_length=30)
    tf2 = TF_with_CNN(device="cuda", seq_length=30)
    tf.cuda()
    tf2.cuda()
    test_dx=torch.load("./test_data/tf/dx.pt")
    test_dirx=torch.load("./test_data/tf/dirx.pt")
    test_diry=torch.load("./test_data/tf/diry.pt")
    test_linx=torch.load("./test_data/tf/linx.pt")
    test_liny=torch.load("./test_data/tf/liny.pt")
    input = (test_dx, test_dirx, test_diry, test_linx, test_liny)
    o = tf2(input)
    print(o.squeeze())
    print(o.shape)
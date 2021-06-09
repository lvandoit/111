#!/usr/bin/env python
# -*- coding: utf-8 -*-



import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy
from torch.autograd import Variable
from src.lsp import train_lsp
#import max  
batch=2
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)

def l2_norm(x1,x2):
    out=torch.dist(x1,x2)
    return out
class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        yx = self.dropout(y)
       
        y = self.w2(yx)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)
        
        fac=l2_norm(x,y)
        att=1-self.sigmoid(fac) 
        
        out = att*x + y

        return out,yx,y
    

class LinearModel(nn.Module):
    def __init__(self,
                 num_2d_coords,
                 num_3d_coords,
                 num_3d_coord,
                 linear_size,
                 num_stage,
                 p_dropout,
                 predict_scale,
                 scale_range,
                 unnorm_op,
                 unnorm_init):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        self.scale_range = scale_range
        self.unnorm_op = unnorm_op
        self.predict_scale = predict_scale

        # 2d joints
        self.input_size =  num_2d_coords
        # 3d joints
        self.output_size = num_3d_coords

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        # weights that predict the scale of the image
        self.ws = nn.Linear(self.linear_size, 1)
        # sigmoid that makes sure the resulting scale is positive
        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)
        # self.mult = nn.Parameter(torch.ones(1)*unnorm_init)
        self.mult = nn.Parameter(torch.ones(num_3d_coords)*unnorm_init)
        #自加的处理过程
        self.w3=Linear(self.linear_size,self.p_dropout)
        self.w4=nn.Linear(self.output_size,self.linear_size)
        
        #self.w5=nn.Linear(self.linear_size,self.linear_size)
        self.w6=nn.Linear(self.linear_size,1)
        self.n1=nn.Linear(2*3,self.linear_size)
        self.n2=nn.Linear(2*2,self.linear_size)
        self.n3=nn.Linear(1024*5,self.linear_size)
        self.p=nn.Linear(self.linear_size, self.input_size)
    def base(self,x,sig):
        if sig==3:
            output =self.n1(x)
        else:
            output= self.n2(x)
        output = self.batch_norm1(output)
        output = self.relu(output)
        output = self.dropout(output)
        return output
        
    def forward(self, x): 
        # pre-processing
        #将给定的姿态的按照语义信息分组(先考虑Lsp数据集)
        '''
        g1=x[:,0:2*3]
        g2=x[:,6:2*6]
        g3=x[:,2*6:2*9]
        g4=x[:,2*9:2*12]
        g5=x[:,2*12:2*14]
        out1=self.base(g1,3)
        out1,_,_=self.w3(out1)
        out2=self.base(g2,3)
        out2,_,_=self.w3(out2)
        out3=self.base(g3,3)
        out3,_,_=self.w3(out3)
        out4=self.base(g4,3)
        out4,_,_=self.w3(out4)
        out5=self.base(g5,2)
        out5,_,_=self.w3(out5)
        y1=torch.cat((out1,out2,out3,out4,out5),1)
        y1=self.n3(y1)
        '''
        #y1=torch.cat(out1+out2+out3+out4+out5)/5
        #########################################
        #print('输入',x.shape)
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y1 =self.dropout(y)
        #########################################
        '''
        fg1=torch.cat((g1,g2),1)
        fg2=torch.cat((g1,g3),1)
        fg3=torch.cat((g1,g4),1)
        fg4=torch.cat((g1,g5),1)#g1:2*3;g5:2*2
        fg5=torch.cat((g2,g3),1)
        fg6=torch.cat((g2,g4),1)
        fg7=torch.cat((g2,g5),1)#g5
        fg8=torch.cat((g3,g4),1)
        fg9=torch.cat((g3,g5),1)#g5
        fg10=torch.cat((g4,g5),1)#g5
        '''
        #########################################
        #自加处理过程
        y2,F1,F2=self.w3(y1)
        new_out=self.w2(y2)#生成的中间三维姿态[8,42]
        #添加一个分支用于生成权重参数w
        #使用了三个层的二维特征
        w1=self.w6(y1)
        w2=self.w6(F1)
        w3=self.w6(F2)
        #w1=w1/w1+w2+w3
        #w2=w2/w1+w2+w3
        #w3=w3/w1+w2+w3
        #针对身体的8个关节点做进一步优化
        y=self.w4(new_out)
        # linear layers
        for i in range(self.num_stage):
            y3,_,_ = self.linear_stages[i](y)
        #loss=nn.MSELoss()
        #print(y3.shape) 
        att1=w1*l2_norm(y1,y3)
        att2=w2*l2_norm(F1,y3)
        att3=w3*l2_norm(F2,y3)
        att1_1=1-self.sigmoid(att1)
        att2_1=1-self.sigmoid(att2)
        att3_1=1-self.sigmoid(att3)
        y=y3+att1_1*y1+att2_1*F1+att3_1*F2
        out=self.w2(y)
               
        #########################################
               
        # apply the unnormalization parameters to the output
        if self.unnorm_op:
            out =out*self.mult
            
        # predict the scale :that will multiply the poses
        scale = self.scale_range * self.sigmoid(self.ws(y))
        #########################################
        #d2_out=self.w4(out)
        #d2_out,_,_=self.w3(d2_out)
        #scale=self.p(d2_out)
        #######################################
        #return hm1, out , scale   
         
        return out , scale ,new_out


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
def cos_similar(x1,x2):
    out=torch.cosine_similarity(x1,x2)#[64]
    #print(out.shape)
    return out.unsqueeze(1)

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

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        yi = self.relu(y)
        yx = self.dropout(yi)
        #yx=yx+yi
       
        y = self.w2(yx)
        y = self.batch_norm2(y)
        yp = self.relu(y)
        y = self.dropout(yp)
        #y=yf+yp
        out = x + y

        return out, yx, y

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
        self.w3=Linear(self.linear_size,self.p_dropout)
        self.w4=nn.Linear(self.output_size,self.linear_size)
        #self.w5=nn.Linear(self.linear_size,self.linear_size)
        self.w6=nn.Linear(self.linear_size,1)
        self.p=nn.Linear(self.linear_size,self.input_size)
    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        yx = self.relu(y)
        y1 =self.dropout(yx)
        y2, x1, x2=self.w3(y1)
        y3, x4, x5=self.w3(y2)
     
        new_out=self.w2(y3)
        w1=self.w6(y1)
        w2=self.w6(x1)
        w3=self.w6(x2)
        w4=self.w6(x4)
        w5=self.w6(x5)
        y=self.w4(new_out)
        # linear layers
        for i in range(self.num_stage):
            y6,_,_ = self.linear_stages[i](y) 
        att1=w1*l2_norm(y1,y6)
        att2=w2*l2_norm(x1,y6)
        att3=w3*l2_norm(x2,y6)
        att4=w4*l2_norm(x4,y6)
        att5=w5*l2_norm(x5,y6)
        att1_1=1-self.sigmoid(att1)
        att2_1=1-self.sigmoid(att2)
        att3_1=1-self.sigmoid(att3)
        att4_1=1-self.sigmoid(att4)
        att5_1=1-self.sigmoid(att5)
        y=y6+(att1_1*y1+att2_1*x1+att3_1*x2+att4_1*x4+att5_1*x5)
        out=self.w2(y)
        
       
              
        
        #########################################
               
        # apply the unnormalization parameters to the output
        if self.unnorm_op:
            out =out*self.mult
            
        # predict the scale :that will multiply the poses
        scale = self.scale_range * self.sigmoid(self.ws(y))
        #########################################
        '''  
        hm1=torch.Tensor(batch,14,16,16).cuda()
        for i in range (batch):
            for j in range(14):
                for k in range(16):
                    hm1[i,j,:,k]=hm[i,j,:,k]*scale[0,:]
                for t in range(16):
                    hm1[i,j,t,:]=hm[i,j,t,:]*scale[1,:]
        '''
        #######################################
        #return hm1, out , scale   
         
        return out , scale ,new_out

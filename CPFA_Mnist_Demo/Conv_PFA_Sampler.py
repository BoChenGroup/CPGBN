#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:14:31 2018

@author: wangchaojie
"""

import numpy as np
import numpy.ctypeslib as npct
import matplotlib.pyplot as plt
import ctypes  
from ctypes import *

realmin = 2.2e-308

## Define ctype Input
array_2d_double = npct.ndpointer(dtype=np.double,ndim=2,flags='C')
array_1d_double = npct.ndpointer(dtype=np.double,ndim=1,flags='C')
array_int = npct.ndpointer(dtype=np.int32,ndim=0,flags='C')
ll = ctypes.cdll.LoadLibrary   

Multi_lib = ll("./libMulti_Sample.so")
Multi_lib.Multi_Sample.restype = None
Multi_lib.Multi_Sample.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int, c_int]
Multi_lib.Multi_Input.restype = None
Multi_lib.Multi_Input.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int]

Conv_Multi_lib = ll("./libConv_Multi_Sample.so")
Conv_Multi_lib.Multi_Sample.restype = None
Conv_Multi_lib.Multi_Sample.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int, c_int, c_int, c_int, c_int]

Crt_Multi_lib = ll("./libCrt_Multi_Sample.so")
Crt_Multi_lib.Crt_Multi_Sample.restype = None
Crt_Multi_lib.Crt_Multi_Sample.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int, c_int]

Crt_lib =  ll("./libCrt_Sample.so")
Crt_lib.Crt_Sample.restype = None
Crt_lib.Crt_Sample.argtypes = [array_2d_double,array_2d_double, array_2d_double, c_int,c_int]

def Conv_Aug(Kernel,Score_Shape):
    
    [K1 , K2] = Score_Shape
    [K3 , K4] = Kernel.shape
    V1 = K1 + K3 - 1
    V2 = K2 + K4 - 1
    
    ## Padding
    Kernel_Pad = np.zeros([2*V1 - K3, 2*V2 - K4])## Pad [V1 - K3, V2 - K4]
    Kernel_Pad [V1 - K3 : V1,  V2 - K4 : V2] = Kernel
    Kernel_Pad = Kernel_Pad.T
    M,N = Kernel_Pad.shape
    # Parameters
    col_extent = N - K1 + 1
    row_extent = M - K2 + 1
    
    # Get Starting block indices
    start_idx = np.arange(K2)[:,None]*N + np.arange(K1)
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take (Kernel_Pad,start_idx.ravel()[:,None] + offset_idx.ravel())
    
    return np.flip(out.T,axis=1)


def Multrnd_Matrix(X_t,Phi_t,Theta_t):
    
    V = X_t.shape[0]
    J = X_t.shape[1]
    K = Theta_t.shape[0]
    Xt_to_t1_t = np.zeros([K,J], order = 'C').astype('double')
    WSZS_t = np.zeros([V,K], order = 'C').astype('double')
    
    Multi_lib.Multi_Sample(X_t,Phi_t,Theta_t, WSZS_t, Xt_to_t1_t, V,K,J)
    
    return Xt_to_t1_t, WSZS_t  

def Conv_Multrnd_Matrix(X_t,Phi_t,Theta_t):
    
    V = X_t.shape[0]
    J = X_t.shape[1]
    K = Theta_t.shape[0]
    V1 = int(np.sqrt(V))
    K1 = int(np.sqrt(K))
    K3 = V1 - K1 + 1
    Xt_to_t1_t = np.zeros([K,J], order = 'C').astype('double')
    D_t = np.zeros([K3,K3], order = 'C').astype('double')
    
    Conv_Multi_lib.Multi_Sample(X_t,Phi_t,Theta_t, D_t, Xt_to_t1_t, V,K,J,V1,K1,K3)
    
    return Xt_to_t1_t, D_t 

def Crt_Matrix(Xt_to_t1_t, p ):
    Kt = Xt_to_t1_t.shape[0]
    J = Xt_to_t1_t.shape[1]
    Xt_t1 = np.zeros([Kt,J],order = 'C').astype('double')
    
    Crt_lib.Crt_Sample(Xt_to_t1_t, p.astype('double'), Xt_t1, Kt, J)
    return Xt_t1

def Dis_Dic(D):
    [K,K1,K2] = D.shape
    w_n = np.ceil(np.sqrt(K))
    h_n = np.ceil(K/w_n)
    weight =  w_n * K2
    height =  h_n * K1
    Dic = np.zeros([ np.int32(weight),np.int32(height)])
    count = 0
    for k1 in range(np.int32(w_n)):
        for k2 in range(np.int32(h_n)):
            Dic[ k1*K1 : (k1+1)*K1, k2*K2 : (k2+1)*K2] = D[count,:,:]
            count += 1
            if count == K:
                break
        if count == K:
                break    
    plt.figure
    plt.imshow(Dic)
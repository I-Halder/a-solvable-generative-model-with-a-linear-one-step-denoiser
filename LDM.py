#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import sklearn as sk
from sklearn import linear_model

mean_list = [10] # mean of the Gaussian distribution
std_list = [1] #std of the Gaussian distribution
n=10 # number of pixels
d=n*n # dimension of the data
num_total=10**4 #size of total data
#r=0.1 # frac of total data used for training
T=torch.tensor(2) # large time cut off for the diffusion process
chunk_size=25 # chunk size for calculating the distance - should divide both num_total and train_size
l=1*torch.exp(-2*T) # generalization parameter

print("Isotropic Gaussian of mean:"+str(float(mean_list[0])))
print("Isotropic Gaussian of standard deviation:"+str(float(std_list[0])))
print('Number of data points in the original set '+str(float(num_total)))
print('Dimension of the data '+str(float(n*n)))
print("Large time cut-off "+str(float(T.numpy())))
print("Generalization parameter "+str(float((l*torch.exp(2*T)).numpy())))
print("Chunk size for GPU computation "+str(float(chunk_size)))

num_samples_start=25 # number of train samples to start from
num_samples_end=120 # number of train samples to end at
num_samples_difference=25 # step size for training samples
num_simulation=300 # number of simulations


if torch.cuda.is_available():
  gpu_name = torch.cuda.get_device_name()
  print("GPU available:", gpu_name)
else:
  print("GPU not available")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_gaussian(num_samples,n):
        dataset=torch.zeros(num_samples, n*n)
        dataset=dataset+(torch.randn(num_samples, n*n)*std_list[0]+mean_list[0])
        data_input = dataset.reshape(num_samples, 1, n, n)
        return data_input

def fwd(data):
    fwd_data=[]
    for i in torch.arange(len(data)):
        fwd_data.append(torch.exp(-T)*data[i]-torch.sqrt(l*(1-torch.exp(-2*T)))*torch.randn(n*n)) # modified the noise term
    return fwd_data

def rev(data):
    fwd_data=[]
    for i in torch.arange(len(data)):
        fwd_data.append(torch.exp(T)*data[i]-torch.sqrt(l*(torch.exp(2*T)-1))*torch.randn(n*n)) # modified the noise term
    return fwd_data

def c_det(tensor):
    if tensor.dim() > 1:
        val= (torch.linalg.det(tensor))
    else:
        val= tensor
    if val>+torch.exp(-torch.tensor(25,device=device)):
        return val
    else:
        return torch.tensor(0.0,device=device)

def distance(mu,sigma, data,  chunk_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.tensor(data, device=device)
    N = len(data)
    epsilon = torch.tensor(std_list[0]*torch.tensor((1/N)**(1/d)), device=device)  
    pair_sums_tensor = torch.tensor(0.0, device=device)
    for i in range(0, N, chunk_size):
                data_chunk_a = data[i:i + chunk_size].unsqueeze(1)
                for j in range(0, N, chunk_size):
                    data_chunk_b = data[j:j + chunk_size].unsqueeze(0)
                    pair_tensor = ((data_chunk_a - data_chunk_b)**2)
                    pair_tensor = ((data_chunk_a - data_chunk_b)**2).reshape(chunk_size, chunk_size, d)
                    pair_sums_tensor += torch.sum(torch.exp(-torch.sum(pair_tensor, dim =2) / (4 * epsilon**2)))
    
    single_sums_tensor = torch.tensor(0.0, device=device)
    for i in range(0, N,1):
        single_sums_tensor+=torch.exp(-torch.sum((1/2)*torch.matmul(torch.matmul((mu - data[i]).t(),torch.linalg.inv(torch.matmul(sigma.t(),sigma)+epsilon**2*torch.eye(d, device=device))),(mu - data[i]))))
                
    distance1 = 1 / (torch.sqrt((2*torch.tensor(torch.pi, device=device))**1 *c_det( 2*torch.matmul(sigma.t(),sigma)))) 
    distance2= (1 / ( torch.sqrt((2*torch.tensor(torch.pi, device=device))**1) * (2*epsilon**2)**(d/2) * (N**2))) * (pair_sums_tensor) 
    distance3= - (2 /(N*torch.sqrt((2*torch.tensor(torch.pi, device=device))**1 *c_det( torch.matmul(sigma.t(),sigma)+epsilon**2*torch.eye(d, device=device))))) * (single_sums_tensor)
    distance =   distance2 + distance3
    return distance

def distance_th(mu, sigma, mean_list, std_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean_list = [torch.tensor(mean, device=device, dtype=torch.float64) for mean in mean_list]
    std_list = [torch.tensor(std, device=device, dtype=torch.float64) for std in std_list]
    epsilon = torch.tensor(std_list[0], device=device, dtype=torch.float64)  
    N = 1
    single_sums_tensor = torch.tensor(0.0, device=device, dtype=torch.float64)
    single_sums_tensor += torch.exp(-torch.sum((1/4) * torch.matmul(
        torch.matmul((mu - mean_list[0] * torch.ones(d, device=device, dtype=torch.float64)).t(),
                     torch.linalg.inv(torch.matmul(sigma.t(), sigma) + epsilon**2 * torch.eye(d, device=device, dtype=torch.float64))),
        (mu - mean_list[0] * torch.ones(d, device=device, dtype=torch.float64))
    )))
    factor1 = torch.sqrt(c_det(torch.matmul(sigma.t(), sigma))).to(torch.float64)
    factor2 = torch.sqrt((epsilon**2)**d)
    distanceINT = ((2**(d/2) * torch.sqrt(factor1 * factor2)) /
                   torch.sqrt(c_det(torch.matmul(sigma.t(), sigma) + epsilon**2 * torch.eye(d, device=device, dtype=torch.float64)))) * single_sums_tensor

    return 1 - distanceINT


def linear_generator(mean_list, std_list,n,num_total, train_size,T):
    r=train_size/num_total
    data_train=generate_gaussian(train_size,n)
    data_train_vec=data_train.reshape(train_size, n*n)
    data_original=generate_gaussian(num_total,n)
    data_original_vec=data_original.reshape(num_total, n*n)
    late_data_vec=fwd(data_train_vec)
    regr = linear_model.LinearRegression()
    results = regr.fit(late_data_vec,data_train_vec)
    beta0 = torch.tensor(regr.intercept_)
    beta1 = torch.tensor(regr.coef_)
    data_train_vec=data_train_vec.to(device)
    beta0=beta0.to(device)
    beta1=beta1.to(device)
    mu = (torch.tensor(beta0, device=device, dtype=torch.float64)+torch.matmul(beta1,torch.exp(-T)*torch.mean(data_train_vec, dim=0))).to(torch.float64)
    sigma = (torch.tensor(beta1, device=device, dtype=torch.float64)*torch.sqrt(torch.exp(-2*T)*torch.mean(torch.var(data_train_vec, dim=0))+l*(1-torch.exp(-2*T)))).to(torch.float64)
    
    print('Calculating the error between train and generated data.')
    e_TG=distance(mu,sigma,data_train_vec,  chunk_size)
    print('Calculating the error between original and generated data.')
    e_OG=distance(mu,sigma,data_original_vec,  chunk_size)
    e_OG_th=distance_th(mu, sigma, mean_list, std_list)
    return e_OG, e_TG, e_OG_th
    
e_result=[]
e_net_result=[]
range_size=torch.arange(num_samples_start,num_samples_end,num_samples_difference)
input_arr=[num for num in range_size]
for i in torch.arange(num_simulation):
    arr_e=[]
    arr_e_net=[]
    for train_size in range_size:
        print("Now working at input size "+str(train_size.item()))
        e_OG, e_TG, e_OG_th=linear_generator(mean_list, std_list,n,num_total, train_size,T)
        arr_e.append(float(e_OG_th))
        arr_e_net.append(np.sign(float(e_TG-e_OG)))
    e_net_result.append(arr_e_net)
    e_result.append(arr_e)
    for train_size in range_size:
        r=train_size/num_total
        col=int((train_size-num_samples_start)/num_samples_difference)
        print("Average Hellinger distance square between original and generated distribution at input size "+str(train_size.item()))
        c_array=np.array(e_result)[:,col:(col+1)]
        print(np.sum(c_array)/len(c_array))
        print("Probability of Delta>0 at input size "+str(train_size.item()))
        print(0.5+0.5*np.mean(np.array(e_net_result)[:,col:(col+1)]))

print('The simulation is completed.')


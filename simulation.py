#!/usr/bin/env python
# coding: utf-8




max_epoch = 10 # number of epoch to train
diffusion_steps = 200 # time steps for the diffusion process multiple of 5
batch_size = 128 # bath size for training
num_generated=10 # number of generated samples
mean_list = [0.5] # mean of different Gaussian distributions
std_list = [0.1] # std of different Gaussian distributions
weights =  [1] # weights of different Gaussian distributions
n=8 # number of pixels (multiple of 8)
chunk_cutoff=500 # max number of data points to consider for calculating the distance
num_total = 10**4 # number of total data
num_samples_start=50 # number of train samples to start from
num_samples_end=1100 # number of train samples to end at (if above 500, adjust chunk size)
num_samples_difference=100 # step size for training samples
num_simulation=500 # number of simulations


print("Gaussian mixture model with mean:",mean_list)
print("Gaussian mixture model with standard deviation:",std_list)
print("Gaussian mixture model with weights:",weights)
print('Number of data points in the original set '+str(float(num_total)))
print('Number of diffusion steps '+str(float(diffusion_steps)))
print('Number of training epoch '+str(float(max_epoch)))
print('Batch size during training '+str(float(batch_size)))
print('Number of data points generated for calculating the distances '+str(float(num_generated)))
print('Dimension of the data points '+str(float(n*n)))


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorboard
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from model import DiffusionModel



if torch.cuda.is_available():
  gpu_name = torch.cuda.get_device_name()
  print("GPU available:", gpu_name)
else:
  print("GPU not available")



def find_chunk_size(n, chunk_cutoff):
    for i in torch.arange(min(n,chunk_cutoff),0,-1):
        if n % i == 0:
            return i
    

def generate_isotropic_gmm(num_samples, dim, mean_list, std_list, weights):

    device = torch.device('cpu')
    
    weights=torch.tensor(weights,device=device, dtype=torch.float)
    mean_list=torch.tensor(mean_list,device=device)
    std_list=torch.tensor(std_list,device=device)

    n_components=len(mean_list)
    means = torch.zeros(n_components, dim, device=device)
    covariances = torch.zeros(n_components, dim, dim, device=device)
    for i in range(n_components):
        means[i]=torch.tensor(mean_list[i])*torch.ones(dim, device=device)
        covariances[i]=(std_list[i]**2)*torch.eye(dim, device=device)

    samples = torch.zeros(num_samples, dim, device=device)
    component_indices = torch.multinomial(weights, num_samples, replacement=True)

    for i in range(n_components):
        
        num_samples_i = (component_indices == i).sum().item()

        if num_samples_i > 0:
            
            samples_i =255* torch.distributions.MultivariateNormal(means[i], covariances[i]).sample((num_samples_i,))
            samples[component_indices == i] = samples_i

    return samples.reshape(num_samples, 1, n, n)


class diffSet(): 
        def __init__(self, frac, dataset):
            num_len=len(dataset)
            transform = transforms.Compose([transforms.ToTensor()])
            if frac>0.5: 
              input_range=int(num_len*frac)
              data=dataset[:input_range]
            else:
              input_range=int(num_len*(1-frac))
              data=dataset[input_range:]

            self.input_seq = ((data / 255.0) * 2.0) - 1.0
            self.dataset_len = len(data)
            self.depth = 1
            self.size = n

        def __len__(self):
            return self.dataset_len

        def __getitem__(self, item):
            return self.input_seq[item]



def pair_sums_tensor(data, epsilon,em_std):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.tensor(data, device=device)
    epsilon = torch.tensor(epsilon, device=device)
    data_shape=data.shape
    N = data_shape[0]
    d = data_shape[1]
    
    
    chunk_size = find_chunk_size(N, chunk_cutoff)
    epsilon_pair_tensor=(epsilon.expand(chunk_size,chunk_size,d))**2
    pair_sums_tensor = 0*torch.ones(d, device=device)
    for i in range(0, N, chunk_size):
                data_chunk_a = data[i:i + chunk_size].unsqueeze(1)
                
                for j in range(0, N, chunk_size):
                    data_chunk_b = data[j:j + chunk_size].unsqueeze(0)
                    pair_tensor = ((data_chunk_a - data_chunk_b)**2).reshape(chunk_size, chunk_size, d)
                    pair_sums_tensor += torch.sum(torch.exp(-pair_tensor / (4 * epsilon_pair_tensor)),dim=(0,1))
                    
    return pair_sums_tensor/ (torch.sqrt(2*(epsilon/em_std)**2) * (N**2))
    
def single_sums_tensor(data, epsilon, mu, epsilon1,em_std):
      
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      data = torch.tensor(data, device=device)
      epsilon = torch.tensor(epsilon, device=device)
      data_shape=data.shape
      N = data_shape[0]
      d = data_shape[1]
      mu = torch.tensor(mu, device=device)
      epsilon1 = torch.tensor(epsilon1, device=device)
      

      single_sums_tensor = 0*torch.ones(d, device=device)
      for i in range(0, N,1):
        single_sums_tensor+=torch.exp(-(1/2)*(mu - data[i])**2/(epsilon1**2+epsilon**2))

      return single_sums_tensor/(((epsilon**2+epsilon1**2)/em_std**2)**(1/2) * (N))    

def distance(data1,data2,em_std): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data1 = torch.tensor(data1).to(device)
    data2 = torch.tensor(data2).to(device)
    N1 = len(data1)
    N2 = len(data2)
    d=data1.shape[1]
    
    
    em_std=torch.tensor(em_std, device=device).reshape(d) 
    epsilon1 = torch.tensor((1/N1)*em_std, device=device).reshape(d)  
    epsilon2 = torch.tensor((1/N2)*em_std, device=device).reshape(d)

  

    pair_sums_tensor1=  pair_sums_tensor(data1, epsilon1,em_std)
    pair_sums_tensor2=  pair_sums_tensor(data2, epsilon2,em_std)

    
    
    cross_sums_tensor = 0*torch.ones(d, device=device)
    for i in range(0, N1,1):
        cross_sums_tensor+= (1/N1)*single_sums_tensor(data2, epsilon2, data1[i], epsilon1,em_std)
    
    cross_sums_tensor= -2* cross_sums_tensor 

    

    distance_vec = pair_sums_tensor1 + pair_sums_tensor2 + cross_sums_tensor
    return torch.mean(distance_vec/em_std)

def run_simulation(num_total, num_samples, diffusion_steps, max_epoch, batch_size, num_generated):    
    frac=0.9 
    
    data_total=generate_isotropic_gmm(num_total, n*n, mean_list, std_list, weights)
    data_input=generate_isotropic_gmm(num_samples, n*n, mean_list, std_list, weights)
    

    em_mean=torch.mean(data_total, dim=0)/255
    em_std=torch.std(data_total, dim=0)/255
     
    model = DiffusionModel(n*n, diffusion_steps, 1)
    tb_logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/",
        name = 'Gaussian data',
        version = None,
    )

    trainer = pl.Trainer(
        max_epochs=max_epoch,
        log_every_n_steps=1000,
        logger=False,
        enable_checkpointing=False
    )

    train_dataset = diffSet(frac,data_input)
    val_dataset = diffSet(1-frac,data_input)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 2, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers = 2, shuffle = True)
    
    trainer.fit(model, train_loader, val_loader) 
    train_dataset_arr=np.array(train_dataset).tolist()

    gen_dataset_arr=[]

    for i in range(num_generated):

      sample_batch_size = 1 
      gen_samples = []
      x = torch.randn((sample_batch_size, train_dataset.depth, train_dataset.size, train_dataset.size))
      sample_steps = torch.arange(model.t_range, 0, -1)
      for t in sample_steps:
          x = model.denoise_sample(x, t)
          if t % 1 == 0:
              gen_samples.append(x)

      gen_samples1 = torch.stack(gen_samples, dim = 0).moveaxis(2, 4).squeeze(-1)
      gen_samples2 = (gen_samples1.clamp(-1, 1) + 1) / 2
      img=gen_samples2[gen_samples2.shape[0]-1, 0 , :, :]
      gen_dataset_arr.append(img)
    
    g_data=torch.stack(gen_dataset_arr).reshape(len(gen_dataset_arr), n*n)
    t_data=torch.tensor(train_dataset_arr).reshape(len(train_dataset_arr), n*n)
    o_data=torch.tensor(data_total).reshape(len(data_total), n*n)

    e_OG_new=distance(g_data,o_data,em_std)
    e_TG_new=distance(g_data,t_data,em_std)

    


    return  e_OG_new, e_TG_new

e_result=[]
e_net_result=[]
err_TG=[]
err_OG=[]


range_size=torch.arange(num_samples_start,num_samples_end,num_samples_difference)
for i in torch.arange(num_simulation):
    
    arr_e=[]
    arr_e_net=[]
    arr_TG=[]
    arr_OG=[]
    
    for num_samples in range_size:
        print("Now working at input size "+str(num_samples.item()))
        e_OG, e_TG=run_simulation(num_total, num_samples, diffusion_steps, max_epoch, batch_size, num_generated)
        print("Distance between original and generated data at input size "+str(num_samples.item()))
        print(float(e_OG))
        print("Distance between train and generated data at input size "+str(num_samples.item()))
        print(float(e_TG))
        print("Net advantage is ",float(e_TG-e_OG))

        arr_TG.append(float(e_TG))
        arr_OG.append(float(e_OG))
        arr_e.append(float(e_TG-e_OG))
        arr_e_net.append(np.sign(float(e_TG-e_OG)))
    
    e_net_result.append(arr_e_net)
    e_result.append(arr_e)
    err_TG.append(arr_TG)
    err_OG.append(arr_OG)

    for num_samples in range_size:
        col=int((num_samples-num_samples_start)/num_samples_difference)
        print("Avg value of Delta so far at input size "+str(num_samples.item()))
        print(np.mean(np.array(e_result)[:,col:(col+1)]))
        print("Net probability of Delta>0 so far at input size "+str(num_samples.item()))
        print(0.5+0.5*np.mean(np.array(e_net_result)[:,col:(col+1)]))
        print("Avg distance between original and generated data at input size "+str(num_samples.item()))
        print(np.mean(np.array(err_OG)[:,col:(col+1)]))
        print("Avg distance between train and generated data at input size "+str(num_samples.item()))
        print(np.mean(np.array(err_TG)[:,col:(col+1)]))

        
print("The simulation is completed.")


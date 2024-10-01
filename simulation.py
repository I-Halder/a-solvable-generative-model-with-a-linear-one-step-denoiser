#!/usr/bin/env python
# coding: utf-8

max_epoch = 20 # number of epoch to train
batch_size = 128 # bath size for training
num_generated=10 # number of generated samples
mean_list = [0.5] # mean of different Gaussian distributions
std_list = [0.1] # std of different Gaussian distributions
n=16 # number of pixels (multiple of 8)
diffusion_steps = 10 # time steps for the diffusion process multiple of 5
steps=16 # number of steps for generating the probability distribution
num_std=5 # number of std to consider for the probability distribution

num_total = 10**5 # number of total data
num_samples_start=10 # number of train samples to start from
num_samples_end=300 # number of train samples to end at
num_samples_difference=80 # step size for training samples
num_simulation=300 # number of simulations

print("Isotropic Gaussian of mean:"+str(float(mean_list[0])))
print("Isotropic Gaussian of standard deviation:"+str(float(std_list[0])))
print('Number of data points in the original set '+str(float(num_total)))
print('Number of diffusion steps '+str(float(diffusion_steps)))
print('Number of training epoch '+str(float(max_epoch)))
print('Batch size during training '+str(float(batch_size)))
print('Number of data points generated for calculating the distances '+str(float(num_generated)))
print('Dimension of the data points '+str(float(n*n)))
print('Number of steps for calculating probability distributions '+str(float(steps)))
print('Number of standard deviation distace to consider for calculating probability distributions '+str(float(num_std)))

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

def generate_isotropic_gaussian(num_samples,n):

        generator = torch.Generator()
        samples=torch.zeros(num_samples, n * n)
        samples=samples+(torch.randn(num_samples, n * n,generator=generator)*std_list[0]+mean_list[0])
        array = 255* torch.full((num_samples,n*n),1) * samples
        data_output = array.reshape(num_samples, 1, n, n)
        return data_output

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

def prob_dis(data,i,j,x):
  pdis=0
  N=len(data)
  
  epsilon=std_list[0]*(1/N)
  for k in np.arange(N):
    mean=torch.tensor(data[k]).reshape(n,n)[i,j]
    pdis=pdis+(1/N)*(1/np.sqrt(2*np.pi*epsilon**2))*torch.exp(-((x-mean)**2) / (2*epsilon**2))
  return pdis

def prob(data):
    x_arr = torch.linspace(mean_list[0]-num_std*std_list[0], mean_list[0]+num_std*std_list[0], steps)
    prob_data=[]
    for i in np.arange(n):
        for j in np.arange(n):
            prob_pixel=[]
            for x in x_arr:
              prob_pixel.append(prob_dis(data,i,j,x))
            prob_data.append(prob_pixel)
    return prob_data

def run_simulation(num_total, num_samples, diffusion_steps, max_epoch, batch_size, num_generated):    
    frac=0.9 
    
    data_total=generate_isotropic_gaussian(num_total,n)
    data_input=generate_isotropic_gaussian(num_samples,n)
     
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
          if t % 5 == 0:
              gen_samples.append(x)

      gen_samples1 = torch.stack(gen_samples, dim = 0).moveaxis(2, 4).squeeze(-1)
      gen_samples2 = (gen_samples1.clamp(-1, 1) + 1) / 2
      img=gen_samples2[gen_samples2.shape[0]-1, 0 , :, :]
      gen_dataset_arr.append(img)



    p = torch.tensor(prob(data_total[:min(10*num_generated,num_total)]))
    q = torch.tensor(prob(gen_dataset_arr))

    e_OG=torch.sum((p.flatten()-q.flatten())**2)/(n*n*steps)


    q = torch.tensor(prob(gen_dataset_arr))
    p = torch.tensor(prob(train_dataset_arr[:min(num_generated,len(train_dataset_arr))]))

    e_TG=torch.sum((p.flatten()-q.flatten())**2)/(n*n*steps)
    

    return  e_OG, e_TG

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
        print("Average value of Delta so far at input size "+str(num_samples.item()))
        print(np.mean(np.array(e_result)[:,col:(col+1)]))
        print("Probability of Delta>0 so far at input size "+str(num_samples.item()))
        print(0.5+0.5*np.mean(np.array(e_net_result)[:,col:(col+1)]))
        print("Average distance between original and generated data at input size "+str(num_samples.item()))
        print(np.mean(np.array(err_OG)[:,col:(col+1)]))
        print("Average distance between train and generated data at input size "+str(num_samples.item()))
        print(np.mean(np.array(err_TG)[:,col:(col+1)]))
        
print("The simulation is completed.")

